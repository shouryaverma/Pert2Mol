import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

class ResidualBlock(nn.Module):
    """Residual block with normalization and dropout"""
    def __init__(self, in_dim, out_dim, dropout=0.1):
        super().__init__()
        self.main_branch = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, out_dim),
            nn.Dropout(dropout)
        )
        
        # Skip connection with projection if dimensions don't match
        self.skip = nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim)
        
    def forward(self, x):
        return self.main_branch(x) + self.skip(x)

class GeneMultiHeadAttention(nn.Module):
    """Multi-head self-attention for genes to attend to each other."""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projections for all heads at once
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x: [batch_size, num_genes, embed_dim]
        """
        B, N, D = x.shape  # B=batch, N=num_genes, D=embed_dim
        
        # Generate Q, K, V for all heads
        qkv = self.qkv_proj(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, num_heads, N, N]
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # [B, num_heads, N, head_dim]
        out = out.transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        
        return self.out_proj(out), attn_weights

class RNAEncoder(nn.Module):
    """
    Encoder for RNA expression data with real self-attention over genes.
    Genes can dynamically attend to each other based on expression context.
    """
    def __init__(self, input_dim, hidden_dims=[512, 256], output_dim=256, 
                dropout=0.1, use_gene_relations=False, num_heads=4, relation_rank=25,
                gene_embed_dim=512, num_attention_layers=1,
                use_kg=False, kg_processor=None, kg_data=None, 
                gene_to_kg_mapping=None, gene_names=None):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_gene_relations = use_gene_relations
        self.num_heads = num_heads
        self.relation_rank = relation_rank
        self.gene_embed_dim = gene_embed_dim
        self.use_kg = use_kg
        
        # ===== KG INTEGRATION =====
        if use_kg and kg_processor is not None and kg_data is not None and gene_to_kg_mapping is not None:
            # Create PrimeKGEncoder internally (same pattern as DrugEmbedding)
            kg_encoder = PrimeKGEncoder(
                node_features=kg_data['num_nodes_per_type'],
                relation_types=list(kg_data['edge_mappings'].keys()),
                hidden_dim=256,
                output_dim=128,
                num_layers=3
            )
            
            self.kg_gene_encoder = KnowledgeGraphGeneEncoder(
                kg_encoder=kg_encoder,
                gene_to_kg_mapping=gene_to_kg_mapping,
                gene_names=gene_names or [f"gene_{i}" for i in range(input_dim)],
                kg_embed_dim=128,
                output_dim=gene_embed_dim,
                kg_data=kg_data
            )
        else:
            self.kg_gene_encoder = None
        
        # Project raw gene expressions to embedding space for attention
        self.gene_embedding = nn.Sequential(
            nn.Linear(1, gene_embed_dim // 2),
            nn.SiLU(),
            nn.Linear(gene_embed_dim // 2, gene_embed_dim),
            nn.LayerNorm(gene_embed_dim)
        )

        self.gene_names = gene_names
        self.gene_relation_projection = nn.Linear(gene_embed_dim, 1)
        
        # Multi-layer self-attention for genes
        self.attention_layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': GeneMultiHeadAttention(gene_embed_dim, num_heads, dropout),
                'norm1': nn.LayerNorm(gene_embed_dim),
                'norm2': nn.LayerNorm(gene_embed_dim),
                'ffn': nn.Sequential(
                    nn.Linear(gene_embed_dim, gene_embed_dim * 2),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(gene_embed_dim * 2, gene_embed_dim),
                    nn.Dropout(dropout)
                )
            }) for _ in range(num_attention_layers)
        ])

        # Low-rank gene relations
        if use_gene_relations:
            self.gene_relation_net_base = nn.Sequential(
                nn.Linear(gene_embed_dim * input_dim, 256),
                nn.LayerNorm(256),
                nn.SiLU(),
                nn.Dropout(dropout)
            )
            self.gene_relation_factors_head = nn.Linear(256, 2 * input_dim * self.relation_rank)

        # Pooling to get sample-level representation
        self.pooling_type = 'attention'
        if self.pooling_type == 'attention':
            self.pooling_attention = nn.Sequential(
                nn.Linear(gene_embed_dim, 1),
                nn.Tanh()
            )
        
        # Final encoder layers
        pooled_dim = gene_embed_dim
        layers = []
        prev_dim = pooled_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.LayerNorm(prev_dim),
                ResidualBlock(prev_dim, hidden_dim, dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Final projection
        self.final_encoder = nn.Sequential(
            nn.LayerNorm(prev_dim),
            nn.Linear(prev_dim, output_dim),
            nn.Dropout(dropout),
            nn.LayerNorm(output_dim)
        )

    def apply_gene_relations(self, x_attended):
        """Apply learned gene-gene relationships using low-rank factorization."""
        batch_size, num_genes, embed_dim = x_attended.shape

        # Flatten attended features for relation learning
        x_flat = x_attended.view(batch_size, -1)  # [B, num_genes * embed_dim]
        
        # Get cell-specific embedding from attended gene features
        cell_embedding_for_relations = self.gene_relation_net_base(x_flat)

        # Predict parameters for U and V factor matrices
        relation_factors_params = self.gene_relation_factors_head(cell_embedding_for_relations)

        # Reshape to get U [B, G, K] and V [B, K, G] matrices per cell
        U = relation_factors_params[:, :num_genes * self.relation_rank].view(
            batch_size, num_genes, self.relation_rank
        )
        V = relation_factors_params[:, num_genes * self.relation_rank:].view(
            batch_size, self.relation_rank, num_genes
        )

        # Apply transformation to mean-pooled gene features for relations
        # gene_values = x_attended.mean(dim=-1)  # [B, G] - average over embedding dim
        gene_values = self.gene_relation_projection(x_attended).squeeze(-1)  # [B, G]
        gene_values_unsqueezed = gene_values.unsqueeze(1)  # [B, 1, G]
        temp = torch.bmm(gene_values_unsqueezed, U)  # [B, 1, K]
        gene_relations = torch.bmm(temp, V).squeeze(1)  # [B, G]
        
        # Apply relations back to attended features
        relation_weights = torch.sigmoid(gene_relations).unsqueeze(-1)  # [B, G, 1]
        return x_attended * (1 + 0.1 * relation_weights)

    def pool_gene_features(self, x):
        """Pool gene features to get sample-level representation."""
        if self.pooling_type == 'mean':
            return x.mean(dim=1)  # [B, embed_dim]
        elif self.pooling_type == 'max':
            return x.max(dim=1)[0]  # [B, embed_dim]
        elif self.pooling_type == 'attention':
            # Attention-based pooling
            attn_weights = self.pooling_attention(x)  # [B, N, 1]
            attn_weights = F.softmax(attn_weights, dim=1)
            return (x * attn_weights).sum(dim=1)  # [B, embed_dim]

    def forward(self, x):
        """
        Forward pass with optional KG enhancement.
        x: [batch_size, num_genes] - raw gene expressions
        """
        batch_size, num_genes = x.shape
        
        # Embed each gene expression individually
        x_reshaped = x.unsqueeze(-1)  # [B, G, 1]
        gene_embeds = self.gene_embedding(x_reshaped)  # [B, G, embed_dim]
        
        # ===== KG ENHANCEMENT =====
        if self.use_kg and self.kg_gene_encoder is not None:
            # Get edge dict from the gene encoder if it exists
            if hasattr(self.kg_gene_encoder, 'get_kg_edge_dict'):
                edge_index_dict = self.kg_gene_encoder.get_kg_edge_dict()
            else:
                edge_index_dict = {}
            
            # CRITICAL FIX: Only use genes for the current filtered subset
            if hasattr(self, 'gene_names') and self.gene_names:
                gene_subset = self.gene_names[:num_genes]  # Take only what we need
            else:
                gene_subset = [f"gene_{i}" for i in range(num_genes)]
            
            # Get KG-enhanced gene embeddings for ONLY the current subset
            kg_gene_embeds = self.kg_gene_encoder.get_gene_kg_embeddings(
                gene_subset=gene_subset,
                edge_index_dict=edge_index_dict
            )  # Should be [num_genes, embed_dim]
            
            # Ensure KG embeddings match the current gene count
            if kg_gene_embeds.shape[0] != num_genes:
                # If mismatch, truncate or pad to match
                if kg_gene_embeds.shape[0] > num_genes:
                    kg_gene_embeds = kg_gene_embeds[:num_genes]
                else:
                    # Pad with zeros if needed
                    padding = torch.zeros(num_genes - kg_gene_embeds.shape[0], kg_gene_embeds.shape[1], 
                                        device=kg_gene_embeds.device)
                    kg_gene_embeds = torch.cat([kg_gene_embeds, padding], dim=0)
            
            # Broadcast to batch size and add to regular embeddings
            kg_gene_embeds = kg_gene_embeds.unsqueeze(0).expand(batch_size, -1, -1)
            gene_embeds = gene_embeds + 0.3 * kg_gene_embeds
        
        # Apply multi-layer self-attention between genes
        attention_weights_all = []
        x_attended = gene_embeds
        
        for layer in self.attention_layers:
            # Self-attention with residual connection
            attn_out, attn_weights = layer['attention'](x_attended)
            x_attended = layer['norm1'](x_attended + attn_out)
            
            # Feed-forward with residual connection  
            ffn_out = layer['ffn'](x_attended)
            x_attended = layer['norm2'](x_attended + ffn_out)
            
            attention_weights_all.append(attn_weights)
        
        # Apply gene relations if enabled
        if self.use_gene_relations:
            x_attended = self.apply_gene_relations(x_attended)
        
        # Pool to get sample-level representation
        pooled_features = self.pool_gene_features(x_attended)  # [B, embed_dim]
        
        # Pass through encoder layers
        encoded_features = self.encoder(pooled_features)
        
        # Final projection
        final_embeddings = self.final_encoder(encoded_features)
        
        return final_embeddings
    
    def get_attention_weights(self, x):
        """Get attention weights for interpretability."""
        with torch.no_grad():
            batch_size, num_genes = x.shape
            
            # Embed genes
            x_reshaped = x.unsqueeze(-1)
            gene_embeds = self.gene_embedding(x_reshaped)
            gene_embeds = gene_embeds + self.gene_position_embed.unsqueeze(0)
            
            # Get attention weights from each layer
            attention_weights_all = []
            x_attended = gene_embeds
            
            for layer in self.attention_layers:
                attn_out, attn_weights = layer['attention'](x_attended)
                x_attended = layer['norm1'](x_attended + attn_out)
                ffn_out = layer['ffn'](x_attended)
                x_attended = layer['norm2'](x_attended + ffn_out)
                attention_weights_all.append(attn_weights.cpu())
            
            return attention_weights_all  # List of [B, num_heads, N, N] tensors

class ResBlock(nn.Module):
    """ResNet-style residual block for image processing"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(8, out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # Skip connection
        if in_channels != out_channels or stride > 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.GroupNorm(8, out_channels)
            )
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        return out

class ImageEncoder(nn.Module):
    """Much better replacement for your basic CNN encoder"""
    def __init__(self, img_channels=4, output_dim=256):
        super().__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(img_channels, 64, 7, stride=2, padding=3, bias=False),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        
        # Residual layers (like ResNet)
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global pooling and projection
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        self._init_weights()
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [ResBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)      # [B, 64, H/4, W/4]
        x = self.layer1(x)    # [B, 64, H/4, W/4]
        x = self.layer2(x)    # [B, 128, H/8, W/8]
        x = self.layer3(x)    # [B, 256, H/16, W/16]
        x = self.layer4(x)    # [B, 512, H/32, W/32]
        
        x = self.global_pool(x)  # [B, 512, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 512]
        return self.head(x)  # [B, output_dim]