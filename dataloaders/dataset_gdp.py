import os
import sys
import logging
import json
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from rdkit import Chem
from dataloaders.dataloader import DatasetWithDrugs, image_transform

logger = logging.getLogger(__name__)

def convert_to_aromatic_smiles(smiles):
    """Convert SMILES to aromatic notation with lowercase aromatic atoms."""
    if not smiles:
        return smiles
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Generate aromatic SMILES (kekuleSmiles=False is the default)
            aromatic_smiles = Chem.MolToSmiles(mol, kekuleSmiles=False)
            return aromatic_smiles
        else:
            return smiles
    except:
        return smiles

class RawDrugDataset(DatasetWithDrugs):
    """Dataset that handles raw PubChem CSV drug data and biological conditioning"""
    
    def __init__(self,
                metadata_control: pd.DataFrame,
                metadata_drug: pd.DataFrame,
                drug_data_path: str = None,
                raw_drug_csv_path: str = None,
                gene_count_matrix: pd.DataFrame = None,
                image_json_dict: Dict[str, List[str]] = None,
                transform=None,
                target_size=256,
                debug_mode=False,
                compound_name_label='compound',
                smiles_cache: Optional[Dict] = None,
                smiles_only: bool = False,  # Add this parameter
                **kwargs):
        
        # Load raw drug CSV data
        self.compound_name_label = compound_name_label
        self.raw_drug_df = pd.read_csv(raw_drug_csv_path)
        logger.info(f"Loaded {len(self.raw_drug_df)} raw drug entries from {raw_drug_csv_path}")
        
        # Create drug name to SMILES mapping
        self.drug_name_to_smiles = {}
        for _, row in self.raw_drug_df.iterrows():
            if pd.notna(row['canonical_smiles']) and pd.notna(row[self.compound_name_label]):
                self.drug_name_to_smiles[row[self.compound_name_label]] = convert_to_aromatic_smiles(row['canonical_smiles'])
        
        logger.info(f"Created SMILES mapping for {len(self.drug_name_to_smiles)} drugs")
        sample_drugs = list(self.drug_name_to_smiles.keys())[:3]
        print("SMILES Conversion Verification:")
        for drug_name in sample_drugs:
            converted_smiles = self.drug_name_to_smiles[drug_name]
            # Find original from CSV
            original_row = self.raw_drug_df[self.raw_drug_df[self.compound_name_label] == drug_name].iloc[0]
            original_smiles = original_row['canonical_smiles']
            
            print(f"Drug: {drug_name}")
            print(f"  Original: {original_smiles}")
            print(f"  Converted: {converted_smiles}")
            print(f"  Has lowercase: {'c' in converted_smiles or 'n' in converted_smiles or 'o' in converted_smiles}")
            print()

        if smiles_cache is None:
            self.smiles_cache = {}
        else:
            # If a cache is passed (e.g., from main process), use it but it won't be shared across workers
            self.smiles_cache = smiles_cache
            
        # Initialize parent with fallback SMILES dictionary
        super().__init__(
            metadata_control=metadata_control,
            metadata_drug=metadata_drug,
            gene_count_matrix=gene_count_matrix,
            image_json_dict=image_json_dict,
            drug_data_path=drug_data_path,
            fallback_smiles_dict=self.drug_name_to_smiles,
            enable_smiles_fallback=False,
            transform=transform,
            target_size=target_size,
            debug_mode=debug_mode,
            smiles_cache=smiles_cache,
            smiles_only=smiles_only,  # Pass this to parent
            **kwargs
        )
    
    def get_target_drug_info(self, idx):
        """Get target drug information for conditional generation"""
        treatment_sample = self.filtered_drug_metadata.iloc[idx]
        compound_name = treatment_sample['compound']
        
        drug_info = {
            'compound_name': compound_name,
            'smiles': self.drug_name_to_smiles.get(compound_name, ''),
        }
        
        # Add additional drug properties if available in raw CSV
        if compound_name in self.drug_name_to_smiles:
            drug_row = self.raw_drug_df[
                self.raw_drug_df[self.compound_name_label] == compound_name
            ]
            if not drug_row.empty:
                row = drug_row.iloc[0]
                drug_info.update({
                    'molecular_weight': row.get('molecular_weight', 0),
                    'xlogp': row.get('xlogp', 0),
                    'tpsa': row.get('tpsa', 0),
                })
        
        return drug_info
    
    def __getitem__(self, idx):
        # Get base sample with biological data
        sample = super().__getitem__(idx)
        
        # Add target drug information for conditional training
        target_drug_info = self.get_target_drug_info(idx)
        sample.update({
            'target_drug_info': target_drug_info,
            'target_smiles': target_drug_info['smiles']
        })
        
        return sample

    def _init_smiles_processor(self):
        """Initialize components for on-demand SMILES processing."""
        try:
            self.rdkit_available = True
            
            # Store processing parameters from drug_data for consistency
            if self.drug_data is not None:
                self.fingerprint_size = self.drug_data.get('preprocessing_params', {}).get('fingerprint_size', 1024)
                self.normalize_descriptors = self.drug_data.get('preprocessing_params', {}).get('normalize_descriptors', True)
            else:
                self.fingerprint_size = 1024
                self.normalize_descriptors = True
            
            if  self.drug_data is not None and ('modality_dims' in self.drug_data and self.drug_data['drug_embeddings']):
                sample_drug = next(iter(self.drug_data['drug_embeddings'].values()))
                if 'descriptors_2d' in sample_drug and hasattr(self, 'normalization_params'):
                    self.desc_mean = self.normalization_params.get('mean')
                    self.desc_std = self.normalization_params.get('std')
            else:
                # No normalization parameters available - will be computed on-demand
                self.desc_mean = None
                self.desc_std = None
                
        except ImportError:
            logger.warning("RDKit not available - SMILES fallback disabled")
            self.rdkit_available = False


def create_raw_drug_dataloader(
    metadata_control: pd.DataFrame,
    metadata_drug: pd.DataFrame,
    drug_data_path: str,
    raw_drug_csv_path: str,
    image_json_path: str = None,
    gene_count_matrix: pd.DataFrame = None,
    compound_name_label='compound',
    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
    transform=None,
    target_size: int = 256,
    use_highly_variable_genes: bool = True,
    n_top_genes: int = 2000,
    normalize: bool = True,
    zscore: bool = True,
    debug_mode: bool = False,
    debug_samples: int = 50,
    debug_cell_lines: Optional[List[str]] = None,
    debug_drugs: Optional[List[str]] = None,
    smiles_cache: Optional[Dict] = None,
    **kwargs
):
    """Create DataLoader for raw drug CSV data with biological conditioning"""
    
    # Load image paths
    if image_json_path is not None:
        with open(image_json_path, 'r') as f:
            image_json_dict = json.load(f)
    else:
        image_json_dict = {}
        logger.info("No image JSON path provided or file does not exist - proceeding without images")

    logger.info(f"drug_data_path={drug_data_path}")

    # Create dataset
    dataset = RawDrugDataset(
        metadata_control=metadata_control,
        metadata_drug=metadata_drug,
        gene_count_matrix=gene_count_matrix,
        image_json_dict=image_json_dict,
        drug_data_path=drug_data_path,
        raw_drug_csv_path=raw_drug_csv_path,
        compound_name_label=compound_name_label,
        transform=transform or image_transform,
        target_size=target_size,
        smiles_cache=smiles_cache,
        debug_mode=debug_mode,
        debug_samples=debug_samples,
        debug_cell_lines=debug_cell_lines,
        debug_drugs=debug_drugs,
        smiles_only=True,
        **kwargs
    )
    
    # Apply highly variable gene selection
    if use_highly_variable_genes:
        # Same HVG processing as in original dataloader
        import scanpy as sc
        import anndata as ad
        
        adata = ad.AnnData(
            X=dataset.gene_count_matrix.T.values,
            obs=dataset.gene_count_matrix.T.index.to_frame(),
            var=dataset.gene_count_matrix.T.columns.to_frame()
        )
        adata.layers['counts'] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.log1p(adata)
        # adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        hvg_genes = adata.var_names[adata.var['highly_variable']]
        
        if zscore:
            sc.pp.scale(adata)
        if not normalize:
            adata.X = adata.layers['counts']
            
        dataset.gene_count_matrix = adata[:, hvg_genes].to_df().T
    
    # collate function
    def conditional_collate_fn(batch):
        collated = {
            'control_transcriptomics': torch.stack([item['control_transcriptomics'] for item in batch]),
            'treatment_transcriptomics': torch.stack([item['treatment_transcriptomics'] for item in batch]),
            'control_images': torch.stack([item['control_images'] for item in batch]),
            'treatment_images': torch.stack([item['treatment_images'] for item in batch]),
            'compound_name': [item['compound_name'] for item in batch],
            'conditioning_info': [item['conditioning_info'] for item in batch],
            'target_smiles': [item['target_smiles'] for item in batch],
            'target_drug_info': [item['target_drug_info'] for item in batch]
        }
        
        # Handle drug conditions (same as original)
        drug_conditions = []
        for item in batch:
            drug_cond = item['drug_condition']
            if drug_cond.numel() == 0:
                drug_cond = torch.zeros(1047, dtype=torch.float32)
            drug_conditions.append(drug_cond)
        
        if len(set(dc.shape for dc in drug_conditions)) == 1:
            collated['drug_condition'] = torch.stack(drug_conditions)
        else:
            max_len = max(dc.shape[0] for dc in drug_conditions)
            padded_conditions = []
            for dc in drug_conditions:
                if dc.shape[0] < max_len:
                    padded = torch.zeros(max_len, dtype=torch.float32)
                    padded[:dc.shape[0]] = dc
                    padded_conditions.append(padded)
                else:
                    padded_conditions.append(dc)
            collated['drug_condition'] = torch.stack(padded_conditions)
        
        return collated
    
    # Adjust for debug mode
    if debug_mode:
        batch_size = min(batch_size, 4)
        num_workers = 0
        shuffle = False
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=conditional_collate_fn,
        pin_memory=torch.cuda.is_available()
    )

# if __name__ == '__main__':
#     dataloader = create_raw_drug_dataloader(
#         metadata_control=metadata_control,
#         metadata_drug=metadata_drug,
#         gene_count_matrix=gene_expression,
#         image_json_path=args.image_json_path,
#         drug_data_path=args.drug_data_path,
#         raw_drug_csv_path=args.raw_drug_csv_path,
#         batch_size=args.batch_size,
#         debug_mode=args.debug_mode,
#         debug_samples=50 if args.debug_mode else None,
#         compound_name_label='original_name',
#         smiles_cache=smiles_cache,
#     )