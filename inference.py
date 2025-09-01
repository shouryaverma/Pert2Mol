import torch
import torch.distributed as dist
from models import DiT_models
from dataloaders.download import find_model
from diffusion.rectified_flow import create_rectified_flow
import argparse
import pandas as pd
import numpy as np
import json
from train_autoencoder import ldmol_autoencoder
from utils import AE_SMILES_decoder, regexTokenizer, dual_rna_image_encoder, get_validity
from encoders import ImageEncoder, RNAEncoder
from dataloaders.dataset_gdp import create_raw_drug_dataloader
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, AllChem, Scaffolds
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs
import time
import math
from tqdm import tqdm
from collections import defaultdict, Counter
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import random
warnings.filterwarnings('ignore')

@torch.no_grad()
def sample_with_cfg(model, flow, shape, y_full, pad_mask,
                   cfg_scale_rna=2.0, cfg_scale_image=2.0,
                   num_steps=50, device=None):
    """Sample with compositional CFG using both modality baselines."""
    if device is None:
        device = next(model.parameters()).device
    
    batch_size = shape[0]
    x = torch.randn(*shape, device=device)
    dt = 1.0 / num_steps
    
    # Pre-compute conditioning variants
    y_rna_only = y_full.clone()
    y_rna_only[:, :, :128] = 0  # Zero out image features
    
    y_image_only = y_full.clone() 
    y_image_only[:, :, 128:] = 0  # Zero out RNA features
    
    for i in range(num_steps):
        t = torch.full((batch_size,), i * dt, device=device)
        t_discrete = (t * 999).long()
        
        with torch.no_grad():
            # Batch all predictions for efficiency
            y_batch = torch.cat([y_full, y_rna_only, y_image_only], dim=0)
            pad_mask_batch = pad_mask.repeat(3, 1)
            x_batch = x.repeat(3, 1, 1, 1)
            
            velocity_batch = model(x_batch, t_discrete.repeat(3), y=y_batch, pad_mask=pad_mask_batch)
            
            if velocity_batch.shape[1] == 2 * x.shape[1]:
                velocity_batch, _ = torch.split(velocity_batch, x.shape[1], dim=1)
            
            # Split predictions
            cond_velocity, rna_velocity, image_velocity = torch.chunk(velocity_batch, 3, dim=0)
            
            # Compositional CFG: guide each modality toward full conditioning
            velocity = (rna_velocity + 
                       cfg_scale_rna * (cond_velocity - rna_velocity) +
                       image_velocity + 
                       cfg_scale_image * (cond_velocity - image_velocity)) / 2
                
        x = x + dt * velocity 
    return x

def exact_retrieval(query_features, training_data, top_k=5, similarity_metric='cosine'):
    """Find most similar biological conditions and return corresponding molecules."""
    
    if not training_data['biological_features']:
        return [], []
    
    # Concatenate all training features
    all_training_features = torch.cat(training_data['biological_features'], dim=0)
    
    # Calculate similarities
    query_flat = query_features.flatten(1)  # [B, features]
    training_flat = all_training_features.flatten(1)  # [N, features]
    
    if similarity_metric == 'cosine':
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(query_flat.cpu(), training_flat.cpu())
    else:  # euclidean
        distances = torch.cdist(query_flat, training_flat)
        similarities = 1 / (1 + distances.cpu())
    
    # Get top-k similar conditions for each query
    retrieved_smiles = []
    similarity_scores = []
    
    for i in range(query_features.shape[0]):
        sample_similarities = similarities[i]
        top_indices = sample_similarities.argsort()[-top_k:][::-1]
        
        sample_smiles = [training_data['smiles'][idx] for idx in top_indices]
        sample_scores = [sample_similarities[idx] for idx in top_indices]
        
        retrieved_smiles.append(sample_smiles)
        similarity_scores.append(sample_scores)
    
    return retrieved_smiles, similarity_scores

def estimate_generation_confidence(model, flow, x_final, y_features, pad_mask, device):
    """Estimate confidence of generated molecules using multiple metrics."""
    
    with torch.no_grad():
        # 1. Velocity consistency - check if we're at equilibrium
        t_final = torch.ones(x_final.shape[0], device=device)
        t_discrete = (t_final * 999).long()
        final_velocity = model(x_final, t_discrete, y=y_features, pad_mask=pad_mask)
        if final_velocity.shape[1] == 2 * x_final.shape[1]:
            final_velocity, _ = torch.split(final_velocity, x_final.shape[1], dim=1)
        
        velocity_magnitude = torch.norm(final_velocity.flatten(1), dim=1)
        velocity_confidence = torch.exp(-velocity_magnitude)  # Lower velocity = higher confidence
        
        # 2. Sampling consistency - generate multiple times and check agreement
        consistency_samples = []
        for _ in range(3):
            sample = sample_with_cfg(model, flow, x_final.shape, y_features, pad_mask, 
                                   num_steps=20, device=device)  # Fewer steps for speed
            consistency_samples.append(sample)
        
        # Measure variance across samples
        sample_stack = torch.stack(consistency_samples)
        sample_variance = torch.var(sample_stack, dim=0).flatten(1).mean(1)
        consistency_confidence = torch.exp(-sample_variance)
        
        # 3. Autoencoder reconstruction confidence (if you want to add this)
        # This would require checking how well the latent reconstructs
        
        # Combine confidences
        overall_confidence = (velocity_confidence + consistency_confidence) / 2
        
    return overall_confidence.cpu()

class ComprehensiveMolecularEvaluator:
    """ molecular evaluation with multiple similarity metrics and baselines."""
    
    def __init__(self, training_smiles=None):
        self.training_smiles = training_smiles or []
        self.fingerprint_types = ['morgan_r2', 'morgan_r3', 'maccs', 'rdk', 'atom_pairs']
        
    def calculate_multi_fingerprint_similarity(self, target_smiles, generated_smiles):
        """Calculate similarity using multiple fingerprint types."""
        similarities = {}
        
        try:
            target_mol = Chem.MolFromSmiles(target_smiles)
            generated_mol = Chem.MolFromSmiles(generated_smiles)
            
            if target_mol is None or generated_mol is None:
                return {fp_type: 0.0 for fp_type in self.fingerprint_types}
            
            # Morgan fingerprints with different radii
            target_morgan_r2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(target_mol, 2)
            gen_morgan_r2 = rdMolDescriptors.GetMorganFingerprintAsBitVect(generated_mol, 2)
            similarities['morgan_r2'] = DataStructs.TanimotoSimilarity(target_morgan_r2, gen_morgan_r2)
            
            target_morgan_r3 = rdMolDescriptors.GetMorganFingerprintAsBitVect(target_mol, 3)
            gen_morgan_r3 = rdMolDescriptors.GetMorganFingerprintAsBitVect(generated_mol, 3)
            similarities['morgan_r3'] = DataStructs.TanimotoSimilarity(target_morgan_r3, gen_morgan_r3)
            
            # MACCS keys
            target_maccs = MACCSkeys.GenMACCSKeys(target_mol)
            gen_maccs = MACCSkeys.GenMACCSKeys(generated_mol)
            similarities['maccs'] = DataStructs.TanimotoSimilarity(target_maccs, gen_maccs)
            
            # RDKit fingerprint
            target_rdk = Chem.RDKFingerprint(target_mol)
            gen_rdk = Chem.RDKFingerprint(generated_mol)
            similarities['rdk'] = DataStructs.TanimotoSimilarity(target_rdk, gen_rdk)
            
            # Atom pairs
            target_pairs = rdMolDescriptors.GetAtomPairFingerprint(target_mol)
            gen_pairs = rdMolDescriptors.GetAtomPairFingerprint(generated_mol)
            similarities['atom_pairs'] = DataStructs.TanimotoSimilarity(target_pairs, gen_pairs)
            
        except Exception as e:
            print(f"Error calculating fingerprint similarities: {e}")
            similarities = {fp_type: 0.0 for fp_type in self.fingerprint_types}
            
        return similarities
    
    def calculate_scaffold_similarity(self, target_smiles, generated_smiles):
        """Calculate Murcko scaffold similarity."""
        try:
            target_mol = Chem.MolFromSmiles(target_smiles)
            generated_mol = Chem.MolFromSmiles(generated_smiles)
            
            if target_mol is None or generated_mol is None:
                return 0.0
            
            target_scaffold = MurckoScaffold.GetScaffoldForMol(target_mol)
            generated_scaffold = MurckoScaffold.GetScaffoldForMol(generated_mol)
            
            if target_scaffold is None or generated_scaffold is None:
                return 0.0
            
            target_scaffold_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(target_scaffold, 2)
            gen_scaffold_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(generated_scaffold, 2)
            
            return DataStructs.TanimotoSimilarity(target_scaffold_fp, gen_scaffold_fp)
            
        except Exception as e:
            print(f"Error calculating scaffold similarity: {e}")
            return 0.0
    
    def analyze_substructure_overlap(self, target_smiles, generated_smiles):
        """Analyze common substructures between target and generated molecules."""
        try:
            target_mol = Chem.MolFromSmiles(target_smiles)
            generated_mol = Chem.MolFromSmiles(generated_smiles)
            
            if target_mol is None or generated_mol is None:
                return {'common_rings': 0, 'common_chains': 0, 'substructure_match': False}
            
            # Count rings
            target_rings = target_mol.GetRingInfo().NumRings()
            generated_rings = generated_mol.GetRingInfo().NumRings()
            common_rings = min(target_rings, generated_rings)
            
            # Simple substructure matching
            substructure_match = target_mol.HasSubstructMatch(generated_mol) or generated_mol.HasSubstructMatch(target_mol)
            
            return {
                'target_rings': target_rings,
                'generated_rings': generated_rings,
                'common_rings': common_rings,
                'substructure_match': substructure_match
            }
            
        except Exception as e:
            print(f"Error in substructure analysis: {e}")
            return {'common_rings': 0, 'common_chains': 0, 'substructure_match': False}
    
    def calculate_novelty_score(self, generated_smiles):
        """Calculate novelty of generated molecule vs training set."""
        if not self.training_smiles or not generated_smiles:
            return 0.0
        
        try:
            generated_mol = Chem.MolFromSmiles(generated_smiles)
            if generated_mol is None:
                return 0.0
            
            generated_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(generated_mol, 2)
            max_similarity = 0.0
            
            for train_smiles in self.training_smiles:
                try:
                    train_mol = Chem.MolFromSmiles(train_smiles)
                    if train_mol is not None:
                        train_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(train_mol, 2)
                        sim = DataStructs.TanimotoSimilarity(generated_fp, train_fp)
                        max_similarity = max(max_similarity, sim)
                except:
                    continue
            
            return 1.0 - max_similarity  # Higher novelty = less similar to training
            
        except Exception as e:
            print(f"Error calculating novelty: {e}")
            return 0.0

class BaselineGenerator:
    """Generate baseline molecules for comparison."""
    
    def __init__(self, training_smiles):
        self.training_smiles = training_smiles
        
    def generate_random_baseline(self, num_samples, seed=42):
        """Generate random molecules from training set."""
        random.seed(seed)
        if not self.training_smiles:
            return [''] * num_samples
        if len(self.training_smiles) < num_samples:
            return random.choices(self.training_smiles, k=num_samples)
        else:
            return random.sample(self.training_smiles, num_samples)
    
    def generate_retrieval_baseline(self, query_rna_signature, all_rna_signatures, all_smiles, k=1):
        """Generate baseline by retrieving most similar RNA signatures."""
        try:
            similarities = []
            for i, rna_sig in enumerate(all_rna_signatures):
                # Simple cosine similarity
                sim = cosine_similarity([query_rna_signature], [rna_sig])[0, 0]
                similarities.append((sim, all_smiles[i]))
            
            # Sort by similarity and return top-k
            similarities.sort(reverse=True, key=lambda x: x[0])
            return [smiles for _, smiles in similarities[:k]]
            
        except Exception as e:
            print(f"Error in retrieval baseline: {e}")
            return self.generate_random_baseline(k)
    
    def generate_structure_similar_baseline(self, target_smiles, k=5):
        """Generate baseline by finding structurally similar molecules."""
        try:
            target_mol = Chem.MolFromSmiles(target_smiles)
            if target_mol is None:
                return self.generate_random_baseline(k)
            
            target_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(target_mol, 2)
            similarities = []
            
            for smiles in self.training_smiles:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2)
                        sim = DataStructs.TanimotoSimilarity(target_fp, fp)
                        similarities.append((sim, smiles))
                except:
                    continue
            
            similarities.sort(reverse=True, key=lambda x: x[0])
            return [smiles for _, smiles in similarities[:k]]
            
        except Exception as e:
            print(f"Error in structure baseline: {e}")
            return self.generate_random_baseline(k)


def calculate_comprehensive_molecular_properties(smiles):
    """Extended molecular property calculation."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        Chem.SanitizeMol(mol)
        props = {}
        
        # Basic properties
        props['MW'] = Descriptors.MolWt(mol)
        props['LogP'] = Descriptors.MolLogP(mol)
        props['HBA'] = Descriptors.NumHAcceptors(mol)
        props['HBD'] = Descriptors.NumHDonors(mol)
        props['TPSA'] = Descriptors.TPSA(mol)
        props['RotBonds'] = Descriptors.NumRotatableBonds(mol)
        props['AromaticRings'] = Descriptors.NumAromaticRings(mol)
        props['HeavyAtoms'] = Descriptors.HeavyAtomCount(mol)
        props['FractionCsp3'] = rdMolDescriptors.CalcFractionCSP3(mol)
        props['QED'] = Descriptors.qed(mol)
        
        # Additional descriptors
        props['BertzCT'] = Descriptors.BertzCT(mol)  # Complexity
        props['MolMR'] = Descriptors.MolMR(mol)  # Molar refractivity
        props['SASA'] = Descriptors.TPSA(mol)  # Solvent accessible surface area approximation
        props['NumRings'] = Descriptors.RingCount(mol)
        props['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
        props['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
        props['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
        
        # Drug-likeness indicators
        props['Lipinski_violations'] = sum([
            props['MW'] > 500,
            props['LogP'] > 5,
            props['HBA'] > 10,
            props['HBD'] > 5
        ])
        
        # Veber's rule
        props['Veber_violations'] = sum([
            props['RotBonds'] > 10,
            props['TPSA'] > 140
        ])
        
        return props
        
    except Exception as e:
        print(f"Failed to process SMILES {smiles}: {e}")
        return None


@torch.no_grad()
def sample_fully_conditioned(model, flow, shape, y_full, pad_mask, num_steps=50, device=None):
    """Sample with full conditioning on both RNA and image modalities."""
    if device is None:
        device = next(model.parameters()).device
    
    batch_size = shape[0]
    x = torch.randn(*shape, device=device)
    dt = 1.0 / num_steps
    
    for i in range(num_steps):
        t = torch.full((batch_size,), i * dt, device=device)
        t_discrete = (t * 999).long()
        
        with torch.no_grad():
            velocity = model(x, t_discrete, y=y_full, pad_mask=pad_mask)
            if velocity.shape[1] == 2 * x.shape[1]:
                velocity, _ = torch.split(velocity, x.shape[1], dim=1)
        x = x + dt * velocity 
    return x


def rank_candidates_by_comprehensive_score(candidates):
    """Rank candidates using comprehensive drug-likeness scoring."""
    scored_candidates = []
    
    for smi in candidates:
        props = calculate_comprehensive_molecular_properties(smi)
        if props is None:
            continue
        
        # Multi-criteria drug-likeness score
        score = 0
        
        # QED (Quantitative Estimate of Drug-likeness)
        score += props['QED'] * 0.25
        
        # Lipinski compliance
        score += (4 - props['Lipinski_violations']) / 4 * 0.20
        
        # Veber compliance
        score += (2 - props['Veber_violations']) / 2 * 0.15
        
        # Molecular complexity (moderate is better)
        complexity_score = 1 - abs(props['BertzCT'] - 400) / 400  # Normalize around 400
        score += max(0, complexity_score) * 0.15
        
        # Size appropriateness
        mw_score = 1 - abs(props['MW'] - 350) / 350  # Target around 350 Da
        score += max(0, mw_score) * 0.10
        
        # Fraction Csp3 (3D character)
        score += min(props['FractionCsp3'], 0.5) * 2 * 0.10  # Target ~0.25-0.5
        
        # TPSA appropriateness
        tpsa_score = 1 - abs(props['TPSA'] - 80) / 140  # Target around 60-100
        score += max(0, tpsa_score) * 0.05
        
        scored_candidates.append((smi, score, props))
    
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    return scored_candidates


def diversity_analysis(smiles_list):
    """Diversity analysis with multiple metrics."""
    if len(smiles_list) < 2:
        return {
            'internal_diversity': 0.0,
            'scaffold_diversity': 0.0,
            'property_diversity': 0.0,
            'num_unique_scaffolds': 0
        }
    
    # Fingerprint-based diversity
    fingerprints = []
    scaffolds = []
    properties = []
    
    for smi in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                # Fingerprints
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2)
                fingerprints.append(fp)
                
                # Scaffolds
                scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                scaffold_smiles = Chem.MolToSmiles(scaffold) if scaffold else ""
                scaffolds.append(scaffold_smiles)
                
                # Properties for property-based diversity
                props = calculate_comprehensive_molecular_properties(smi)
                if props:
                    prop_vector = [props['MW'], props['LogP'], props['TPSA'], props['QED']]
                    properties.append(prop_vector)
        except:
            continue
    
    results = {}
    
    # Fingerprint diversity
    if len(fingerprints) >= 2:
        similarities = []
        for i in range(len(fingerprints)):
            for j in range(i + 1, len(fingerprints)):
                sim = DataStructs.TanimotoSimilarity(fingerprints[i], fingerprints[j])
                similarities.append(sim)
        results['internal_diversity'] = 1.0 - np.mean(similarities)
    else:
        results['internal_diversity'] = 0.0
    
    # Scaffold diversity
    unique_scaffolds = len(set(scaffolds))
    results['scaffold_diversity'] = unique_scaffolds / len(smiles_list)
    results['num_unique_scaffolds'] = unique_scaffolds
    
    # Property-based diversity
    if len(properties) >= 2:
        from sklearn.metrics.pairwise import euclidean_distances
        prop_matrix = np.array(properties)
        # Normalize properties
        scaler = StandardScaler()
        prop_matrix_norm = scaler.fit_transform(prop_matrix)
        distances = euclidean_distances(prop_matrix_norm)
        # Get upper triangle (excluding diagonal)
        upper_triangle = distances[np.triu_indices_from(distances, k=1)]
        results['property_diversity'] = np.mean(upper_triangle)
    else:
        results['property_diversity'] = 0.0
    
    return results


def analyze_mechanism_consistency(compound_name, generated_smiles, target_smiles):
    """Analyze if generated molecule is mechanistically consistent."""
    # compound classification
    compound_name_lower = compound_name.lower()
    
    mechanism_classes = {
        'steroid': ['steroid', 'cortisone', 'predni', 'hydrocortisone', 'dexamethasone', 'testosterone'],
        'taxane': ['taxol', 'paclitaxel', 'docetaxel'],
        'antibiotic': ['doxorubicin', 'mitomycin', 'bleomycin', 'streptomycin'],
        'kinase_inhibitor': ['kinase', 'inhibitor', 'dasatinib', 'imatinib'],
        'antimetabolite': ['methotrexate', '5-fluorouracil', 'cytarabine'],
        'alkylating_agent': ['cyclophosphamide', 'cisplatin', 'carboplatin'],
        'topoisomerase_inhibitor': ['etoposide', 'topotecan', 'irinotecan'],
        'antimicrotubule': ['vincristine', 'vinblastine', 'colchicine']
    }
    
    predicted_mechanism = 'other'
    for mechanism, keywords in mechanism_classes.items():
        if any(keyword in compound_name_lower for keyword in keywords):
            predicted_mechanism = mechanism
            break
    
    # Calculate structural features that might indicate mechanism
    def get_structural_features(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {}
            
            features = {
                'has_steroid_like': False,
                'num_rings': mol.GetRingInfo().NumRings(),
                'aromatic_rings': Descriptors.NumAromaticRings(mol),
                'has_heterocycles': any(atom.GetAtomicNum() not in [1, 6] for atom in mol.GetAtoms() if atom.IsInRing()),
                'molecular_flexibility': Descriptors.NumRotatableBonds(mol),
                'polar_surface_area': Descriptors.TPSA(mol)
            }
            
            # Simple steroid-like pattern detection (4 fused rings)
            if features['num_rings'] >= 4 and features['aromatic_rings'] <= 1:
                features['has_steroid_like'] = True
            
            return features
        except:
            return {}
    
    target_features = get_structural_features(target_smiles)
    generated_features = get_structural_features(generated_smiles)
    
    # Calculate feature consistency
    consistency_score = 0.0
    if target_features and generated_features:
        # Ring consistency
        ring_diff = abs(target_features['num_rings'] - generated_features['num_rings'])
        ring_consistency = max(0, 1 - ring_diff / 5)  # Normalize by max expected difference
        
        # Aromatic ring consistency
        aromatic_diff = abs(target_features['aromatic_rings'] - generated_features['aromatic_rings'])
        aromatic_consistency = max(0, 1 - aromatic_diff / 3)
        
        # Steroid-like consistency
        steroid_consistency = 1.0 if target_features['has_steroid_like'] == generated_features['has_steroid_like'] else 0.0
        
        consistency_score = (ring_consistency + aromatic_consistency + steroid_consistency) / 3
    
    return {
        'predicted_mechanism': predicted_mechanism,
        'target_features': target_features,
        'generated_features': generated_features,
        'mechanism_consistency_score': consistency_score
    }


@torch.no_grad()
def main(args):
    """ inference with comprehensive evaluation."""
    torch.backends.cuda.matmul.allow_tf32 = True
    assert torch.cuda.is_available(), "Inference requires GPU"
    torch.set_grad_enabled(False)

    device = torch.device("cuda:0")
    torch.manual_seed(args.global_seed)
    torch.cuda.set_device(0)
    print(f"Using device: {device}, seed: {args.global_seed}")

    if args.ckpt is None:
        raise ValueError("Please specify checkpoint path with --ckpt")

    # Load data
    metadata_control = pd.read_csv(args.metadata_control_path)
    metadata_drug = pd.read_csv(args.metadata_drug_path)
    gene_count_matrix = pd.read_parquet(args.gene_count_matrix_path)
    print(f"Loaded gene matrix: {gene_count_matrix.shape}")

    # Create DiT model
    latent_size = 127
    in_channels = 64
    cross_attn = 192
    condition_dim = 192
    
    model = DiT_models[args.model](
        input_size=latent_size,
        in_channels=in_channels,
        cross_attn=cross_attn,
        condition_dim=condition_dim
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    print(f"Loaded DiT model from {args.ckpt}")

    # Setup encoders
    image_encoder = ImageEncoder(img_channels=4, output_dim=128).to(device)
    image_encoder.load_state_dict(checkpoint['image_encoder'], strict=True)
    image_encoder.eval()
    
    rna_encoder = RNAEncoder(
        input_dim=gene_count_matrix.shape[0],
        output_dim=64,
        dropout=0.1
    ).to(device)
    rna_encoder.load_state_dict(checkpoint['rna_encoder'], strict=True) 
    rna_encoder.eval()
    print("Loaded RNA and Image encoders")

    # Setup autoencoder
    ae_config = {
        'bert_config_decoder': './config_decoder.json',
        'bert_config_encoder': './config_encoder.json',
        'embed_dim': 256,
    }
    tokenizer = regexTokenizer(vocab_path='./dataloaders/vocab_bpe_300_sc.txt', max_len=127)
    ae_model = ldmol_autoencoder(config=ae_config, no_train=True, tokenizer=tokenizer, use_linear=True)
    
    if args.vae:
        ae_checkpoint = torch.load(args.vae, map_location='cpu')
        try:
            ae_state_dict = ae_checkpoint['model']
        except:
            ae_state_dict = ae_checkpoint['state_dict']
        ae_model.load_state_dict(ae_state_dict, strict=False)
    
    for param in ae_model.parameters():
        param.requires_grad = False
    del ae_model.text_encoder2
    ae_model = ae_model.to(device)
    ae_model.eval()
    print("Loaded autoencoder")

    # Setup rectified flow
    flow = create_rectified_flow(num_timesteps=1000)

    # Create dataloader
    loader = create_raw_drug_dataloader(
        metadata_control=metadata_control,
        metadata_drug=metadata_drug,
        gene_count_matrix=gene_count_matrix,
        image_json_path=args.image_json_path,
        drug_data_path=args.drug_data_path,
        raw_drug_csv_path=args.raw_drug_csv_path,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        compound_name_label='compound',
        debug_mode=args.debug_mode,
    )

    print(f"Created dataloader with {len(loader)} batches")

    # Collect training SMILES for baselines and novelty calculation
    print("Collecting training SMILES for baselines...")
    training_data = {
        'smiles': [],
        'biological_features': [],
        'compound_names': [],
        'rna_signatures': [],
        'image_features': [],
        'full_features': []
    }

    # Initialize evaluators
    mol_evaluator = ComprehensiveMolecularEvaluator(training_smiles=[])
    baseline_generator = BaselineGenerator([])

    # Initialize results storage
    results = []
    all_generated_smiles = []
    molecular_properties = []

    output_file = f'./inference_results_{args.inference_mode}_{args.global_seed}.txt'
    detailed_output_file = f'./detailed_results_{args.inference_mode}_{args.global_seed}.json'

    # Update header to include method and confidence
    with open(output_file, 'w') as f:
        f.write("sample_id\ttarget_smiles\tcompound_name\tmethod\tconfidence\t"
            "generation_confidence\tretrieval_similarity\tvalidity\t"
            "morgan_r2_sim\tmaccs_sim\trdk_sim\tscaffold_sim\t"
            "novelty_score\tdrug_score\tmechanism_consistency\tnum_candidates\tbest_generated\n")

    start_time = time.time()

    sample_counter = 0
    # Main generation loop with multiple modes (replace the entire generation section)
    for batch_idx, batch in enumerate(tqdm(loader, desc=f"Molecular {args.inference_mode}")):
        if batch_idx >= args.max_batches and args.max_batches > 0:
            break
            
        # Encode features
        control_imgs = batch['control_images']
        treatment_imgs = batch['treatment_images']
        control_rna = batch['control_transcriptomics']
        treatment_rna = batch['treatment_transcriptomics']
        target_smiles = batch['target_smiles']
        compound_names = batch['compound_name']

        # Update training data storage
        training_data['smiles'].extend(target_smiles)
        training_data['compound_names'].extend(compound_names)

        y_features, pad_mask = dual_rna_image_encoder(
            control_imgs, treatment_imgs, control_rna, treatment_rna,
            image_encoder, rna_encoder, device
        )
        
        # Store training features for retrieval
        training_data['biological_features'].append(y_features.cpu())
        training_data['full_features'].extend([feat.cpu() for feat in y_features])
        
        # Extract RNA and image features separately
        rna_features = y_features[:, :, 128:].mean(dim=1).cpu()
        image_features = y_features[:, :, :128].mean(dim=1).cpu()
        training_data['rna_signatures'].extend([feat.cpu() for feat in rna_features])
        training_data['image_features'].extend([feat.cpu() for feat in image_features])

        # Update evaluators with new training data
        mol_evaluator.training_smiles = training_data['smiles']
        baseline_generator.training_smiles = training_data['smiles']
        
        batch_size = y_features.shape[0]
        shape = (batch_size, in_channels, latent_size, 1)
        
        # MODE-SPECIFIC GENERATION
        if args.inference_mode == 'retrieval':
            # Pure retrieval mode
            if len(training_data['full_features']) > batch_size:
                # We have enough training data for retrieval
                retrieved_smiles, similarity_scores = exact_retrieval(
                    y_features, training_data, top_k=args.retrieval_top_k, 
                    similarity_metric=args.similarity_metric
                )
                
                # Process retrieved results
                for i in range(batch_size):
                    target = target_smiles[i]
                    compound = compound_names[i]
                    
                    if retrieved_smiles[i]:
                        best_candidate = retrieved_smiles[i][0]
                        retrieval_confidence = similarity_scores[i][0]
                        validity = 1 if best_candidate and best_candidate != "" else 0
                        num_candidates = len([s for s in retrieved_smiles[i] if s and s != ""])
                    else:
                        best_candidate = ""
                        retrieval_confidence = 0.0
                        validity = 0
                        num_candidates = 0
                    
                    # Evaluate retrieved molecule
                    similarity_results = {}
                    if validity and target:
                        similarity_results = mol_evaluator.calculate_multi_fingerprint_similarity(target, best_candidate)
                        similarity_results['scaffold_similarity'] = mol_evaluator.calculate_scaffold_similarity(target, best_candidate)
                    
                    novelty_score = mol_evaluator.calculate_novelty_score(best_candidate) if validity else 0.0
                    mechanism_analysis = analyze_mechanism_consistency(compound, best_candidate, target)
                    
                    props = calculate_comprehensive_molecular_properties(best_candidate) if validity else {}
                    drug_score = 0.0
                    if props:
                        # Calculate drug-likeness score
                        drug_score = props.get('QED', 0) * 0.5 + (4 - sum([
                            props['MW'] > 500, props['LogP'] > 5, 
                            props.get('HBA', 0) > 10, props.get('HBD', 0) > 5
                        ])) / 4 * 0.5
                    
                    result = {
                        'method': 'retrieval',
                        'target_smiles': target,
                        'generated_smiles': best_candidate,
                        'compound_name': compound,
                        'confidence': float(retrieval_confidence),
                        'generation_confidence': 0.0,
                        'retrieval_similarity': float(retrieval_confidence),
                        'validity': validity,
                        'similarity_analysis': similarity_results,
                        'novelty_score': novelty_score,
                        'druglikeness_score': drug_score,
                        'molecular_properties': props,
                        'mechanism_analysis': mechanism_analysis,
                        'num_candidates': num_candidates
                    }
                    results.append(result)
            else:
                # Not enough training data, return empty results
                for i in range(batch_size):
                    result = {
                        'method': 'retrieval',
                        'target_smiles': target_smiles[i],
                        'generated_smiles': '',
                        'compound_name': compound_names[i],
                        'confidence': 0.0,
                        'generation_confidence': 0.0,
                        'retrieval_similarity': 0.0,
                        'validity': 0,
                        'similarity_analysis': {},
                        'novelty_score': 0.0,
                        'druglikeness_score': 0.0,
                        'molecular_properties': {},
                        'mechanism_analysis': {},
                        'num_candidates': 0
                    }
                    results.append(result)
        
        elif args.inference_mode == 'generation':
            # Pure generation mode
            all_generated = []
            all_confidences = []
            
            for sample_idx in range(args.num_samples_per_condition):
                samples = sample_with_cfg(
                    model=model, flow=flow, shape=shape, y_full=y_features, 
                    pad_mask=pad_mask, cfg_scale_rna=2.0, cfg_scale_image=2.0,
                    num_steps=args.num_sampling_steps, device=device
                )
                
                # Estimate confidence for this generation
                generation_confidence = estimate_generation_confidence(
                    model, flow, samples, y_features, pad_mask, device
                )
                
                samples = samples.squeeze(-1).permute((0, 2, 1))
                generated_smiles = AE_SMILES_decoder(samples, ae_model, stochastic=False, k=1)
                all_generated.append(generated_smiles)
                all_confidences.append(generation_confidence)
            
            # Process generated results
            for i in range(batch_size):
                target = target_smiles[i]
                compound = compound_names[i]
                
                # Collect all candidates for this sample
                candidates_with_conf = []
                for gen_idx in range(len(all_generated)):
                    candidate = all_generated[gen_idx][i]
                    confidence = all_confidences[gen_idx][i]
                    candidates_with_conf.append((candidate, float(confidence)))
                
                # Validate candidates and select best
                valid_candidates = []
                for cand, conf in candidates_with_conf:
                    try:
                        if cand and cand != "":
                            mol = Chem.MolFromSmiles(cand)
                            if mol is not None:
                                canonical = Chem.MolToSmiles(mol, canonical=True)
                                valid_candidates.append((canonical, conf))
                    except:
                        continue
                
                if valid_candidates:
                    # Select candidate with highest confidence
                    best_candidate, best_confidence = max(valid_candidates, key=lambda x: x[1])
                    validity = 1
                    num_candidates = len(valid_candidates)
                else:
                    best_candidate = ""
                    best_confidence = 0.0
                    validity = 0
                    num_candidates = 0
                
                # Evaluate generated molecule
                similarity_results = {}
                if validity and target:
                    similarity_results = mol_evaluator.calculate_multi_fingerprint_similarity(target, best_candidate)
                    similarity_results['scaffold_similarity'] = mol_evaluator.calculate_scaffold_similarity(target, best_candidate)
                
                novelty_score = mol_evaluator.calculate_novelty_score(best_candidate) if validity else 0.0
                mechanism_analysis = analyze_mechanism_consistency(compound, best_candidate, target)
                
                props = calculate_comprehensive_molecular_properties(best_candidate) if validity else {}
                drug_score = 0.0
                if props:
                    drug_score = props.get('QED', 0) * 0.5 + (4 - sum([
                        props['MW'] > 500, props['LogP'] > 5,
                        props.get('HBA', 0) > 10, props.get('HBD', 0) > 5
                    ])) / 4 * 0.5
                
                result = {
                    'method': 'generation',
                    'target_smiles': target,
                    'generated_smiles': best_candidate,
                    'compound_name': compound,
                    'confidence': float(best_confidence),
                    'generation_confidence': float(best_confidence),
                    'retrieval_similarity': 0.0,
                    'validity': validity,
                    'similarity_analysis': similarity_results,
                    'novelty_score': novelty_score,
                    'druglikeness_score': drug_score,
                    'molecular_properties': props,
                    'mechanism_analysis': mechanism_analysis,
                    'num_candidates': num_candidates
                }
                results.append(result)
        
        elif args.inference_mode == 'adaptive':
            # Adaptive mode - try generation first, fall back to retrieval if confidence is low
            
            # Generate first
            samples = sample_with_cfg(
                model, flow, shape, y_features, pad_mask, 
                cfg_scale_rna=2.0, cfg_scale_image=2.0,
                num_steps=args.num_sampling_steps, device=device
            )
            
            generation_confidence = estimate_generation_confidence(
                model, flow, samples, y_features, pad_mask, device
            )
            
            samples = samples.squeeze(-1).permute((0, 2, 1))
            generated_smiles = AE_SMILES_decoder(samples, ae_model, stochastic=False, k=1)
            
            # Get retrieval backup if we have enough training data
            retrieved_smiles, retrieval_scores = [], []
            if len(training_data['full_features']) > batch_size:
                retrieved_smiles, retrieval_scores = exact_retrieval(
                    y_features, training_data, top_k=1, similarity_metric=args.similarity_metric
                )
            
            # Process each sample
            for i in range(batch_size):
                target = target_smiles[i]
                compound = compound_names[i]
                
                gen_conf = generation_confidence[i]
                generated_candidate = generated_smiles[i]
                
                # Check generation validity
                gen_valid = False
                try:
                    if generated_candidate and generated_candidate != "":
                        mol = Chem.MolFromSmiles(generated_candidate)
                        if mol is not None:
                            generated_candidate = Chem.MolToSmiles(mol, canonical=True)
                            gen_valid = True
                except:
                    pass
                
                # Decision logic for adaptive mode
                use_generation = (gen_conf >= args.confidence_threshold and gen_valid)
                
                if use_generation:
                    # Use generation
                    best_candidate = generated_candidate
                    method = 'generation'
                    confidence = float(gen_conf)
                    retrieval_sim = 0.0
                    validity = 1
                else:
                    # Fall back to retrieval
                    if retrieved_smiles and i < len(retrieved_smiles) and retrieved_smiles[i]:
                        best_candidate = retrieved_smiles[i][0]
                        method = 'retrieval_fallback'
                        confidence = float(retrieval_scores[i][0]) if retrieval_scores[i] else 0.0
                        retrieval_sim = confidence
                        validity = 1 if best_candidate and best_candidate != "" else 0
                    else:
                        best_candidate = ""
                        method = 'retrieval_fallback'
                        confidence = 0.0
                        retrieval_sim = 0.0
                        validity = 0
                
                # Evaluate final molecule
                similarity_results = {}
                if validity and target:
                    similarity_results = mol_evaluator.calculate_multi_fingerprint_similarity(target, best_candidate)
                    similarity_results['scaffold_similarity'] = mol_evaluator.calculate_scaffold_similarity(target, best_candidate)
                
                novelty_score = mol_evaluator.calculate_novelty_score(best_candidate) if validity else 0.0
                mechanism_analysis = analyze_mechanism_consistency(compound, best_candidate, target)
                
                props = calculate_comprehensive_molecular_properties(best_candidate) if validity else {}
                drug_score = 0.0
                if props:
                    drug_score = props.get('QED', 0) * 0.5 + (4 - sum([
                        props['MW'] > 500, props['LogP'] > 5,
                        props.get('HBA', 0) > 10, props.get('HBD', 0) > 5
                    ])) / 4 * 0.5
                
                result = {
                    'method': method,
                    'target_smiles': target,
                    'generated_smiles': best_candidate,
                    'compound_name': compound,
                    'confidence': confidence,
                    'generation_confidence': float(gen_conf),
                    'retrieval_similarity': retrieval_sim,
                    'validity': validity,
                    'similarity_analysis': similarity_results,
                    'novelty_score': novelty_score,
                    'druglikeness_score': drug_score,
                    'molecular_properties': props,
                    'mechanism_analysis': mechanism_analysis,
                    'num_candidates': 1
                }
                results.append(result)
        
        # Write results to file for this batch
        for result in results[-batch_size:]:  # Only process the results from this batch
            target = result['target_smiles']
            best_candidate = result['generated_smiles']
            compound = result['compound_name']
            method = result['method']
            confidence = result['confidence']
            gen_conf = result['generation_confidence']
            ret_sim = result['retrieval_similarity']
            validity = result['validity']
            
            # Get similarity scores
            sim_analysis = result.get('similarity_analysis', {})
            morgan_sim = sim_analysis.get('morgan_r2', 0.0)
            maccs_sim = sim_analysis.get('maccs', 0.0)
            rdk_sim = sim_analysis.get('rdk', 0.0)
            scaffold_sim = sim_analysis.get('scaffold_similarity', 0.0)
            
            novelty_score = result.get('novelty_score', 0.0)
            drug_score = result.get('druglikeness_score', 0.0)
            mechanism_score = result.get('mechanism_analysis', {}).get('mechanism_consistency_score', 0.0)
            num_candidates = result.get('num_candidates', 0)
            
            with open(output_file, 'a') as f:
                sample_counter += 1
                f.write(f"{sample_counter}\t{target}\t{compound}\t{method}\t{confidence:.3f}\t"
                    f"{gen_conf:.3f}\t{ret_sim:.3f}\t{validity}\t"
                    f"{morgan_sim:.3f}\t{maccs_sim:.3f}\t{rdk_sim:.3f}\t{scaffold_sim:.3f}\t"
                    f"{novelty_score:.3f}\t{drug_score:.3f}\t{mechanism_score:.3f}\t{num_candidates}\t{best_candidate}\n")

    # Calculate comprehensive metrics (replace the final summary section)
    total_time = time.time() - start_time
    valid_results = [r for r in results if r['validity'] == 1]
    validity_rate = len(valid_results) / len(results) if results else 0
    
    # Method distribution
    method_counts = {}
    for r in results:
        method = r.get('method', 'unknown')
        method_counts[method] = method_counts.get(method, 0) + 1
    
    # Confidence statistics by method
    confidence_stats = {}
    generation_confidence_stats = {}
    
    for method in ['retrieval', 'generation', 'retrieval_fallback']:
        method_results = [r for r in results if r.get('method') == method]
        if method_results:
            confidences = [r['confidence'] for r in method_results]
            confidence_stats[method] = {
                'mean': np.mean(confidences),
                'std': np.std(confidences), 
                'min': np.min(confidences),
                'max': np.max(confidences)
            }
            
            if method in ['generation', 'retrieval_fallback']:
                gen_confs = [r['generation_confidence'] for r in method_results]
                generation_confidence_stats[method] = {
                    'mean': np.mean(gen_confs),
                    'std': np.std(gen_confs),
                    'min': np.min(gen_confs),
                    'max': np.max(gen_confs)
                }
    
    # Multi-fingerprint similarity averages
    similarity_averages = {}
    for sim_type in ['morgan_r2', 'maccs', 'rdk', 'atom_pairs']:
        values = [r['similarity_analysis'].get(sim_type, 0) for r in valid_results 
                 if 'similarity_analysis' in r]
        similarity_averages[sim_type] = np.mean(values) if values else 0.0
    
    scaffold_similarities = [r['similarity_analysis'].get('scaffold_similarity', 0) for r in valid_results 
                           if 'similarity_analysis' in r]
    avg_scaffold_similarity = np.mean(scaffold_similarities) if scaffold_similarities else 0.0
    
    avg_novelty = np.mean([r['novelty_score'] for r in valid_results]) if valid_results else 0
    avg_druglikeness = np.mean([r['druglikeness_score'] for r in valid_results]) if valid_results else 0
    
    # Property statistics
    property_stats = {}
    molecular_properties = [r['molecular_properties'] for r in valid_results if r.get('molecular_properties')]
    if molecular_properties:
        extended_props = ['MW', 'LogP', 'HBA', 'HBD', 'TPSA', 'QED', 'BertzCT', 'NumRings']
        for prop in extended_props:
            values = [p[prop] for p in molecular_properties if prop in p]
            if values:
                property_stats[prop] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
    
    # Generate all SMILES for diversity analysis
    all_generated_smiles = [r['generated_smiles'] for r in valid_results if r['generated_smiles']]
    
    # Print results with mode-specific information
    print(f"\n{'='*120}")
    print(f"COMPREHENSIVE MOLECULAR {args.inference_mode.upper()} EVALUATION")
    print(f"{'='*120}")
    
    print(f"Mode Configuration:")
    print(f"  Inference mode: {args.inference_mode}")
    if args.inference_mode == 'adaptive':
        print(f"  Confidence threshold: {args.confidence_threshold}")
    if args.inference_mode in ['retrieval', 'adaptive']:
        print(f"  Retrieval top-k: {args.retrieval_top_k}")
        print(f"  Similarity metric: {args.similarity_metric}")
    
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {len(results)}")
    print(f"  Valid molecules: {len(valid_results)}/{len(results)} ({validity_rate:.1%})")
    print(f"  Processing time: {total_time:.1f}s ({total_time/len(results):.2f}s/sample)")
    
    print(f"\nMethod Distribution:")
    for method, count in method_counts.items():
        percentage = count / len(results) * 100
        print(f"  {method}: {count}/{len(results)} ({percentage:.1f}%)")
    
    print(f"\nConfidence Analysis:")
    for method, stats in confidence_stats.items():
        print(f"  {method} confidence: {stats['mean']:.3f} ± {stats['std']:.3f} "
              f"(range: {stats['min']:.3f}-{stats['max']:.3f})")
    
    if args.inference_mode == 'adaptive':
        print(f"\nGeneration Confidence Analysis (for adaptive mode):")
        for method, stats in generation_confidence_stats.items():
            print(f"  {method} generation confidence: {stats['mean']:.3f} ± {stats['std']:.3f} "
                  f"(range: {stats['min']:.3f}-{stats['max']:.3f})")
    
    print(f"\nMulti-Fingerprint Similarity Analysis:")
    for sim_type, avg_sim in similarity_averages.items():
        print(f"  {sim_type.upper()}: {avg_sim:.3f}")
    print(f"  Scaffold similarity: {avg_scaffold_similarity:.3f}")
    
    print(f"\nNovelty and Drug-likeness:")
    print(f"  Average novelty score: {avg_novelty:.3f}")
    print(f"  Average drug-likeness score: {avg_druglikeness:.3f}")
    
    print(f"\nDiversity Analysis:")
    overall_diversity = diversity_analysis(all_generated_smiles)
    print(f"  Internal diversity: {overall_diversity.get('internal_diversity', 0):.3f}")
    print(f"  Scaffold diversity: {overall_diversity.get('scaffold_diversity', 0):.3f}")
    print(f"  Property diversity: {overall_diversity.get('property_diversity', 0):.3f}")
    print(f"  Unique scaffolds: {overall_diversity.get('num_unique_scaffolds', 0)}")
    
    mechanism_scores = [r['mechanism_analysis']['mechanism_consistency_score'] 
                       for r in valid_results if 'mechanism_analysis' in r]
    if mechanism_scores:
        print(f"\nMechanism Consistency Analysis:")
        print(f"  Average mechanism consistency: {np.mean(mechanism_scores):.3f}")
    
    print(f"\nExtended Molecular Property Statistics:")
    for prop, stats in property_stats.items():
        print(f"  {prop}: {stats['mean']:.2f} ± {stats['std']:.2f} "
              f"(range: {stats['min']:.2f}-{stats['max']:.2f})")
    
    # Show top examples by method and confidence/similarity
    print(f"\nTop Examples by Method:")
    
    for method in method_counts.keys():
        method_results = [r for r in valid_results if r.get('method') == method]
        if not method_results:
            continue
            
        # Sort by confidence for generation, by similarity for retrieval
        if method == 'generation':
            sorted_method_results = sorted(method_results, key=lambda x: x['confidence'], reverse=True)
        else:
            sorted_method_results = sorted(method_results, key=lambda x: x.get('retrieval_similarity', x['confidence']), reverse=True)
        
        print(f"\n{method.upper()} - Top Example:")
        if sorted_method_results:
            result = sorted_method_results[0]
            print(f"  Target compound: {result['compound_name']}")
            print(f"  Generated: {result['generated_smiles']}")
            print(f"  Target:    {result['target_smiles']}")
            
            sim_analysis = result.get('similarity_analysis', {})
            print(f"  Similarities: Morgan={sim_analysis.get('morgan_r2', 0):.3f}, "
                  f"MACCS={sim_analysis.get('maccs', 0):.3f}, "
                  f"Scaffold={sim_analysis.get('scaffold_similarity', 0):.3f}")
            
            print(f"  Confidence: {result['confidence']:.3f}")
            if result.get('generation_confidence', 0) > 0:
                print(f"  Generation confidence: {result['generation_confidence']:.3f}")
            print(f"  Novelty: {result['novelty_score']:.3f}")
            print(f"  Drug-likeness: {result['druglikeness_score']:.3f}")
    
    print(f"\n{'='*120}")
    print(f"Results saved to: {output_file}")
    
    # Save comprehensive detailed results
    comprehensive_summary = {
        'evaluation_metadata': {
            'evaluation_type': f'comprehensive_molecular_{args.inference_mode}',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': {
                'inference_mode': args.inference_mode,
                'confidence_threshold': args.confidence_threshold if args.inference_mode == 'adaptive' else None,
                'retrieval_top_k': args.retrieval_top_k,
                'similarity_metric': args.similarity_metric,
                'num_samples_per_condition': args.num_samples_per_condition,
                'num_sampling_steps': args.num_sampling_steps,
                'global_seed': args.global_seed
            }
        },
        'generation_statistics': {
            'total_samples': len(results),
            'valid_molecules': len(valid_results),
            'validity_rate': validity_rate,
            'processing_time': total_time,
            'method_distribution': method_counts
        },
        'confidence_analysis': {
            'confidence_stats': confidence_stats,
            'generation_confidence_stats': generation_confidence_stats
        },
        'similarity_analysis': {**similarity_averages, 'scaffold_similarity': avg_scaffold_similarity},
        'novelty_and_druglikeness': {
            'average_novelty': avg_novelty,
            'average_druglikeness': avg_druglikeness,
        },
        'diversity_metrics': overall_diversity,
        'property_statistics': property_stats,
        'mechanism_analysis': {
            'average_mechanism_consistency': np.mean(mechanism_scores) if mechanism_scores else 0.0
        },
        'training_set_info': {
            'num_training_molecules': len(training_data['smiles']),
            'unique_training_molecules': len(set(training_data['smiles']))
        }
    }
    
    # Add top examples by method
    for method in method_counts.keys():
        method_results = [r for r in valid_results if r.get('method') == method]
        if method_results:
            if method == 'generation':
                sorted_method_results = sorted(method_results, key=lambda x: x['confidence'], reverse=True)
            else:
                sorted_method_results = sorted(method_results, key=lambda x: x.get('retrieval_similarity', x['confidence']), reverse=True)
            comprehensive_summary[f'top_{method}_examples'] = sorted_method_results[:5]
    
    with open(detailed_output_file, 'w') as f:
        json.dump(comprehensive_summary, f, indent=2, default=str)
    
    print(f"Comprehensive results saved to: {detailed_output_file}")
    print(f"Mode used: {args.inference_mode}")
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="LDMol")
    parser.add_argument("--ckpt", type=str, default='/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/src_new/LDMol/results/001-LDMol/checkpoints/0050000.pt')
    parser.add_argument("--vae", type=str, default="/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/src_new/LDMol/dataloaders/checkpoint_autoencoder.ckpt")
    
    # Data paths
    parser.add_argument("--image-json-path", type=str, default="/depot/natallah/data/Mengbo/HnE_RNA/PertRF/data/processed_data/image_paths.json")
    parser.add_argument("--drug-data-path", type=str, default="/depot/natallah/data/Mengbo/HnE_RNA/drug/PubChem/GDP_compatible/preprocessed_drugs.synonymous.pkl")
    parser.add_argument("--raw-drug-csv-path", type=str, default="/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/PertRF/drug/PubChem/GDP_compatible/complete_drug_data.csv")
    parser.add_argument("--metadata-control-path", type=str, default="/depot/natallah/data/Mengbo/HnE_RNA/PertRF/data/processed_data/metadata_control.csv")
    parser.add_argument("--metadata-drug-path", type=str, default="/depot/natallah/data/Mengbo/HnE_RNA/PertRF/data/processed_data/metadata_drug.csv")
    parser.add_argument("--gene-count-matrix-path", type=str, default="/depot/natallah/data/Mengbo/HnE_RNA/PertRF/data/processed_data/GDPx1x2_gene_counts.parquet")
    
    # Generation parameters
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-samples-per-condition", type=int, default=3)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--debug-mode", action="store_true")
    
    # New inference mode arguments
    parser.add_argument("--inference-mode", type=str, choices=['retrieval', 'generation', 'adaptive'], 
                       default='generation', help='Inference mode: retrieval, generation, or adaptive')
    parser.add_argument("--confidence-threshold", type=float, default=0.7,
                       help='Confidence threshold for adaptive mode (0-1)')
    parser.add_argument("--retrieval-top-k", type=int, default=3,
                       help='Number of similar conditions to consider for retrieval')
    parser.add_argument("--similarity-metric", type=str, choices=['cosine', 'euclidean'], default='cosine',
                       help='Similarity metric for retrieval mode')
    
    args = parser.parse_args()
    main(args)