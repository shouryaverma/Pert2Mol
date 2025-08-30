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
    """Sample with proper classifier-free guidance for separate modalities."""
    if device is None:
        device = next(model.parameters()).device
    
    batch_size = shape[0]
    x = torch.randn(*shape, device=device)
    dt = 1.0 / num_steps
    
    # Pre-compute conditioning variants
    y_rna_only = y_full.clone()
    y_rna_only[:, :, :128] = 0  # Zero out image features (first 128 dims)
    
    y_image_only = y_full.clone() 
    y_image_only[:, :, 128:] = 0  # Zero out RNA features (last 64 dims)
    
    y_uncond = torch.zeros_like(y_full)  # No conditioning
    
    for i in range(num_steps):
        t = torch.full((batch_size,), i * dt, device=device)
        t_discrete = (t * 999).long()
        
        with torch.no_grad():
            # Get predictions for all conditioning variants
            cond_velocity = model(x, t_discrete, y=y_full, pad_mask=pad_mask)
            rna_velocity = model(x, t_discrete, y=y_rna_only, pad_mask=pad_mask)  
            image_velocity = model(x, t_discrete, y=y_image_only, pad_mask=pad_mask)
            uncond_velocity = model(x, t_discrete, y=y_uncond, pad_mask=pad_mask)
            
            # Handle learn_sigma case
            if cond_velocity.shape[1] == 2 * x.shape[1]:
                cond_velocity, _ = torch.split(cond_velocity, x.shape[1], dim=1)
                rna_velocity, _ = torch.split(rna_velocity, x.shape[1], dim=1)
                image_velocity, _ = torch.split(image_velocity, x.shape[1], dim=1)
                uncond_velocity, _ = torch.split(uncond_velocity, x.shape[1], dim=1)
            
            # Apply separate guidance for each modality
            # Method 1: Additive guidance
            velocity = (uncond_velocity + 
                       cfg_scale_rna * (rna_velocity - uncond_velocity) +
                       cfg_scale_image * (image_velocity - uncond_velocity))
            
            # Alternative Method 2: Compositional guidance (uncomment to use)
            # rna_contribution = cfg_scale_rna * (rna_velocity - uncond_velocity)
            # image_contribution = cfg_scale_image * (image_velocity - uncond_velocity)
            # velocity = uncond_velocity + 0.5 * (rna_contribution + image_contribution)
                
        x = x + dt * velocity 
    return x


@torch.no_grad()
def sample_with_advanced_cfg(model, flow, shape, y_full, pad_mask,
                            cfg_scale_rna=2.0, cfg_scale_image=2.0, 
                            cfg_schedule='constant', num_steps=50, device=None):
    """Advanced CFG with scheduling and better control."""
    if device is None:
        device = next(model.parameters()).device
    
    batch_size = shape[0]
    x = torch.randn(*shape, device=device)
    dt = 1.0 / num_steps
    
    # Pre-compute conditioning variants
    y_rna_only = y_full.clone()
    y_rna_only[:, :, :128] = 0
    
    y_image_only = y_full.clone() 
    y_image_only[:, :, 128:] = 0
    
    y_uncond = torch.zeros_like(y_full)
    
    for i in range(num_steps):
        t = torch.full((batch_size,), i * dt, device=device)
        t_discrete = (t * 999).long()
        
        # Dynamic CFG scaling based on timestep
        if cfg_schedule == 'linear_decay':
            # Start high, decay to 1.0
            progress = i / num_steps
            current_cfg_rna = cfg_scale_rna * (1.0 - progress) + 1.0 * progress
            current_cfg_image = cfg_scale_image * (1.0 - progress) + 1.0 * progress
        elif cfg_schedule == 'cosine':
            # Cosine decay
            progress = i / num_steps
            current_cfg_rna = 1.0 + (cfg_scale_rna - 1.0) * 0.5 * (1 + math.cos(math.pi * progress))
            current_cfg_image = 1.0 + (cfg_scale_image - 1.0) * 0.5 * (1 + math.cos(math.pi * progress))
        else:  # constant
            current_cfg_rna = cfg_scale_rna
            current_cfg_image = cfg_scale_image
        
        with torch.no_grad():
            # Batch all predictions for efficiency
            y_batch = torch.cat([y_full, y_rna_only, y_image_only, y_uncond], dim=0)
            pad_mask_batch = pad_mask.repeat(4, 1)
            x_batch = x.repeat(4, 1, 1, 1)
            
            # Single forward pass for all variants
            velocity_batch = model(x_batch, t_discrete.repeat(4), y=y_batch, pad_mask=pad_mask_batch)
            
            if velocity_batch.shape[1] == 2 * x.shape[1]:
                velocity_batch, _ = torch.split(velocity_batch, x.shape[1], dim=1)
            
            # Split back to individual predictions
            cond_velocity, rna_velocity, image_velocity, uncond_velocity = torch.chunk(velocity_batch, 4, dim=0)
            
            # Apply guidance
            velocity = (uncond_velocity + 
                       current_cfg_rna * (rna_velocity - uncond_velocity) +
                       current_cfg_image * (image_velocity - uncond_velocity))
                
        x = x + dt * velocity 
    return x

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
    training_smiles = []
    training_rna_signatures = []
    
    # Initialize evaluators (will collect training data during inference)
    mol_evaluator = ComprehensiveMolecularEvaluator(training_smiles=[])
    baseline_generator = BaselineGenerator([])

    # Initialize results storage
    results = []
    all_generated_smiles = []
    molecular_properties = []
    training_smiles = []
    training_rna_signatures = []

    output_file = f'./inference_results_{args.global_seed}.txt'
    detailed_output_file = f'./detailed_results_{args.global_seed}.json'

    # header
    with open(output_file, 'w') as f:
        f.write("target_smiles\tbest_generated\tcompound_name\tvalidity\t"
            "morgan_r2_sim\tmaccs_sim\trdk_sim\tscaffold_sim\t"
            "novelty_score\tdrug_score\tmechanism_consistency\tnum_candidates\n")

    start_time = time.time()

    # Generate molecules with evaluation (collect training data on-the-fly)
    for batch_idx, batch in enumerate(tqdm(loader, desc="molecular generation")):
        if batch_idx >= args.max_batches and args.max_batches > 0:
            break
            
        # Encode features
        control_imgs = batch['control_images']
        treatment_imgs = batch['treatment_images']
        control_rna = batch['control_transcriptomics']
        treatment_rna = batch['treatment_transcriptomics']
        target_smiles = batch['target_smiles']
        compound_names = batch['compound_name']
        training_smiles.extend(target_smiles)

        # Update the molecular evaluator with new training data
        mol_evaluator.training_smiles = training_smiles

        # Update baseline generator with collected data
        baseline_generator.training_smiles = training_smiles

        y_features, pad_mask = dual_rna_image_encoder(
            control_imgs, treatment_imgs, control_rna, treatment_rna,
            image_encoder, rna_encoder, device,
            rna_dropout_prob=0.0, image_dropout_prob=0.0, training=False
        )
        
        batch_size = y_features.shape[0]
        shape = (batch_size, in_channels, latent_size, 1)
        
        # Generate multiple candidates per sample
        all_generated = []
        for sample_idx in range(args.num_samples_per_condition):

            samples = sample_with_cfg(
                model=model, flow=flow, shape=shape, y_full=y_features, 
                pad_mask=pad_mask, cfg_scale_rna=2.0, cfg_scale_image=2.0,
                num_steps=args.num_sampling_steps, device=device
            )
            
            samples = samples.squeeze(-1).permute((0, 2, 1))
            generated_smiles = AE_SMILES_decoder(samples, ae_model, stochastic=False, k=1)
            all_generated.append(generated_smiles)
        
        # Process each sample in the batch
        for i in range(batch_size):
            candidates = [gen[i] for gen in all_generated]
            target = target_smiles[i]
            compound = compound_names[i]
            
            # Validate candidates
            valid_candidates = []
            for cand in candidates:
                try:
                    if cand and cand != "":
                        mol = Chem.MolFromSmiles(cand)
                        if mol is not None:
                            canonical = Chem.MolToSmiles(mol, canonical=True)
                            valid_candidates.append(canonical)
                except:
                    continue
            
            # Remove duplicates
            valid_candidates = list(dict.fromkeys(valid_candidates))
            all_generated_smiles.extend(valid_candidates)
            
            # Rank by comprehensive drug-likeness
            if valid_candidates:
                ranked_candidates = rank_candidates_by_comprehensive_score(valid_candidates)
                if ranked_candidates:
                    best_candidate = ranked_candidates[0][0]
                    best_score = ranked_candidates[0][1]
                    best_props = ranked_candidates[0][2]
                    validity = 1
                else:
                    best_candidate = valid_candidates[0]
                    best_score = 0
                    best_props = {}
                    validity = 1
            else:
                best_candidate = ""
                best_score = 0
                best_props = {}
                validity = 0
            
            # similarity analysis
            similarity_results = {}
            if validity and target:
                similarity_results = mol_evaluator.calculate_multi_fingerprint_similarity(
                    target, best_candidate
                )
                similarity_results['scaffold_similarity'] = mol_evaluator.calculate_scaffold_similarity(
                    target, best_candidate
                )
                similarity_results['substructure_analysis'] = mol_evaluator.analyze_substructure_overlap(
                    target, best_candidate
                )
            
            # Novelty analysis
            novelty_score = mol_evaluator.calculate_novelty_score(best_candidate) if validity else 0.0
            
            # Mechanism consistency
            mechanism_analysis = analyze_mechanism_consistency(compound, best_candidate, target)

            # Diversity analysis
            diversity_results = diversity_analysis(valid_candidates)
            
            # Generate baselines for comparison
            random_baseline = baseline_generator.generate_random_baseline(5)
            structure_baseline = baseline_generator.generate_structure_similar_baseline(target, 3)
            
            # Store comprehensive results
            result = {
                'target_smiles': target,
                'generated_smiles': best_candidate,
                'compound_name': compound,
                'all_candidates': valid_candidates,
                'validity': validity,
                'similarity_analysis': similarity_results,
                'novelty_score': novelty_score,
                'druglikeness_score': best_score,
                'molecular_properties': best_props,
                'diversity_analysis': diversity_results,
                'mechanism_analysis': mechanism_analysis,
                'baselines': {
                    'random': random_baseline,
                    'structure_similar': structure_baseline
                },
                'num_candidates': len(valid_candidates)
            }
            results.append(result)
            
            if best_props:
                molecular_properties.append(best_props)
            
            # Write output
            with open(output_file, 'a') as f:
                morgan_sim = similarity_results.get('morgan_r2', 0.0)
                maccs_sim = similarity_results.get('maccs', 0.0)
                rdk_sim = similarity_results.get('rdk', 0.0)
                scaffold_sim = similarity_results.get('scaffold_similarity', 0.0)
                mechanism_score = mechanism_analysis.get('mechanism_consistency_score', 0.0)
                
                f.write(f"{target}\t{best_candidate}\t{compound}\t{validity}\t"
                       f"{morgan_sim:.3f}\t{maccs_sim:.3f}\t{rdk_sim:.3f}\t{scaffold_sim:.3f}\t"
                       f"{novelty_score:.3f}\t{best_score:.3f}\t{mechanism_score:.3f}\t"
                       f"{len(valid_candidates)}\n")

    # Calculate comprehensive metrics
    total_time = time.time() - start_time
    valid_results = [r for r in results if r['validity'] == 1]
    validity_rate = len(valid_results) / len(results) if results else 0
    
    # Multi-fingerprint similarity averages
    similarity_averages = {}
    for sim_type in ['morgan_r2', 'maccs', 'rdk', 'atom_pairs']:
        values = [r['similarity_analysis'].get(sim_type, 0) for r in valid_results 
                 if 'similarity_analysis' in r]
        similarity_averages[sim_type] = np.mean(values) if values else 0.0
    
    avg_novelty = np.mean([r['novelty_score'] for r in valid_results]) if valid_results else 0
    avg_druglikeness = np.mean([r['druglikeness_score'] for r in valid_results]) if valid_results else 0
    
    # property statistics
    property_stats = {}
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
    
    # Print results
    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE MOLECULAR GENERATION EVALUATION")
    print(f"{'='*100}")
    print(f"Dataset Statistics:")
    print(f"  Total samples: {len(results)}")
    print(f"  Valid molecules: {len(valid_results)}/{len(results)} ({validity_rate:.1%})")
    print(f"  Processing time: {total_time:.1f}s ({total_time/len(results):.2f}s/sample)")
    
    print(f"\nMulti-Fingerprint Similarity Analysis:")
    for sim_type, avg_sim in similarity_averages.items():
        print(f"  {sim_type.upper()}: {avg_sim:.3f}")
    
    print(f"\nNovelty and Drug-likeness:")
    print(f"  Average novelty score: {avg_novelty:.3f}")
    print(f"  Average drug-likeness score: {avg_druglikeness:.3f}")
    
    print(f"\nDiversity Analysis:")
    overall_diversity = diversity_analysis(all_generated_smiles)
    print(f"  Internal diversity: {overall_diversity.get('internal_diversity', 0):.3f}")
    print(f"  Scaffold diversity: {overall_diversity.get('scaffold_diversity', 0):.3f}")
    print(f"  Property diversity: {overall_diversity.get('property_diversity', 0):.3f}")
    print(f"  Unique scaffolds: {overall_diversity.get('num_unique_scaffolds', 0)}")
    
    print(f"\nMechanism Consistency Analysis:")
    mechanism_scores = [r['mechanism_analysis']['mechanism_consistency_score'] 
                       for r in valid_results if 'mechanism_analysis' in r]
    if mechanism_scores:
        print(f"  Average mechanism consistency: {np.mean(mechanism_scores):.3f}")
    
    print(f"\nExtended Molecular Property Statistics:")
    for prop, stats in property_stats.items():
        print(f"  {prop}: {stats['mean']:.2f} ± {stats['std']:.2f} "
              f"(range: {stats['min']:.2f}-{stats['max']:.2f})")
    
    print(f"\nTop Generated Examples (by drug-likeness):")
    sorted_results = sorted(valid_results, key=lambda x: x['druglikeness_score'], reverse=True)
    
    for i, result in enumerate(sorted_results[:3]):
        print(f"\nExample {i+1}:")
        print(f"  Target compound: {result['compound_name']}")
        print(f"  Generated: {result['generated_smiles']}")
        print(f"  Target:    {result['target_smiles']}")
        
        sim_analysis = result.get('similarity_analysis', {})
        print(f"  Similarities: Morgan={sim_analysis.get('morgan_r2', 0):.3f}, "
              f"MACCS={sim_analysis.get('maccs', 0):.3f}, "
              f"Scaffold={sim_analysis.get('scaffold_similarity', 0):.3f}")
        
        print(f"  Novelty: {result['novelty_score']:.3f}")
        print(f"  Drug-likeness: {result['druglikeness_score']:.3f}")
        
        if result['molecular_properties']:
            props = result['molecular_properties']
            print(f"  Properties: MW={props['MW']:.1f}, LogP={props['LogP']:.2f}, "
                  f"QED={props['QED']:.2f}, Complexity={props.get('BertzCT', 0):.1f}")
    
    print(f"\n{'='*100}")
    print(f"Results saved to: {output_file}")
    
    # Save comprehensive detailed results
    comprehensive_summary = {
        'evaluation_metadata': {
            'evaluation_type': 'comprehensive_molecular_generation',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'parameters': {
                'num_samples_per_condition': args.num_samples_per_condition,
                'num_sampling_steps': args.num_sampling_steps,
                'global_seed': args.global_seed
            }
        },
        'generation_statistics': {
            'total_samples': len(results),
            'valid_molecules': len(valid_results),
            'validity_rate': validity_rate,
            'processing_time': total_time
        },
        'similarity_analysis': similarity_averages,
        'novelty_and_druglikeness': {
            'average_novelty': avg_novelty,
            'average_druglikeness': avg_druglikeness,
        },
        'diversity_metrics': overall_diversity,
        'property_statistics': property_stats,
        'mechanism_analysis': {
            'average_mechanism_consistency': np.mean(mechanism_scores) if mechanism_scores else 0.0
        },
        'top_examples': sorted_results[:10] if sorted_results else [],
        'training_set_info': {
            'num_training_molecules': len(training_smiles),
            'unique_training_molecules': len(set(training_smiles))
        }
    }
    
    with open(detailed_output_file, 'w') as f:
        json.dump(comprehensive_summary, f, indent=2, default=str)
    
    print(f"Comprehensive results saved to: {detailed_output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="LDMol")
    parser.add_argument("--ckpt", type=str, default='/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/src_new/LDMol/results/004-LDMol/checkpoints/0040000.pt')
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
    parser.add_argument("--num-samples-per-condition", type=int, default=8)
    parser.add_argument("--num-sampling-steps", type=int, default=50)
    parser.add_argument("--max-batches", type=int, default=50)
    parser.add_argument("--global-seed", type=int, default=42)
    parser.add_argument("--debug-mode", action="store_true")
    
    args = parser.parse_args()
    main(args)