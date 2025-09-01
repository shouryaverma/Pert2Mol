import json
import pickle
import pandas as pd
import numpy as np
import argparse
import time
from typing import Dict, List, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, Scaffolds, MACCSkeys
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import DataStructs
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

class ComprehensiveMolecularEvaluator:
    """Comprehensive molecular evaluation with multiple similarity metrics and baselines."""
    
    def __init__(self, training_smiles=None, pubchem_fingerprints_path=None, reference_csv_path=None, reference_smiles_column='canonical_smiles'):
        self.training_smiles = training_smiles or []
        self.reference_smiles = []
        self.reference_data = None
        self.fingerprint_types = ['morgan_r2', 'morgan_r3', 'maccs', 'rdk', 'atom_pairs']
        self.pubchem_fingerprints = None
        
        # Load reference CSV dataset if provided
        if reference_csv_path:
            print(f"Loading reference dataset from {reference_csv_path}...")
            try:
                self.reference_data = pd.read_csv(reference_csv_path)
                if reference_smiles_column in self.reference_data.columns:
                    self.reference_smiles = self.reference_data[reference_smiles_column].dropna().tolist()
                    print(f"Loaded {len(self.reference_smiles)} reference SMILES from CSV")
                else:
                    print(f"Warning: Column '{reference_smiles_column}' not found in CSV")
                    available_columns = list(self.reference_data.columns)
                    print(f"Available columns: {available_columns}")
            except Exception as e:
                print(f"Failed to load reference CSV: {e}")
                self.reference_data = None
        
        # Load PubChem fingerprints if provided
        if pubchem_fingerprints_path:
            print(f"Loading PubChem fingerprints from {pubchem_fingerprints_path}...")
            try:
                with open(pubchem_fingerprints_path, 'rb') as f:
                    self.pubchem_fingerprints = pickle.load(f)
                print(f"Loaded {len(self.pubchem_fingerprints)} PubChem fingerprints")
            except Exception as e:
                print(f"Failed to load PubChem fingerprints: {e}")
                self.pubchem_fingerprints = None
        
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
    
    def calculate_novelty_score(self, generated_smiles, use_pubchem=True, use_reference_csv=True):
        """Calculate novelty of generated molecule vs training set, reference CSV, or PubChem."""
        if not generated_smiles:
            return 0.0, 'none'
        
        try:
            generated_mol = Chem.MolFromSmiles(generated_smiles)
            if generated_mol is None:
                return 0.0, 'invalid'
            
            generated_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(generated_mol, 2)
            max_similarity = 0.0
            comparison_set = None
            
            # Priority: PubChem > Reference CSV > Training set
            if use_pubchem and self.pubchem_fingerprints:
                print("Using PubChem for novelty calculation...")
                comparison_set = 'pubchem'
                for pubchem_smiles, pubchem_fp in self.pubchem_fingerprints.items():
                    try:
                        sim = DataStructs.TanimotoSimilarity(generated_fp, pubchem_fp)
                        max_similarity = max(max_similarity, sim)
                        if max_similarity > 0.99:  # Early stopping for near-perfect matches
                            break
                    except:
                        continue
                        
            elif use_reference_csv and self.reference_smiles:
                comparison_set = 'reference_csv'
                for ref_smiles in self.reference_smiles:
                    try:
                        ref_mol = Chem.MolFromSmiles(ref_smiles)
                        if ref_mol is not None:
                            ref_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(ref_mol, 2)
                            sim = DataStructs.TanimotoSimilarity(generated_fp, ref_fp)
                            max_similarity = max(max_similarity, sim)
                            if max_similarity > 0.99:
                                break
                    except:
                        continue
                        
            else:
                # Fallback to training set
                comparison_set = 'training_set'
                for train_smiles in self.training_smiles:
                    try:
                        train_mol = Chem.MolFromSmiles(train_smiles)
                        if train_mol is not None:
                            train_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(train_mol, 2)
                            sim = DataStructs.TanimotoSimilarity(generated_fp, train_fp)
                            max_similarity = max(max_similarity, sim)
                    except:
                        continue
            
            novelty_score = 1.0 - max_similarity
            return novelty_score, comparison_set
            
        except Exception as e:
            print(f"Error calculating novelty: {e}")
            return 0.0, 'error'

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

def calculate_drug_likeness_score(props):
    """Calculate comprehensive drug-likeness score."""
    if not props:
        return 0.0
    
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
    
    return score

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
        ring_consistency = max(0, 1 - ring_diff / 5)
        
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

def create_evaluation_plots(results, output_dir="./plots"):
    """Create comprehensive evaluation plots."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    valid_results = [r for r in results if r.get('is_valid', False)]
    if not valid_results:
        print("No valid results to plot")
        return
    
    # 1. Confidence distribution by method
    plt.figure(figsize=(12, 8))
    methods = list(set(r.get('method', 'unknown') for r in valid_results))
    
    for i, method in enumerate(methods):
        method_results = [r for r in valid_results if r.get('method') == method]
        confidences = [r.get('confidence', 0) for r in method_results]
        
        plt.subplot(2, 2, i+1)
        plt.hist(confidences, bins=20, alpha=0.7, label=method)
        plt.xlabel('Confidence Score')
        plt.ylabel('Count')
        plt.title(f'{method.title()} Confidence Distribution')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/confidence_distributions.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Molecular properties distribution
    properties = []
    for r in valid_results:
        if r.get('molecular_properties'):
            properties.append(r['molecular_properties'])
    
    if properties:
        prop_df = pd.DataFrame(properties)
        
        plt.figure(figsize=(15, 10))
        key_props = ['MW', 'LogP', 'TPSA', 'QED', 'HBA', 'HBD', 'RotBonds', 'NumRings']
        
        for i, prop in enumerate(key_props):
            if prop in prop_df.columns:
                plt.subplot(2, 4, i+1)
                plt.hist(prop_df[prop], bins=20, alpha=0.7)
                plt.xlabel(prop)
                plt.ylabel('Count')
                plt.title(f'{prop} Distribution')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/molecular_properties.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Similarity analysis
    similarities = []
    for r in valid_results:
        if r.get('similarity_analysis'):
            similarities.append(r['similarity_analysis'])
    
    if similarities:
        sim_df = pd.DataFrame(similarities)
        
        plt.figure(figsize=(12, 8))
        sim_types = ['morgan_r2', 'maccs', 'rdk', 'scaffold_similarity']
        
        for i, sim_type in enumerate(sim_types):
            if sim_type in sim_df.columns:
                plt.subplot(2, 2, i+1)
                plt.hist(sim_df[sim_type], bins=20, alpha=0.7)
                plt.xlabel(f'{sim_type.title()} Similarity')
                plt.ylabel('Count')
                plt.title(f'{sim_type.title()} Distribution')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/similarity_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Comprehensive evaluation of generated molecules")
    parser.add_argument("--input-file", type=str, default="/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/src_new/LDMol/generated_molecules_generation_42.json",
                       help="JSON file with generated molecules from inference.py")
    parser.add_argument("--pubchem-fingerprints-path", type=str, default=None,
                       help="Path to pre-computed PubChem fingerprints pickle file")
    parser.add_argument("--reference-csv-path", type=str, default="/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/drug/PubChem/GDP_compatible/drug_basic_info.csv",
                       help="Path to reference dataset CSV with molecular data")
    parser.add_argument("--reference-smiles-column", type=str, default="canonical_smiles",
                       help="Column name containing SMILES in reference CSV")
    parser.add_argument("--use-pubchem-novelty", action="store_true",
                       help="Use PubChem dataset for novelty calculation (requires --pubchem-fingerprints-path)")
    parser.add_argument("--use-reference-csv-novelty", action="store_true",
                       help="Use reference CSV for novelty calculation (requires --reference-csv-path)")
    parser.add_argument("--output-prefix", type=str, default="evaluation",
                       help="Prefix for output files")
    parser.add_argument("--create-plots", action="store_true",
                       help="Create evaluation plots")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    print(f"Loading generated molecules from {args.input_file}...")
    
    # Load generated molecules
    with open(args.input_file, 'r') as f:
        data = json.load(f)
    
    metadata = data['metadata']
    training_data = data.get('training_data', {'smiles': []})
    results = data['results']
    
    print(f"Loaded {len(results)} generated molecules")
    print(f"Generation method: {metadata['inference_mode']}")
    print(f"Valid molecules: {metadata['valid_molecules']}/{metadata['total_samples']}")
    
    # Initialize evaluator
    evaluator = ComprehensiveMolecularEvaluator(
        training_smiles=training_data['smiles'],
        pubchem_fingerprints_path=args.pubchem_fingerprints_path,
        reference_csv_path=args.reference_csv_path,
        reference_smiles_column=args.reference_smiles_column
    )
    
    # Process each result
    print("Calculating comprehensive molecular evaluations...")
    
    valid_results = [r for r in results if r.get('is_valid', False)]
    
    start_time = time.time()
    
    for i, result in enumerate(tqdm(valid_results, desc="Evaluating molecules")):
        target_smiles = result['target_smiles']
        generated_smiles = result['generated_smiles']
        compound_name = result['compound_name']
        
        # Multi-fingerprint similarity analysis
        similarity_analysis = evaluator.calculate_multi_fingerprint_similarity(
            target_smiles, generated_smiles
        )
        similarity_analysis['scaffold_similarity'] = evaluator.calculate_scaffold_similarity(
            target_smiles, generated_smiles
        )
        result['similarity_analysis'] = similarity_analysis
        
        # Novelty analysis
        use_pubchem = args.use_pubchem_novelty and evaluator.pubchem_fingerprints is not None
        use_reference_csv = args.use_reference_csv_novelty and evaluator.reference_smiles
        
        novelty_score, novelty_method = evaluator.calculate_novelty_score(
            generated_smiles, 
            use_pubchem=use_pubchem,
            use_reference_csv=use_reference_csv
        )
        result['novelty_score'] = novelty_score
        result['novelty_method'] = novelty_method
        
        # Molecular properties
        props = calculate_comprehensive_molecular_properties(generated_smiles)
        result['molecular_properties'] = props
        
        # Drug-likeness score
        result['druglikeness_score'] = calculate_drug_likeness_score(props)
        
        # Mechanism consistency
        result['mechanism_analysis'] = analyze_mechanism_consistency(
            compound_name, generated_smiles, target_smiles
        )
        
        if args.verbose and i < 5:
            print(f"\nExample {i+1}:")
            print(f"  Target: {target_smiles}")
            print(f"  Generated: {generated_smiles}")
            print(f"  Morgan similarity: {similarity_analysis.get('morgan_r2', 0):.3f}")
            print(f"  Novelty: {result['novelty_score']:.3f}")
            print(f"  Drug-likeness: {result['druglikeness_score']:.3f}")
    
    # Diversity analysis
    all_generated_smiles = [r['generated_smiles'] for r in valid_results]
    diversity_metrics = diversity_analysis(all_generated_smiles)
    
    # Calculate comprehensive statistics
    evaluation_time = time.time() - start_time
    
    # Method distribution
    method_counts = Counter(r.get('method', 'unknown') for r in results)
    
    # Similarity statistics
    similarity_stats = {}
    for sim_type in ['morgan_r2', 'maccs', 'rdk', 'atom_pairs', 'scaffold_similarity']:
        values = [r['similarity_analysis'].get(sim_type, 0) for r in valid_results 
                 if 'similarity_analysis' in r]
        if values:
            similarity_stats[sim_type] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
    
    # Novelty and drug-likeness statistics
    novelty_scores = [r.get('novelty_score', 0) for r in valid_results]
    novelty_methods = [r.get('novelty_method', 'unknown') for r in valid_results]
    druglikeness_scores = [r.get('druglikeness_score', 0) for r in valid_results]
    
    # Count novelty methods used
    novelty_method_counts = Counter(novelty_methods)
    
    novelty_stats = {
        'mean': np.mean(novelty_scores),
        'std': np.std(novelty_scores),
        'min': np.min(novelty_scores),
        'max': np.max(novelty_scores),
        'method_counts': dict(novelty_method_counts)
    } if novelty_scores else {}
    
    druglikeness_stats = {
        'mean': np.mean(druglikeness_scores),
        'std': np.std(druglikeness_scores),
        'min': np.min(druglikeness_scores),
        'max': np.max(druglikeness_scores)
    } if druglikeness_scores else {}
    
    # Molecular property statistics
    property_stats = {}
    molecular_properties = [r.get('molecular_properties', {}) for r in valid_results]
    molecular_properties = [p for p in molecular_properties if p]
    
    if molecular_properties:
        extended_props = ['MW', 'LogP', 'HBA', 'HBD', 'TPSA', 'QED', 'BertzCT', 'NumRings']
        for prop in extended_props:
            values = [p.get(prop) for p in molecular_properties if p.get(prop) is not None]
            if values:
                property_stats[prop] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
    
    # Mechanism consistency
    mechanism_scores = [r['mechanism_analysis']['mechanism_consistency_score'] 
                       for r in valid_results if r.get('mechanism_analysis')]
    mechanism_stats = {
        'mean': np.mean(mechanism_scores),
        'std': np.std(mechanism_scores)
    } if mechanism_scores else {}
    
    # Create comprehensive summary
    comprehensive_summary = {
        'evaluation_metadata': {
            'input_file': args.input_file,
            'evaluation_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'evaluation_time_seconds': evaluation_time,
            'pubchem_novelty_used': args.use_pubchem_novelty and evaluator.pubchem_fingerprints is not None,
            'pubchem_fingerprints_count': len(evaluator.pubchem_fingerprints) if evaluator.pubchem_fingerprints else 0
        },
        'generation_metadata': metadata,
        'evaluation_statistics': {
            'total_samples': len(results),
            'valid_molecules': len(valid_results),
            'validity_rate': len(valid_results) / len(results) if results else 0,
            'method_distribution': dict(method_counts)
        },
        'similarity_analysis': similarity_stats,
        'novelty_analysis': {
            'statistics': novelty_stats,
            'primary_method': novelty_method_counts.most_common(1)[0][0] if novelty_method_counts else 'none',
            'method_distribution': dict(novelty_method_counts) if novelty_method_counts else {},
            'reference_datasets': {
                'pubchem_available': evaluator.pubchem_fingerprints is not None,
                'pubchem_count': len(evaluator.pubchem_fingerprints) if evaluator.pubchem_fingerprints else 0,
                'reference_csv_available': bool(evaluator.reference_smiles),
                'reference_csv_count': len(evaluator.reference_smiles),
                'training_set_count': len(evaluator.training_smiles)
            }
        },
        'druglikeness_analysis': druglikeness_stats,
        'diversity_metrics': diversity_metrics,
        'property_statistics': property_stats,
        'mechanism_consistency': mechanism_stats,
        'evaluated_results': valid_results
    }
    
    # Save comprehensive results
    output_json = f"{args.output_prefix}_{metadata['inference_mode']}_{metadata['global_seed']}_comprehensive.json"
    
    with open(output_json, 'w') as f:
        json.dump(comprehensive_summary, f, indent=2, default=str)
    
    # Create human-readable summary
    output_txt = f"{args.output_prefix}_{metadata['inference_mode']}_{metadata['global_seed']}_summary.txt"
    
    with open(output_txt, 'w') as f:
        f.write("COMPREHENSIVE MOLECULAR EVALUATION REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Input file: {args.input_file}\n")
        f.write(f"Generation mode: {metadata['inference_mode']}\n")
        f.write(f"Evaluation time: {evaluation_time:.1f}s\n\n")
        
        f.write("DATASET STATISTICS:\n")
        f.write(f"  Total samples: {len(results)}\n")
        f.write(f"  Valid molecules: {len(valid_results)} ({len(valid_results)/len(results)*100:.1f}%)\n\n")
        
        f.write("METHOD DISTRIBUTION:\n")
        for method, count in method_counts.items():
            f.write(f"  {method}: {count} ({count/len(results)*100:.1f}%)\n")
        f.write("\n")
        
        f.write("SIMILARITY ANALYSIS:\n")
        for sim_type, stats in similarity_stats.items():
            f.write(f"  {sim_type.upper()}: {stats['mean']:.3f} ± {stats['std']:.3f}\n")
        f.write("\n")
        
        f.write("NOVELTY ANALYSIS:\n")
        if novelty_stats:
            primary_method = novelty_stats.get('method_counts', {})
            if primary_method:
                most_common_method = max(primary_method.items(), key=lambda x: x[1])[0]
                f.write(f"  Primary method: {most_common_method}\n")
                f.write(f"  Method distribution:\n")
                for method, count in primary_method.items():
                    f.write(f"    {method}: {count} samples\n")
            f.write(f"  Mean novelty: {novelty_stats['mean']:.3f} ± {novelty_stats['std']:.3f}\n")
            f.write(f"  Range: {novelty_stats['min']:.3f} - {novelty_stats['max']:.3f}\n")
            f.write(f"  Reference datasets:\n")
            if evaluator.pubchem_fingerprints:
                f.write(f"    PubChem: {len(evaluator.pubchem_fingerprints)} molecules\n")
            if evaluator.reference_smiles:
                f.write(f"    Reference CSV: {len(evaluator.reference_smiles)} molecules\n")
            f.write(f"    Training set: {len(evaluator.training_smiles)} molecules\n")
        f.write("\n")
        
        f.write("DRUG-LIKENESS ANALYSIS:\n")
        if druglikeness_stats:
            f.write(f"  Mean drug-likeness: {druglikeness_stats['mean']:.3f} ± {druglikeness_stats['std']:.3f}\n")
            f.write(f"  Range: {druglikeness_stats['min']:.3f} - {druglikeness_stats['max']:.3f}\n\n")
        
        f.write("DIVERSITY METRICS:\n")
        f.write(f"  Internal diversity: {diversity_metrics.get('internal_diversity', 0):.3f}\n")
        f.write(f"  Scaffold diversity: {diversity_metrics.get('scaffold_diversity', 0):.3f}\n")
        f.write(f"  Property diversity: {diversity_metrics.get('property_diversity', 0):.3f}\n")
        f.write(f"  Unique scaffolds: {diversity_metrics.get('num_unique_scaffolds', 0)}\n\n")
        
        if mechanism_stats:
            f.write("MECHANISM CONSISTENCY:\n")
            f.write(f"  Mean consistency: {mechanism_stats['mean']:.3f} ± {mechanism_stats['std']:.3f}\n\n")
        
        f.write("MOLECULAR PROPERTY STATISTICS:\n")
        for prop, stats in property_stats.items():
            f.write(f"  {prop}: {stats['mean']:.2f} ± {stats['std']:.2f}\n")
    
    # Create plots if requested
    if args.create_plots:
        print("Creating evaluation plots...")
        create_evaluation_plots(valid_results)
    
    # Print summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE MOLECULAR EVALUATION COMPLETE")
    print(f"{'='*80}")
    print(f"Evaluated {len(valid_results)} valid molecules in {evaluation_time:.1f}s")
    print(f"Results saved:")
    print(f"  Comprehensive: {output_json}")
    print(f"  Summary: {output_txt}")
    if args.create_plots:
        print(f"  Plots: ./plots/")
    
    if novelty_stats:
        novelty_method = "PubChem" if args.use_pubchem_novelty and evaluator.pubchem_fingerprints else "training set"
        print(f"\nKey Results:")
        print(f"  Average novelty ({novelty_method}): {novelty_stats['mean']:.3f}")
        print(f"  Average drug-likeness: {druglikeness_stats.get('mean', 0):.3f}")
        print(f"  Internal diversity: {diversity_metrics.get('internal_diversity', 0):.3f}")

if __name__ == "__main__":
    main()