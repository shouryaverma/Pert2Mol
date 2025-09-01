import sys
import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import tifffile
import pickle
import h5py
import json
import logging
from typing import Dict, List, Optional, Callable, Tuple
from torchvision import transforms
import scanpy as sc
import anndata as ad
import networkx as nx
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, RDKFingerprint
from datasets import load_dataset
import warnings
os.environ['RDKit_SILENCE_WARNINGS'] = '1'
import rdkit
rdkit.rdBase.DisableLog('rdApp.*')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_array(data, name, allow_negative=True, max_abs_value=1e6):
    """Comprehensive data validation helper."""
    if data is None:
        raise ValueError(f"{name}: Data is None")
    
    # Check for NaNs
    if np.isnan(data).any():
        nan_count = np.isnan(data).sum()
        raise ValueError(f"{name}: Contains {nan_count} NaN values")
    
    # Check for infinite values
    if np.isinf(data).any():
        inf_count = np.isinf(data).sum()
        raise ValueError(f"{name}: Contains {inf_count} infinite values")
    
    # Check for all zeros (might indicate loading issues)
    if np.all(data == 0):
        raise ValueError(f"{name}: All values are zero")
    
    return True

def scale_down_images(images: np.ndarray, target_size: int) -> np.ndarray:
    """
    Scale down multi-channel images to a given square pixel size using bilinear interpolation.
    
    Args:
        images: numpy array of shape (channels, height, width)
        target_size: int, target size for height and width (square)
        
    Returns:
        numpy array of scaled images with shape (channels, target_size, target_size)
    """
    # Convert numpy array to torch tensor
    images_tensor = torch.tensor(images)
    
    # Add batch dimension and ensure float32
    images_tensor = images_tensor.unsqueeze(0).float()  # (1, C, H, W)
    
    # Resize using bilinear interpolation
    images_resized = F.interpolate(
        images_tensor, 
        size=(target_size, target_size), 
        mode='bilinear',
        align_corners=False
    )
    
    # Remove batch dimension
    images_resized = images_resized.squeeze(0)
    
    # Convert back to numpy
    images_resized_np = images_resized.numpy()
    
    return images_resized_np


class HistologyTranscriptomicsDataset(Dataset):
    """
    Custom Dataset for paired histology images and transcriptomics data
    with drug treatment conditioning.
    """
    
    def __init__(self, 
                 metadata_control: pd.DataFrame,
                 metadata_drug: pd.DataFrame, 
                 gene_count_matrix: pd.DataFrame = None,
                 image_json_dict: Dict[str, List[str]] = None,
                 transform: Optional[Callable] = None,
                 target_size: Optional[int] = None,
                 ):
        """
        Args:
            metadata_control: Control dataset metadata with columns ['cell_line', 'sample_id', 'json_key']
            metadata_drug: Treatment dataset metadata with columns ['cell_line', 'compound', 'timepoint', 
                          'compound_concentration_in_uM', 'sample_id', 'json_key']
            gene_count_matrix: Transcriptomics data with sample_id as columns and genes as rows
            image_json_dict: Dictionary mapping json_key to list of image paths
            transform: Optional transform to be applied on images
            target_size: Optional target size for image resizing
        """
        self.metadata_control = metadata_control
        self.metadata_drug = metadata_drug
        self.gene_count_matrix = gene_count_matrix
        self.image_json_dict = image_json_dict
        self.transform = transform
        self.target_size = target_size

        logger.info(f"`self.metadata_drug.columns`={self.metadata_drug.columns.tolist()}")

        # Convert relevant columns to appropriate types
        for k in ['sample_id', 'cell_line', 'json_key', 'compound']:
            if k in self.metadata_control.columns:
                self.metadata_control[k] = self.metadata_control[k].astype(str)
            if k in self.metadata_drug.columns:
                self.metadata_drug[k] = self.metadata_drug[k].astype(str)
        
        for k in ['timepoint', 'compound_concentration_in_uM']:
            if k in self.metadata_control.columns:
                self.metadata_control[k] = self.metadata_control[k].astype(float)
            if k in self.metadata_drug.columns:
                self.metadata_drug[k] = self.metadata_drug[k].astype(float)

        # Group control metadata by cell_line for efficient sampling
        self.control_grouped = self.metadata_control.groupby('cell_line')
        
        # Get available cell lines in both datasets
        control_cell_lines = set(self.metadata_control['cell_line'].unique())
        drug_cell_lines = set(self.metadata_drug['cell_line'].unique())
        self.common_cell_lines = control_cell_lines.intersection(drug_cell_lines)
        
        # Filter drug metadata to only include samples with matching control cell lines
        self.filtered_drug_metadata = self.metadata_drug[
            self.metadata_drug['cell_line'].isin(self.common_cell_lines)
        ].reset_index(drop=True)
        
        logger.info(f"Dataset initialized with {len(self.filtered_drug_metadata)} treatment samples")
        logger.info(f"Common cell lines: {len(self.common_cell_lines)}")
    
    def __len__(self):
        return len(self.filtered_drug_metadata)
    
    def load_multi_channel_images(self, json_key: str) -> np.ndarray:
        """
        Load all TIFF images for a sample and concatenate as 3D array.
        
        Args:
            json_key: Key to locate image paths in the JSON dictionary
            
        Returns:
            3D numpy array of shape (channels, height, width)
        """
        image_paths = self.image_json_dict.get(json_key, [])
        if not image_paths:
            logger.debug(f"No images found for json_key: \"{json_key}\"")
            return np.zeros((4, self.target_size or 512, self.target_size or 512), dtype=np.float32)
        
        # Sort paths to ensure consistent channel order (w1, w2, w3, w4)
        image_paths = sorted(image_paths)
        
        images = []
        for i, path in enumerate(image_paths):
            try:
                img = tifffile.imread(path)
                # Ensure 2D image (H, W)
                if img.ndim > 2:
                    img = img.squeeze()

                images.append(img)
                
                # Log channel information for debugging
                channel_info = "w1=Blue" if "w1" in path else \
                              "w2=Green" if "w2" in path else \
                              "w3=Red" if "w3" in path else \
                              "w4=DeepRed" if "w4" in path else "Unknown"
                logger.debug(f"Loaded channel {i}: {channel_info} from {path}")
                
            except Exception as e:
                logger.error(f"Error loading image {path}: {e}")
                raise
        
        # Stack images along the channel dimension to form (C, H, W)
        images = np.stack(images, axis=0).astype(np.float32)
        
        # Scale images if target_size is provided
        if self.target_size is not None:
            images = scale_down_images(images, self.target_size)
        
        # Apply transform if provided
        if self.transform:
            images = self.transform(images)
        
        return images
    
    def get_transcriptomics_data(self, sample_id: str, normalize: bool = False, 
                                 ran_default_size: int=128) -> np.ndarray:
        """
        Extract transcriptomics data for a given sample_id.
        
        Args:
            sample_id: Sample identifier
            
        Returns:
            1D numpy array of gene expression values
        """
        if sample_id not in self.gene_count_matrix.columns:
            logger.error(f"self.gene_count_matrix.columns[:10]={self.gene_count_matrix.columns[:10].tolist()}")
            logger.error(f"\"{sample_id}\" in self.gene_count_matrix.columns={sample_id in self.gene_count_matrix.columns}")
            logger.error(f"gene_count_matrix.shape={self.gene_count_matrix.shape}")
            logger.warning(f"Sample ID {sample_id} not found in gene count matrix")
            return np.zeros((ran_default_size,), dtype=np.float32)
        
        if not normalize:
            return self.gene_count_matrix[sample_id].values.astype(np.float32)
        
        logger.warning("[DEPRECATED] Normalization in get_transcriptomics_data is deprecated. Use external preprocessing.")
        raw_data = self.gene_count_matrix[sample_id].values.astype(np.float32)
        
        # Apply the SAME transformation as the encoder
        log_data = np.log1p(raw_data)  # Log transform

        # Normalize (store global stats for consistency)
        if not hasattr(self, 'global_mean'):
            # Compute global statistics once
            all_log_data = np.log1p(self.gene_count_matrix.values)
            self.global_mean = np.mean(all_log_data)
            self.global_std = np.std(all_log_data)
        
        normalized_data = (log_data - self.global_mean) / (self.global_std + 1e-8)
        
        return normalized_data.astype(np.float32)  # Ensure float32 for consistency
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a paired sample (control + treatment) with conditioning information.
        
        Args:
            idx: Index of the treatment sample
            
        Returns:
            Dictionary containing paired data and conditioning information
        """
        # Get treatment sample metadata
        treatment_sample = self.filtered_drug_metadata.iloc[idx]
        cell_line = treatment_sample['cell_line']
        
        # logger.debug(f"treatment_sample['sample_id']=\"{treatment_sample['sample_id']}\"")
        logger.debug(f"treatment_sample={treatment_sample.to_dict()}")
        
        # Sample a random control sample with the same cell_line
        control_samples = self.control_grouped.get_group(cell_line)
        control_sample = control_samples.sample(n=1).iloc[0]
        # logger.debug(f"control_sample=\"{control_sample['sample_id']}\"")
        logger.debug(f"control_sample={control_sample.to_dict()}")
        
        # Load transcriptomics data
        treatment_transcriptomics = self.get_transcriptomics_data(treatment_sample['sample_id'])
        control_transcriptomics = self.get_transcriptomics_data(control_sample['sample_id'])
        
        # Load multi-channel images
        treatment_images = self.load_multi_channel_images(treatment_sample.get('json_key', ''))
        control_images = self.load_multi_channel_images(control_sample.get('json_key',''))
        
        # Prepare conditioning information
        conditioning_info = {
            'treatment': treatment_sample['compound'],
            'cell_line': cell_line,
            'timepoint': treatment_sample.get('timepoint', 24.0),
            'compound_concentration_in_uM': treatment_sample.get('compound_concentration_in_uM', 1.)
        }

        if np.isnan(conditioning_info.get('timepoint',24.)) or np.isnan(conditioning_info.get('compound_concentration_in_uM', 1.)):
            raise ValueError(f"NaN in conditioning info: {conditioning_info}")

        # Return paired data as tensors (CORRECTED - fixed image assignment)
        return {
            'control_transcriptomics': torch.tensor(control_transcriptomics),
            'treatment_transcriptomics': torch.tensor(treatment_transcriptomics),
            'control_images': torch.tensor(control_images),
            'treatment_images': torch.tensor(treatment_images),
            'conditioning_info': conditioning_info
        }


def create_vocab_mappings(dataset):
    """Create vocabulary mappings for categorical variables."""
    compounds = list(set(dataset.metadata_control['compound'].unique()).union(set(dataset.metadata_drug['compound'].unique())))
    cell_lines = list(set(dataset.metadata_control['cell_line'].unique()).union(set(dataset.metadata_drug['cell_line'].unique())))
    
    compound_to_idx = {comp: idx for idx, comp in enumerate(sorted(compounds))}
    cell_line_to_idx = {cl: idx for idx, cl in enumerate(sorted(cell_lines))}
    
    return compound_to_idx, cell_line_to_idx


def image_transform(images):
    """
    Transform for 16-bit multi-channel microscopy images.
    Args:
        images: numpy array of shape (channels, height, width)
    
    Returns:
        Normalized and contrast-enhanced images
    """
    # Normalize 16-bit to 0-1 range (CORRECTED from /255 to /65535)
    images_norm = (images / 32767.5) - 1.0
    # Apply per-channel contrast enhancement
    enhanced_images = np.zeros_like(images_norm)
    for i in range(images_norm.shape[0]):
        channel = images_norm[i]
        p1, p99 = np.percentile(channel, [1, 99])
        if p99 > p1:
            enhanced_images[i] = np.clip((channel - p1) / (p99 - p1) * 2 - 1, -1, 1)
        else:
            enhanced_images[i] = channel
    return enhanced_images.astype(np.float32)


def load_drug_data_hdf5(file_path: str) -> dict:
    """
    Load drug data from HDF5 with exact same structure as pickle version.
    Drop-in replacement for load_preprocessed_drug_data.
    """
    loaded_data = {}
    
    with h5py.File(file_path, 'r') as f:
        # Load metadata from attributes
        for key in f.attrs.keys():
            attr_value = f.attrs[key]
            if isinstance(attr_value, bytes):
                attr_value = attr_value.decode('utf-8')
            
            # Try to parse as JSON for complex structures
            if key in ['modality_dims', 'dataset_stats', 'preprocessing_params']:
                try:
                    loaded_data[key] = json.loads(attr_value)
                except (json.JSONDecodeError, TypeError):
                    loaded_data[key] = attr_value
            else:
                loaded_data[key] = attr_value
        
        # Load drug embeddings
        drug_embeddings = {}
        if 'drug_embeddings' in f:
            for compound_name in f['drug_embeddings'].keys():
                compound_group = f['drug_embeddings'][compound_name]
                embeddings = {}
                
                # Load datasets (numpy arrays) - check type first
                for dataset_name in compound_group.keys():
                    item = compound_group[dataset_name]
                    if isinstance(item, h5py.Dataset):
                        embeddings[dataset_name] = item[:]
                    # Skip groups - they should not be here for embeddings
                
                # Load attributes (scalars, strings)
                for attr_name in compound_group.attrs.keys():
                    attr_value = compound_group.attrs[attr_name]
                    if isinstance(attr_value, bytes):
                        attr_value = attr_value.decode('utf-8')
                    embeddings[attr_name] = attr_value
                
                drug_embeddings[compound_name] = embeddings
        
        loaded_data['drug_embeddings'] = drug_embeddings
        
        # Load compound metadata
        compound_metadata = {}
        if 'compound_metadata' in f:
            for compound_name in f['compound_metadata'].keys():
                metadata_group = f['compound_metadata'][compound_name]
                metadata = {}
                
                for attr_name in metadata_group.attrs.keys():
                    attr_value = metadata_group.attrs[attr_name]
                    if isinstance(attr_value, bytes):
                        attr_value = attr_value.decode('utf-8')
                    
                    # Try to convert back to appropriate type
                    if attr_name in ['cid']:
                        try:
                            metadata[attr_name] = int(attr_value)
                        except (ValueError, TypeError):
                            metadata[attr_name] = attr_value
                    elif attr_name in ['molecular_weight']:
                        try:
                            metadata[attr_name] = float(attr_value)
                        except (ValueError, TypeError):
                            metadata[attr_name] = attr_value
                    else:
                        metadata[attr_name] = attr_value
                
                compound_metadata[compound_name] = metadata
        
        loaded_data['compound_metadata'] = compound_metadata
    
    return loaded_data


class DatasetWithDrugs(HistologyTranscriptomicsDataset):
    """
    Dataset that includes drug conditioning information.
    """
    def __init__(self,
                metadata_control: pd.DataFrame,
                metadata_drug: pd.DataFrame,
                drug_data_path: str,
                gene_count_matrix: pd.DataFrame = None,
                image_json_dict: Dict[str, List[str]] = None,
                transform: Optional[Callable] = None,
                target_size: Optional[int] = None,
                drug_encoder: Optional[torch.nn.Module] = None,
                debug_mode: bool = False,
                debug_samples: int = 50,
                debug_cell_lines: Optional[List[str]] = None,
                debug_drugs: Optional[List[str]] = None,
                exclude_drugs: Optional[List[str]] = None,
                fallback_smiles_dict=None,
                enable_smiles_fallback=False,
                smiles_cache: Optional[Dict] = None,
                smiles_only: bool = False,
                cell_line_label: str = 'cell_line',
                compound_name_label: str = 'compound',
                ):
        """
        Args:
            smiles_only: If True, skip loading drug embeddings and only provide SMILES
        """
        # Store debug parameters
        self.debug_mode = debug_mode
        self.debug_samples = debug_samples
        self.debug_cell_lines = debug_cell_lines
        self.warned_compounds = set()
        self.smiles_only = smiles_only
        
        # Apply debug filtering to metadata BEFORE parent initialization
        if debug_mode:
            original_drug_size = len(metadata_drug)
            original_control_size = len(metadata_control)
            
            # Filter by specific cell lines if provided
            if debug_cell_lines:
                metadata_drug = metadata_drug[metadata_drug[cell_line_label].isin(debug_cell_lines)]
                metadata_control = metadata_control[metadata_control[cell_line_label].isin(debug_cell_lines)]
                print(f"DEBUG MODE: Filtered to cell lines: {debug_cell_lines}")
            
            if debug_drugs:
                metadata_drug = metadata_drug[metadata_drug[compound_name_label].isin(debug_drugs)]
                print(f"DEBUG MODE: Filtered to drugs: {debug_drugs}")
            
            if exclude_drugs:
                metadata_drug = metadata_drug[~metadata_drug[compound_name_label].isin(exclude_drugs)]
                print(f"DEBUG MODE: Excluded drugs: {exclude_drugs}")
            
            # Take only first N samples for debugging
            if debug_samples is not None and debug_samples > 0 and len(metadata_drug) > debug_samples:
                metadata_drug = metadata_drug.head(debug_samples).reset_index(drop=True)
            else:
                logger.warning("DEBUG MODE FALLBACK: debug_samples is None or larger than dataset size; not limiting samples.")
            
            # Ensure indices are reset after filtering
            metadata_drug = metadata_drug.reset_index(drop=True)

            print(f"DEBUG MODE: Reduced dataset size:")
            print(f"Drug metadata: {original_drug_size} → {len(metadata_drug)} samples")
            print(f"Control metadata: {original_control_size} → {len(metadata_control)} samples")
        
        # Initialize parent class with potentially filtered data
        super().__init__(
            metadata_control=metadata_control,
            metadata_drug=metadata_drug,
            gene_count_matrix=gene_count_matrix,
            image_json_dict=image_json_dict,
            transform=transform,
            target_size=target_size
        )
        
        # Skip drug embedding loading if only SMILES needed
        if not smiles_only:
            # Load preprocessed drug data
            self.drug_data = self._load_drug_data(drug_data_path)
            self.drug_encoder = drug_encoder

            if self.drug_data is not None:
                self.compound_lookup = self.drug_data['drug_embeddings']
            else:
                self.compound_lookup = {}
                logger.warning("Drug data is None; compound lookup will be empty.")
            
            # Create compound mapping for quick lookup
            self._create_compound_mapping()
            
            if self.drug_data is not None:
                logger.info(f"Loaded drug data for {len(self.drug_data['drug_embeddings'])} compounds")
        else:
            # Skip all drug embedding logic for SMILES-only mode
            self.drug_data = None
            self.compound_lookup = {}
            self.compound_to_embeddings = {}
            self.available_compounds = set()
            logger.info("SMILES-only mode: Skipped drug embedding loading")

        if debug_mode:
            logger.info(f"DEBUG MODE: Final dataset size: {len(self)} samples")

        self.fallback_smiles_dict = fallback_smiles_dict or {}
        self.enable_smiles_fallback = enable_smiles_fallback

        if enable_smiles_fallback and not smiles_only:
            self._init_smiles_processor()

        self.smiles_cache = smiles_cache if smiles_cache is not None else {}
        self.warned_compounds = set()

    def _init_smiles_processor(self):
        """Initialize components for on-demand SMILES processing."""
        try:
            self.rdkit_available = True
            
            # Store processing parameters from drug_data for consistency
            self.fingerprint_size = self.drug_data.get('preprocessing_params', {}).get('fingerprint_size', 1024)
            self.normalize_descriptors = self.drug_data.get('preprocessing_params', {}).get('normalize_descriptors', True)
            
            # Get normalization parameters if they exist
            if 'modality_dims' in self.drug_data:
                sample_drug = next(iter(self.drug_data['drug_embeddings'].values()))
                if 'descriptors_2d' in sample_drug and hasattr(self, 'normalization_params'):
                    self.desc_mean = self.normalization_params.get('mean')
                    self.desc_std = self.normalization_params.get('std')
        except ImportError:
            logger.warning("RDKit not available - SMILES fallback disabled")
            self.rdkit_available = False

    def _compute_smiles_embeddings(self, smiles: str) -> Dict[str, torch.Tensor]:
        """Convert SMILES to drug embeddings on-demand."""
        if not self.rdkit_available:
            logger.warning(f"Cannot process SMILES {smiles} - RDKit not available")
            return self._get_zero_embeddings()
        
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, rdMolDescriptors, AllChem, RDKFingerprint
            
            # Convert SMILES to RDKit molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                logger.warning(f"Invalid SMILES: {smiles}")
                return self._get_zero_embeddings()
            
            # Compute Morgan fingerprint
            fp_morgan = AllChem.GetMorganFingerprintAsBitVect(
                mol, radius=2, nBits=self.fingerprint_size
            )
            fp_morgan = np.array(fp_morgan, dtype=np.float32)
            
            # Compute RDKit fingerprint
            fp_rdkit = RDKFingerprint(mol, fpSize=self.fingerprint_size)
            fp_rdkit = np.array(fp_rdkit, dtype=np.float32)
            
            # Compute 2D descriptors (same as drug_process.py)
            descriptors = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                Descriptors.NumRotatableBonds(mol),
                Descriptors.NumAromaticRings(mol),
                rdMolDescriptors.CalcFractionCSP3(mol),
                Descriptors.BalabanJ(mol),
                Descriptors.BertzCT(mol),
                rdMolDescriptors.CalcNumRings(mol),
                rdMolDescriptors.CalcNumSaturatedRings(mol),
                rdMolDescriptors.CalcNumAliphaticRings(mol),
                rdMolDescriptors.CalcNumAromaticRings(mol),
                rdMolDescriptors.CalcNumHeterocycles(mol),
                rdMolDescriptors.CalcFractionCSP3(mol),
                rdMolDescriptors.CalcNumRotatableBonds(mol),
                rdMolDescriptors.CalcExactMolWt(mol)
            ]
            descriptors_2d = np.array(descriptors, dtype=np.float32)
            
            # Apply normalization if available
            if self.normalize_descriptors and hasattr(self, 'desc_mean') and self.desc_mean is not None:
                descriptors_2d = (descriptors_2d - self.desc_mean) / (self.desc_std + 1e-8)
            
            # Convert to tensors
            result = {
                'fingerprint_morgan': torch.from_numpy(fp_morgan).float(),
                'fingerprint_rdkit': torch.from_numpy(fp_rdkit).float(),
                'descriptors_2d': torch.from_numpy(descriptors_2d).float(),
                'descriptors_3d': torch.zeros(5, dtype=torch.float32),  # No 3D info from SMILES
                'has_3d_structure': False,
                'has_2d_structure': False,  # Computed from SMILES, not actual 2D file
                'structure_source': 'SMILES_ON_DEMAND',
                'smiles': smiles
            }
            
            logger.info(f"Generated embeddings for new drug from SMILES: {smiles[:2]}...")
            return result
            
        except Exception as e:
            logger.error(f"Error processing SMILES {smiles}: {e}")
            return self._get_zero_embeddings()

    def _load_drug_data(self, drug_data_path: str) -> Dict:
        """Load preprocessed drug data."""
        try:
            if drug_data_path.endswith('.pickle'):
                with open(drug_data_path, 'rb') as f:
                    drug_data = pickle.load(f)
            elif drug_data_path.endswith('.h5') or drug_data_path.endswith('.hdf5'):
                drug_data = load_drug_data_hdf5(drug_data_path)
            else:
                raise ValueError("Unsupported drug data file format. Use .pickle or .h5/.hdf5")
            logger.info(f"Loaded preprocessed drug data from {drug_data_path}")
            return drug_data
        except Exception as e:
            logger.error(f"Failed to load drug data from {drug_data_path}: {e}")
            # raise
            return None
    
    def _create_compound_mapping(self):
        """Create mapping from compound names to drug embeddings."""
        if self.drug_data is None or 'drug_embeddings' not in self.drug_data:
            logger.warning("Drug data is None or missing 'drug_embeddings'; compound mapping will be empty.")
            self.compound_to_embeddings = {}
            self.available_compounds = set()
        else:
            self.compound_to_embeddings = self.drug_data['drug_embeddings']
            self.available_compounds = set(self.compound_to_embeddings.keys())
        
        # Check coverage
        required_compounds = set(self.metadata_drug['compound'].unique())
        missing_compounds = required_compounds - self.available_compounds
        
        if missing_compounds:
            logger.warning(f"Missing drug data for compounds: {missing_compounds}")
    
    def get_drug_embeddings(self, compound_name: str) -> Dict[str, torch.Tensor]:
        """Enhanced lookup with SMILES fallback."""
        # Try preprocessed lookup first
        if compound_name in self.smiles_cache:
            return self.smiles_cache[compound_name]

        if compound_name in self.compound_lookup:
            embeddings = self.compound_lookup[compound_name]
            result = {}
            for key, value in embeddings.items():
                if isinstance(value, np.ndarray):
                    tensor = torch.from_numpy(value).float()
                    result[key] = tensor.contiguous().clone()
                else:
                    result[key] = value
            self.smiles_cache[compound_name] = result
            return result
        # Try SMILES fallback if enabled
        elif self.enable_smiles_fallback:
            # Check if we have SMILES for this compound
            smiles = self.fallback_smiles_dict.get(compound_name)
            if smiles:
                if compound_name not in self.warned_compounds:
                    logger.info(f"Using SMILES fallback for unknown compound: {compound_name}")
                    self.warned_compounds.add(compound_name)
                
                computed_embeddings = self._compute_smiles_embeddings(smiles)
                # Store the new embeddings in the shared cache
                self.smiles_cache[compound_name] = computed_embeddings
                return computed_embeddings
            else:
                # Only warn once per compound
                if compound_name not in self.warned_compounds:
                    logger.warning(f"No SMILES available for unknown compound: {compound_name}")
                    self.warned_compounds.add(compound_name)
                zero_embeddings = self._get_zero_embeddings()
                # Also cache the "not found" result
                self.smiles_cache[compound_name] = zero_embeddings
                return zero_embeddings
        else:
            zero_embeddings = self._get_zero_embeddings()
            self.smiles_cache[compound_name] = zero_embeddings
            return zero_embeddings

    def _get_zero_embeddings(self) -> Dict[str, torch.Tensor]:
        """Get zero embeddings for missing compounds."""
        # Handle case where drug_data is None
        if self.drug_data is None:
            modality_dims = {
                'fingerprint_morgan': 1024,
                'fingerprint_rdkit': 1024,
                'descriptors_2d': 18
            }
        else:
            modality_dims = self.drug_data.get('modality_dims', {
                'fingerprint_morgan': 1024,
                'fingerprint_rdkit': 1024,
                'descriptors_2d': 18
            })
        
        zero_embeddings = {
            'fingerprint_morgan': torch.zeros(
                modality_dims.get('fingerprint_morgan', 1024), 
                dtype=torch.float32
            ).contiguous(),
            'fingerprint_rdkit': torch.zeros(
                modality_dims.get('fingerprint_rdkit', 1024), 
                dtype=torch.float32
            ).contiguous(),
            'descriptors_2d': torch.zeros(
                modality_dims.get('descriptors_2d', 18), 
                dtype=torch.float32
            ).contiguous(),
            'descriptors_3d': torch.zeros(5, dtype=torch.float32).contiguous(),
            'has_3d_structure': False,
            'has_2d_structure': False,
            'structure_source': 'NONE',
            'smiles': ''
        }
        
        return zero_embeddings
    
    def encode_drug_condition(self, compound_name: str) -> torch.Tensor:
        """
        Encode drug into condition embedding.
        
        Args:
            compound_name: Name of the compound
            
        Returns:
            Drug condition tensor
        """
        drug_embeddings = self.get_drug_embeddings(compound_name)
        
        if self.drug_encoder is not None:
            # Use trained drug encoder
            # Create a mini-batch with single item
            batch_dict = {key: value.unsqueeze(0) for key, value in drug_embeddings.items() 
                         if isinstance(value, torch.Tensor)}
            batch_dict.update({key: [value] for key, value in drug_embeddings.items() 
                              if not isinstance(value, torch.Tensor)})
            
            with torch.no_grad():
                drug_condition = self.drug_encoder(batch_dict).squeeze(0)
            return drug_condition
        else:
            # Use raw embeddings (concatenate main modalities)
            main_embeddings = [
                drug_embeddings['fingerprint_morgan'],
                drug_embeddings['descriptors_2d']
            ]
            # Add 3D descriptors if available
            if drug_embeddings['descriptors_3d'].numel() > 0:
                main_embeddings.append(drug_embeddings['descriptors_3d'])
                
            return torch.cat(main_embeddings, dim=0)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Enhanced getitem that includes drug conditioning or just SMILES.
        
        Returns:
            Dictionary with original data plus drug conditioning or just SMILES
        """
        # Get original data
        sample = super().__getitem__(idx)
        
        # Get drug information
        treatment_sample = self.filtered_drug_metadata.iloc[idx]
        compound_name = treatment_sample['compound']
        
        # Add compound name
        sample['compound_name'] = compound_name
        
        if self.smiles_only:
            # Only add SMILES, skip all drug embedding logic
            smiles = self.fallback_smiles_dict.get(compound_name, '')
            sample.update({
                'target_smiles': smiles,
                'drug_condition': torch.zeros(1047, dtype=torch.float32)  # Empty drug condition
            })
        else:
            # Add drug embeddings and condition (original behavior)
            drug_embeddings = self.get_drug_embeddings(compound_name)
            drug_condition = self.encode_drug_condition(compound_name)
            
            sample.update({
                'drug_embeddings': drug_embeddings,
                'drug_condition': drug_condition,
            })
        
        return sample


def create_dataloader(
    metadata_control: pd.DataFrame,
    metadata_drug: pd.DataFrame,
    gene_count_matrix: pd.DataFrame,
    image_json_path: str,
    drug_data_path: str,

    batch_size: int = 2,
    shuffle: bool = True,
    num_workers: int = 0,
    transform: Optional[Callable] = None,
    target_size: Optional[int] = 256,

    use_highly_variable_genes: bool = True,
    n_top_genes: int = 2000,
    normalize: bool = True,
    zscore: bool = True,

    drug_encoder: Optional[torch.nn.Module] = None,
    debug_mode: bool = False,
    debug_samples: int = None,
    debug_cell_lines: Optional[List[str]] = None,
    debug_drugs: Optional[List[str]] = None,
    exclude_drugs: Optional[List[str]] = None,

    fallback_smiles_dict: Optional[Dict[str, str]] = None,
    enable_smiles_fallback: bool = False,
    **kwargs
) -> DataLoader:
    """
    Create DataLoader with drug conditioning and debug options.
    
    Args:
        ... (same as before)
        debug_mode: If True, only load a subset of data for debugging
        debug_samples: Number of samples to load in debug mode
        debug_cell_lines: Specific cell lines to use for debugging
    """
    # Load image paths from JSON file
    with open(image_json_path, 'r') as f:
        image_json_dict = json.load(f)
    
    # Create dataset with debug options
    dataset = DatasetWithDrugs(
        metadata_control=metadata_control,
        metadata_drug=metadata_drug,
        gene_count_matrix=gene_count_matrix,
        image_json_dict=image_json_dict,
        drug_data_path=drug_data_path,

        fallback_smiles_dict=fallback_smiles_dict,
        enable_smiles_fallback=enable_smiles_fallback,

        transform=transform,
        target_size=target_size,
        
        drug_encoder=drug_encoder,
        debug_mode=debug_mode,
        debug_samples=debug_samples,
        debug_cell_lines=debug_cell_lines,
        debug_drugs=debug_drugs,
        exclude_drugs=exclude_drugs
    )
    
    if use_highly_variable_genes:
        adata = ad.AnnData(X=dataset.gene_count_matrix.T.values, 
                           obs=dataset.gene_count_matrix.T.index.to_frame(),
                           var=dataset.gene_count_matrix.T.columns.to_frame())
        adata.layers['counts'] = adata.X.copy()
        sc.pp.normalize_total(adata, target_sum=1e6)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes,)
        hvg_genes = adata.var_names[adata.var['highly_variable']]
        if zscore:
            sc.pp.scale(adata)
        if not normalize:
            adata.X = adata.layers['counts']
        dataset.gene_count_matrix = adata[:, hvg_genes].to_df().T

    # Enhanced collate function (same as before)
    def enhanced_collate_fn(batch):
        """Custom collate function for batch with drug data."""
        collated = {
            'control_transcriptomics': torch.stack([item['control_transcriptomics'] for item in batch]),
            'treatment_transcriptomics': torch.stack([item['treatment_transcriptomics'] for item in batch]),
            'control_images': torch.stack([item['control_images'] for item in batch]),
            'treatment_images': torch.stack([item['treatment_images'] for item in batch]),
            'compound_name': [item['compound_name'] for item in batch],
            'conditioning_info': [item['conditioning_info'] for item in batch]
        }

        # Handle drug_condition with proper error checking
        drug_conditions = []
        for item in batch:
            drug_cond = item['drug_condition']
            if drug_cond.numel() == 0:  # Empty tensor
                # Create a zero tensor with consistent shape
                drug_cond = torch.zeros(1047, dtype=torch.float32)  # Adjust size as needed
            drug_conditions.append(drug_cond)
        
        # Check if all drug conditions have the same shape
        if len(set(dc.shape for dc in drug_conditions)) == 1:
            collated['drug_condition'] = torch.stack(drug_conditions)
        else:
            # Pad to maximum length
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

        # Handle drug embeddings safely
        if 'drug_embeddings' in batch[0]:
            drug_keys = batch[0]['drug_embeddings'].keys()
            collated['drug_embeddings'] = {}
            for key in drug_keys:
                if isinstance(batch[0]['drug_embeddings'][key], torch.Tensor):
                    embeddings = [item['drug_embeddings'][key] for item in batch]
                    # Check if all have same shape
                    if len(set(emb.shape for emb in embeddings)) == 1:
                        collated['drug_embeddings'][key] = torch.stack(embeddings)
                    else:
                        # Handle variable shapes - pad or truncate as needed
                        collated['drug_embeddings'][key] = embeddings  # Keep as list
                else:
                    collated['drug_embeddings'][key] = [
                        item['drug_embeddings'][key] for item in batch
                    ]

        return collated
    
    # Adjust batch size and num_workers for debug mode
    if debug_mode:
        batch_size = min(batch_size, 4)
        num_workers = 0
        shuffle = False
        print(f"DEBUG MODE: Adjusted batch_size={batch_size}, num_workers={num_workers}, shuffle={shuffle}")
    
    # Create DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=enhanced_collate_fn,
        pin_memory=torch.cuda.is_available()
    )
    
    return dataloader, create_vocab_mappings(dataset)