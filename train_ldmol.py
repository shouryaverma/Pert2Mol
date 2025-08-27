# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os
from download import find_model
import pandas as pd

from models import DiT_models
from diffusion import create_diffusion

from transformers import T5ForConditionalGeneration, T5Tokenizer
from train_autoencoder import ldmol_autoencoder
from utils import AE_SMILES_encoder, regexTokenizer, dual_image_encoder # molT5_encoder,
# from dataset import smi_txt_dataset
import random

from encoders import ImageEncoder
from dataset_gdp import create_raw_drug_dataloader
from diffusion.rectified_flow import create_rectified_flow

#################################################################################
#                             Training Helper Functions                         #
#################################################################################


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger



#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new DiT model with flexible single/multi-GPU support.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    
    # Check CUDA compatibility
    try:
        device_capability = torch.cuda.get_device_capability()
        print(f"GPU capability: {device_capability}")
        test_tensor = torch.randn(2, 2, device='cuda')
        test_result = test_tensor * 2.0
        print("CUDA compatibility test passed")
    except RuntimeError as e:
        print(f"CUDA compatibility error: {e}")
        print("Solution: Reinstall PyTorch with proper CUDA version for your GPU")
        raise e

    # Setup distributed or single GPU training based on flag
    if args.use_distributed and torch.cuda.device_count() > 1:
        # Multi-GPU distributed training
        dist.init_process_group("nccl")
        assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
        rank = dist.get_rank()
        device = rank % torch.cuda.device_count()
        seed = args.global_seed * dist.get_world_size() + rank
        torch.manual_seed(seed)
        torch.cuda.set_device(device)
        print(f"Starting distributed training: rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
        
        # Setup experiment folder (only on rank 0)
        if rank == 0:
            os.makedirs(args.results_dir, exist_ok=True)
            experiment_index = len(glob(f"{args.results_dir}/*"))
            model_string_name = args.model.replace("/", "-")
            experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
            checkpoint_dir = f"{experiment_dir}/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            logger = create_logger(experiment_dir)
            logger.info(f"Experiment directory created at {experiment_dir}")
        else:
            logger = create_logger(None)
            
        use_ddp = True
        batch_size = int(args.global_batch_size // dist.get_world_size())
        
    else:
        # Single GPU training
        device = torch.device("cuda:0")
        rank = 0
        seed = args.global_seed
        torch.manual_seed(seed)
        torch.cuda.set_device(0)
        print(f"Starting single GPU training on device={device}, seed={seed}.")
        
        # Setup experiment folder
        os.makedirs(args.results_dir, exist_ok=True)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create simple logger for single GPU
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{experiment_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
        logger.info(f"Experiment directory created at {experiment_dir}")
        
        use_ddp = False
        batch_size = args.global_batch_size

    # Create model:
    latent_size = 127
    in_channels = 64
    cross_attn = 256
    conditioning_dim = 256
    
    model = DiT_models[args.model](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=args.num_classes,
        cross_attn=cross_attn,
        condition_dim=conditioning_dim
    )

    if args.ckpt:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        msg = model.load_state_dict(state_dict, strict=True)
        print('load DiT from ', ckpt_path, msg)

    # Setup model for distributed or single GPU
    ema = deepcopy(model).to(device)
    requires_grad(ema, False)
    
    if use_ddp:
        model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=True)
    else:
        model = model.to(device)
        
    # diffusion = create_diffusion(timestep_respacing="")
    flow = create_rectified_flow(num_timesteps=1000)

    # Load autoencoder
    ae_config = {
        'bert_config_decoder': './config_decoder.json',
        'bert_config_encoder': './config_encoder.json',
        'embed_dim': 256,
    }
    tokenizer = regexTokenizer(vocab_path='./vocab_bpe_300_sc.txt', max_len=127)
    ae_model = ldmol_autoencoder(config=ae_config, no_train=True, tokenizer=tokenizer, use_linear=True)
    
    if args.vae:
        print('LOADING PRETRAINED MODEL..', args.vae)
        checkpoint = torch.load(args.vae, map_location='cpu')
        try:
            state_dict = checkpoint['model']
        except:
            state_dict = checkpoint['state_dict']
        msg = ae_model.load_state_dict(state_dict, strict=False)
        print('autoencoder', msg)
    
    for param in ae_model.parameters():
        param.requires_grad = False
    del ae_model.text_encoder
    ae_model = ae_model.to(device)
    ae_model.eval()
    print(f'AE #parameters: {sum(p.numel() for p in ae_model.parameters())}, #trainable: {sum(p.numel() for p in ae_model.parameters() if p.requires_grad)}')

    logger.info(f"DiT Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup image encoder
    image_encoder = ImageEncoder(img_channels=4, output_dim=256).to(device)
    if use_ddp:
        image_encoder = DDP(image_encoder, device_ids=[rank])
    
    for param in image_encoder.parameters():
        param.requires_grad = True
    image_encoder.train()
    print(f'ImageEncoder #parameters: {sum(p.numel() for p in image_encoder.parameters())}, #trainable: {sum(p.numel() for p in image_encoder.parameters() if p.requires_grad)}')
    
    # Create optimizer
    if use_ddp:
        all_params = list(model.parameters()) + list(image_encoder.parameters())
    else:
        all_params = list(model.parameters()) + list(image_encoder.parameters())
    opt = torch.optim.AdamW(all_params, lr=1e-4, weight_decay=0)

    # Setup data
    loader = create_raw_drug_dataloader(
        metadata_control=metadata_control,
        metadata_drug=metadata_drug, 
        gene_count_matrix=gene_count_matrix,
        image_json_path=args.image_json_path,
        drug_data_path=args.drug_data_path,
        raw_drug_csv_path=args.raw_drug_csv_path,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        compound_name_label='compound',
        debug_mode=False,
    )

    # Prepare models for training
    if use_ddp:
        update_ema(ema, model.module, decay=0)
    else:
        update_ema(ema, model, decay=0)
    model.train()
    ema.eval()

    # Training loop
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        logger.info(f"Beginning epoch {epoch}...")
        for batch in loader:
            with torch.no_grad():
                target_smiles = batch['target_smiles']
                x = AE_SMILES_encoder(target_smiles, ae_model).permute((0, 2, 1)).unsqueeze(-1)
                
            control_imgs = batch['control_images'].to(device)
            treatment_imgs = batch['treatment_images'].to(device)
            control_features = image_encoder(control_imgs)
            treatment_features = image_encoder(treatment_imgs)
            y = torch.stack([control_features, treatment_features], dim=1)
            pad_mask = torch.ones(y.size(0), 2, dtype=torch.bool, device=device)
                
            # t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            # model_kwargs = dict(y=y.type(torch.float32), pad_mask=pad_mask.bool())
            # loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            # loss = loss_dict["loss"].mean()
            model_kwargs = dict(y=y.type(torch.float32), pad_mask=pad_mask.bool())
            loss_dict = flow.training_losses(model, x, model_kwargs=model_kwargs)
            loss = loss_dict["loss"].mean()
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if use_ddp:
                update_ema(ema, model.module)
            else:
                update_ema(ema, model)

            # Logging
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                
                if use_ddp:
                    # Reduce loss across all processes
                    avg_loss = torch.tensor(running_loss / log_steps, device=device)
                    dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                    avg_loss = avg_loss.item() / dist.get_world_size()
                else:
                    avg_loss = running_loss / log_steps
                
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save checkpoint
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if not use_ddp or rank == 0:
                    if use_ddp:
                        checkpoint = {
                            "model": model.module.state_dict(),
                            "image_encoder": image_encoder.module.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "args": args
                        }
                    else:
                        checkpoint = {
                            "model": model.state_dict(),
                            "image_encoder": image_encoder.state_dict(),
                            "ema": ema.state_dict(),
                            "opt": opt.state_dict(),
                            "args": args
                        }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                
                if use_ddp:
                    dist.barrier()

    model.eval()
    logger.info("Done!")
    
    if use_ddp:
        cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--text-encoder-name", type=str, default="molt5")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="LDMol")
    parser.add_argument("--description-length", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=16*6)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, default="/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/src_new/LDMol/checkpoint_autoencoder.ckpt")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--use-distributed", action="store_true", help="Enable distributed training across multiple GPUs")
    
    parser.add_argument("--image-json-path", type=str, default="/depot/natallah/data/Mengbo/HnE_RNA/PertRF/data/processed_data/image_paths.json")
    parser.add_argument("--drug-data-path", type=str, default="/depot/natallah/data/Mengbo/HnE_RNA/drug/PubChem/GDP_compatible/preprocessed_drugs.synonymous.pkl")
    parser.add_argument("--raw-drug-csv-path", type=str, default="/depot/natallah/data/Mengbo/HnE_RNA/DrugGFN/PertRF/drug/PubChem/GDP_compatible/complete_drug_data.csv")
    parser.add_argument("--metadata-control-path", type=str, default="/depot/natallah/data/Mengbo/HnE_RNA/PertRF/data/processed_data/metadata_control.csv")
    parser.add_argument("--metadata-drug-path", type=str, default="/depot/natallah/data/Mengbo/HnE_RNA/PertRF/data/processed_data/metadata_drug.csv")
    parser.add_argument("--gene-count-matrix-path", type=str, default="/depot/natallah/data/Mengbo/HnE_RNA/PertRF/data/processed_data/GDPx1x2_gene_counts.parquet")
    args = parser.parse_args()

    print(args)
    
    metadata_control = pd.read_csv(args.metadata_control_path)
    metadata_drug = pd.read_csv(args.metadata_drug_path)
    gene_count_matrix = pd.read_parquet(args.gene_count_matrix_path)

    main(args)
