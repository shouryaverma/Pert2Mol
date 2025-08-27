# ⚛️ LDMol

Official GitHub repository for LDMol, a latent text-to-molecule diffusion model.
The details can be found in the paper
*[LDMol: Text-Conditioned Molecule Diffusion Model Leveraging Chemically Informative Latent Space](https://arxiv.org/abs/2405.17829)*.

LDMol not only can generate molecules according to the given text prompt, but it's also able to perform various downstream tasks including molecule-to-text retrieval and text-guided molecule editing.

**🎉 The paper is accepted(poster) in ICML 2025.**

![fig1](https://github.com/user-attachments/assets/dcfe5b56-ae1b-4f25-9181-66f081994f71)

![ldmol_fig3 (2)](https://github.com/user-attachments/assets/00c41ec0-cdd1-48fe-8a71-37310c14f38d)

***

## 📑 Abstract
The unavoidable discreteness of a molecule makes it difficult for a diffusion model to connect raw data with highly complex conditions like natural language. To address this, we present a novel latent diffusion model dubbed LDMol for text-conditioned molecule generation. LDMol comprises a molecule autoencoder that produces a learnable and structurally informative feature space, and a natural language-conditioned latent diffusion model. In particular, recognizing that multiple SMILES notations can represent the same molecule, we employ a contrastive learning strategy to extract feature space that is aware of the unique characteristics of the molecule structure. LDMol outperforms the existing baselines on the text-to-molecule generation benchmark, suggesting a potential for diffusion models can outperform autoregressive models in text data generation with a better choice of the latent domain. Furthermore, we show that LDMol can be applied to downstream tasks such as molecule-to-text retrieval and text-guided molecule editing, demonstrating its versatility as a diffusion model.

## 🛠️ Requirements
Run `conda env create -f requirements.yaml` and it will generate a conda environment named `ldmol`.

The model checkpoint and data are too heavy to be included in this repo and can be found in ***[here](https://drive.google.com/drive/folders/170znWA5u3nC7S1mzF7RPNP5faAn56Q45?usp=sharing).***

## 🎯 Inference
Check out the arguments in the script files to see more details.

__1. text-to-molecule generation__

   * zero-shot: The model gets a hand-written text prompt.
       ```
       CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 inference_demo.py --num-samples 100 --ckpt ./Pretrain/checkpoint_ldmol.pt --prompt="This molecule includes benzoyl group." --cfg-scale=2.5
       ```
   * benchmark dataset: The model performs text-to-molecule generation on ChEBI-20 test set. The evaluation metrics will be printed at the end.
       ```
       CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 inference_t2m.py --ckpt ./Pretrain/checkpoint_ldmol.pt --cfg-scale=2.5
       ```

__2. molecule-to-text retrieval__

The model performs molecule-to-text retrieval on the given dataset. `--level` controls the quality of the query text(paragraph/sentence). `--n-iter` is the number of function evaluations of our model.
```
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 inference_retrieval_m2t.py --ckpt ./Pretrain/checkpoint_ldmol.pt --dataset="./data/PCdes/test.txt" --level="paragraph" --n-iter=10
```

__3. text-guided molecule editing__

The model performs a DDS-style text-guided molecule editing. `--source-text` should describe the `--input-smiles`. `--target-text` is your desired molecule description.
```
CUDA_VISIBLE_DEVICES=0 torchrun --nnodes=1 --nproc_per_node=1 inference_dds.py --ckpt ./Pretrain/checkpoint_ldmol.pt --input-smiles="C[C@H](CCc1ccccc1)Nc1ccc(C#N)cc1F" --source-text="This molecule contains fluorine." --target-text="This molecule contains bromine."
```


## 💡 Acknowledgement
* The code for DiT diffusion model is based on & modified from the official code of [DiT](https://github.com/facebookresearch/DiT).
* The code for BERT with cross-attention layers `xbert.py` and schedulers is modified from the one in [ALBEF](https://github.com/salesforce/ALBEF).
# Pert2Mol
