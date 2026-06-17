# enso

Fork of [DiT-MoE](https://github.com/feizc/DiT-MoE) (arXiv:2407.11633) — sparse Mixture-of-Experts Diffusion Transformer scaled to 16B params, with rectified-flow training/sampling. Zen LM's diffusion-MoE research base (Hanzo AI).

Entry points: `models.py` (model defs), `train.py` / `train_deepspeed.py` (DDP / DeepSpeed zero2/zero3), `sample.py` / `sample_ddp.py`, `upcycle.py` (dense→MoE), `analysis/` (expert-routing tools).

- Train: `torchrun --nproc_per_node=N train.py --model DiT-S/2 --num_experts 8 --num_experts_per_tok 2 ...`
- Sample: `python sample.py --model DiT-XL/2 --ckpt <path> --vae-path <path>`

Full docs: README.md
