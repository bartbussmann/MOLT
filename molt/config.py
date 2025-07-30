import transformer_lens.utils as utils
import torch


def get_default_cfg():
    default_cfg = {
        "seed": 42,
        "batch_size": 4096,
        "lr": 3e-4,
        "num_tokens": int(1e9),
        "l1_coeff": 0,
        "beta1": 0.9,
        "beta2": 0.99,
        "max_grad_norm": 100000,
        "seq_len": 128,
        "dtype": torch.float32,
        "model_dtype": torch.float32,
        "model_name": "gpt2-small",
        "site": "resid_pre",
        "layer": 8,
        "act_size": 768,
        "device": "cuda:0",
        "model_batch_size": 512,
        "num_batches_in_buffer": 10,
        "dataset_path": "Skylion007/openwebtext",
        "wandb_project": "sparse_autoencoders",
        "perf_log_freq": 1000,
        "checkpoint_freq": 10000,
        "decoder_type": "MOLT",
    }

    default_cfg = post_init_cfg(default_cfg)
    return default_cfg


def post_init_cfg(cfg):
    cfg["name"] = f"{cfg['model_name']}_{cfg['decoder_type']}_l1_{cfg['l1_coeff']}"
    return cfg
