# %%
# !pip install transformer_lens
# b996b5b2faffea971ac27f3de099ffb0a1c98ee9
# %%
import importlib
import training
import activation_store
import config
import transformer_lens
import copy
import torch
from training import train_group_seperate_wandb
from molt import MOLT, Transcoder
from activation_store import ActivationsStore
from config import get_default_cfg, post_init_cfg
from transformer_lens import HookedTransformer
from huggingface_hub import login


def get_d_hidden_transcoder_equivalent(
    rank_groups,
    d_model: int = 768
    ):
    # -- MOLT parameter budget -------------------------------------------
    n_transforms = sum(n_t for n_t, _ in rank_groups)
    encoder_params   = d_model * n_transforms + n_transforms
    transform_params = 2 * sum(n_t * d_model * r_t for n_t, r_t in rank_groups)
    params_molt = encoder_params + transform_params

    # -- Solve for d_hidden ----------------------------------------------
    d_hidden_float = (params_molt - d_model) / (2 * d_model + 1)
    d_hidden = int(round(d_hidden_float))
    return d_hidden


decoders = []
cfgs = []

cfg = get_default_cfg()
cfg["model_name"] = "gpt2-small"
cfg["model_dtype"] = torch.bfloat16
cfg["device"] = "cuda"
model = (
    HookedTransformer.from_pretrained_no_processing(cfg["model_name"])
    .to(cfg["model_dtype"])
    .to(cfg["device"])
)

for l1_coeff in [0.05, 0.02, 0.01, 0]:
    cfg = get_default_cfg()
    cfg["model_name"] = "gpt2-small"
    cfg["dataset_path"] = "Skylion007/openwebtext"
    cfg["lr"] = 3e-4
    cfg["input_unit_norm"] = False
    cfg["wandb_project"] = "MOLT-sweep3"
    cfg["act_size"] = 768
    cfg["device"] = "cuda"
    cfg["num_tokens"] = 5e8
    cfg["batch_size"] = 4096
    cfg["model_batch_size"] = 256
    cfg["model_dtype"] = torch.bfloat16
    cfg["checkpoint_freq"] = 10000
    cfg["l1_coeff"] = l1_coeff
    cfg["decoder_type"] = "MOLT"
    cfg["checkpoint_freq"] = 1000

    # Cross-layer transcoder specific configuration
    cfg["hook_points"] = [f"blocks.8.ln2.hook_normalized"]
    cfg["target_hook_point"] = "blocks.8.hook_resid_post"


    N = 20
    cfg["rank_groups"] = [
        (N, 512)
        # (16*N, 32),
    ]

    # Create and train the cross-layer transcoder
    cfg = post_init_cfg(cfg)

    # Create activation store for cross-layer transcoder
    activations_store = ActivationsStore(model, cfg)
    molt = MOLT(cfg)

    decoders.append(molt)
    cfgs.append(cfg)

    print("Transcoder equivalent:")
    print(get_d_hidden_transcoder_equivalent(cfg["rank_groups"]))

    # cfg = get_default_cfg()
    # cfg["model_name"] = "gpt2-small"
    # cfg["dataset_path"] = "Skylion007/openwebtext"
    # cfg["lr"] = 3e-4
    # cfg["input_unit_norm"] = False
    # cfg["wandb_project"] = "MOLT-sweep2"
    # cfg["act_size"] = 768
    # cfg["device"] = "cuda"
    # cfg["num_tokens"] = 5e8
    # cfg["batch_size"] = 4096
    # cfg["model_batch_size"] = 256
    # cfg["model_dtype"] = torch.bfloat16
    # cfg["checkpoint_freq"] = 10000
    # cfg["l1_coeff"] = l1_coeff
    # cfg["hook_points"] = [f"blocks.8.ln2.hook_normalized"]
    # cfg["target_hook_point"] = "blocks.8.hook_resid_post"
    # cfg["decoder_type"] = "SkipTranscoder"

    # N = 50
    # cfg["rank_groups"] = [
    #     (N, 128),
    #     (2*N, 64),
    #     (4*N, 32),
    #     (8*N, 16),
    #     (16*N, 8)
    # ]

    # cfg["hidden_size"] = get_d_hidden_transcoder_equivalent(cfg["rank_groups"], skip=True)
    # print("transcoder hidden size: ", cfg["hidden_size"])

    # # Create and train the cross-layer transcoder
    # cfg = post_init_cfg(cfg)


    # # Create activation store for cross-layer transcoder
    # activations_store = ActivationsStore(model, cfg)
    # transcoder = SkipTranscoder(cfg)

    # decoders.append(transcoder)
    # cfgs.append(cfg)

# Train the cross-layer transcoder
train_group_seperate_wandb(decoders, activations_store, model, cfgs)
# %%
