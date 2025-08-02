#%%

import wandb
import os
import json
import torch
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("..")
from molt import MOLT
from activation_store import ActivationsStore
from transformer_lens import HookedTransformer

def load_MOLT_from_wandb(artifact_name):
    # Initialize wandb
    api = wandb.Api()

    # Download the artifact
    artifact = api.artifact(artifact_name)
    artifact_dir = artifact.download()

    # Load the configuration
    config_path = os.path.join(artifact_dir, "config.json")
    with open(config_path, "r") as f:
        cfg = json.load(f)

    # Convert string representations back to torch.dtype
    if "dtype" in cfg:
        cfg["dtype"] = getattr(torch, cfg["dtype"].split(".")[-1])
    if "model_dtype" in cfg:
        cfg["model_dtype"] = getattr(torch, cfg["model_dtype"].split(".")[-1])

    molt = MOLT(cfg)

    # Load the state dict
    state_dict_path = os.path.join(artifact_dir, "sae.pt")
    state_dict = torch.load(state_dict_path, map_location=cfg["device"])
    molt.load_state_dict(state_dict)

    return molt, cfg


def process_tokens(tokenizer, l):
    SPACE = " "
    NEWLINE = "↩"
    TAB = "→"

    def process_token(tokenizer, s):
        if isinstance(s, torch.Tensor):
            s = s.item()
        if isinstance(s, np.int64):
            s = s.item()
        if isinstance(s, int):
            s = tokenizer.decode([s])
        s = s.replace(" ", SPACE)
        s = s.replace("\n", NEWLINE+"\n")
        s = s.replace("\t", TAB)
        return s
    
    if isinstance(l, str):
        l = tokenizer.encode(l)
    elif isinstance(l, torch.Tensor) and len(l.shape) > 1:
        l = l.squeeze(0)
    return [process_token(tokenizer, s) for s in l]


def make_token_df(tokenizer, tokens, len_prefix=5, len_suffix=1):
    def list_flatten(nested_list):
        return [x for y in nested_list for x in y]

    str_tokens = [process_tokens(tokenizer, t) for t in tokens]
    unique_token = [[f"{s}/{i}" for i, s in enumerate(str_tok)] 
                    for str_tok in str_tokens]

    context = []
    for b in range(tokens.shape[0]):
        for p in range(tokens.shape[1]):
            prefix = "".join(str_tokens[b][max(0, p-len_prefix):p])
            if p == tokens.shape[1]-1:
                suffix = ""
            else:
                suffix = "".join(
                    str_tokens[b][p + 1 : min(tokens.shape[1], p + 1 + len_suffix)]
                )
            current = str_tokens[b][p]
            context.append(f"{prefix}|{current}|{suffix}")
    return pd.DataFrame(dict(
        str_tokens=list_flatten(str_tokens),
        unique_token=list_flatten(unique_token),
        context=context,
    ))


@torch.no_grad()
def show_max_activating(sae, activations_store, feature_idx=0, 
                        threshold=None, len_prefix=10, len_suffix=3, n_batches=1):
    all_tokens = []
    all_features = []
    
    for _ in range(n_batches):
        batch_tokens = activations_store.get_batch_tokens()
        token_df = make_token_df(activations_store.tokenizer, batch_tokens,
                                len_prefix=len_prefix, len_suffix=len_suffix)
        all_tokens.append(token_df)
        batch_activations = activations_store.get_activations(batch_tokens)
        batch_activations = batch_activations.reshape(
            batch_activations.shape[0], -1, batch_activations.shape[-1]
        )
        feature_activations = sae.encode(batch_activations)[:, feature_idx]
        all_features.append(feature_activations.flatten().cpu().numpy())

    token_df = pd.concat(all_tokens, ignore_index=True)
    token_df["feature"] = np.concatenate(all_features)
    
    if threshold is None:
        display(token_df.sort_values("feature", ascending=False).head(10)
                .style.background_gradient("coolwarm"))
    else:
        t = token_df["feature"].max() * threshold
        display(token_df[token_df["feature"] > t].sample(10)
                .sort_values("feature", ascending=False)
                .style.background_gradient("coolwarm"))

#%%
molt, cfg = load_MOLT_from_wandb(
    "mats-sprint/MOLT-sweep3/gpt2-small_MOLT_l1_0_2000:v0"
)

model = (
    HookedTransformer.from_pretrained_no_processing(cfg["model_name"])
    .to(cfg["model_dtype"])
    .to(cfg["device"])
)

activations_store = ActivationsStore(model, cfg)

#%%
show_max_activating(molt, activations_store, feature_idx=0, n_batches=10, len_prefix=20, len_suffix=3, threshold=0.5)
# %%
