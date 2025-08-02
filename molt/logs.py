import wandb
import torch
from functools import partial
import os
import json
import multiprocessing as mp
from queue import Empty
import time


def init_wandb(cfg):
    # Remove W_U and b_U from config before submitting to wandb
    wandb_config = {k: v for k, v in cfg.items() if k not in ["W_U", "b_U"]}
    return wandb.init(
        project=cfg["wandb_project"], name=cfg["name"], config=wandb_config, reinit=True
    )


def new_wandb_process(config, log_queue, entity, project):
    run = wandb.init(
        entity=entity, project=project, config=config, name=config["name"]
    )
    time.sleep(3)
    while True:
        try:
            # Wait up to 1 second for new data
            log = log_queue.get(timeout=1)

            # Check for termination signal
            if log == "DONE":
                break

            # Check if this is a checkpoint signal
            if isinstance(log, dict) and log.get("checkpoint"):
                # Create and log artifact
                artifact = wandb.Artifact(
                    name=f"{config['name']}_{log['step']}",
                    type="model",
                    description=f"Model checkpoint at step {log['step']}",
                )
                save_dir = log["save_dir"]
                artifact.add_file(os.path.join(save_dir, "sae.pt"))
                artifact.add_file(os.path.join(save_dir, "config.json"))
                run.log_artifact(artifact)
            else:
                # Log regular metrics
                wandb.log(log)
        except Empty:
            continue

    wandb.finish()


def save_checkpoint_mp(sae, cfg, step):
    """
    Save checkpoint without requiring a wandb run object.
    Creates an artifact but doesn't log it to wandb directly.
    """
    save_dir = f"checkpoints/{cfg['name']}_{step}"
    os.makedirs(save_dir, exist_ok=True)

    # Save model state
    sae_path = os.path.join(save_dir, "sae.pt")
    torch.save(sae.state_dict(), sae_path)

    # Prepare config for JSON serialization
    json_safe_cfg = {}
    for key, value in cfg.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            json_safe_cfg[key] = value
        elif isinstance(value, (list, tuple, dict)):
            # Preserve lists, tuples, and dicts as-is for MOLT config
            json_safe_cfg[key] = value
        elif isinstance(value, (torch.dtype, type)):
            json_safe_cfg[key] = str(value)
        else:
            json_safe_cfg[key] = str(value)

    # Save config
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(json_safe_cfg, f, indent=4)

    print(f"Model and config saved at step {step} in {save_dir}")
    return save_dir, sae_path, config_path


def log_wandb(output, step, wandb_run, index=None):
    log_dict = {
        k: v.item() if isinstance(v, torch.Tensor) and v.dim() == 0 else v
        for k, v in output.items()
        if isinstance(v, (int, float))
        or (isinstance(v, torch.Tensor) and v.dim() == 0)
    }

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)


def save_checkpoint(wandb_run, sae, cfg, step):
    save_dir = f"checkpoints/{cfg['name']}_{step}"
    os.makedirs(save_dir, exist_ok=True)

    # Save model state
    sae_path = os.path.join(save_dir, "sae.pt")
    torch.save(sae.state_dict(), sae_path)

    # Prepare config for JSON serialization
    json_safe_cfg = {}
    for key, value in cfg.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            json_safe_cfg[key] = value
        elif isinstance(value, (list, tuple, dict)):
            # Preserve lists, tuples, and dicts as-is for MOLT config
            json_safe_cfg[key] = value
        elif isinstance(value, (torch.dtype, type)):
            json_safe_cfg[key] = str(value)
        else:
            json_safe_cfg[key] = str(value)

    # Save config
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(json_safe_cfg, f, indent=4)

    # Create and log artifact
    artifact = wandb.Artifact(
        name=f"{cfg['name']}_{step}",
        type="model",
        description=f"Model checkpoint at step {step}",
    )
    artifact.add_file(sae_path)
    artifact.add_file(config_path)
    wandb_run.log_artifact(artifact)

    print(f"Model and config saved as artifact at step {step}")
