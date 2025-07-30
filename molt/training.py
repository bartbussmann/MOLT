import torch
import tqdm
from logs import new_wandb_process, save_checkpoint_mp
import multiprocessing as mp
from queue import Empty
import wandb
import json
import os
import time

def train_group_seperate_wandb(saes, activation_store, model, cfgs):

    num_batches = int(cfgs[0]["num_tokens"] // cfgs[0]["batch_size"])
    print(f"Number of batches: {num_batches}")
    optimizers = [
        torch.optim.Adam(
            sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"])
        )
        for sae, cfg in zip(saes, cfgs)
    ]
    pbar = tqdm.trange(num_batches)

    # Initialize wandb processes and queues
    wandb_processes = []
    log_queues = []

    for i, cfg in enumerate(cfgs):
        log_queue = mp.Queue()
        log_queues.append(log_queue)
        wandb_config = cfg
        wandb_process = mp.Process(
            target=new_wandb_process,
            args=(
                wandb_config,
                log_queue,
                cfg.get("wandb_entity", ""),
                cfg["wandb_project"],
            ),
        )
        time.sleep(3)
        wandb_process.start()
        time.sleep(3)
        wandb_processes.append(wandb_process)

    for i in pbar:
        batch = activation_store.next_batch()

        for idx, (sae, cfg, optimizer) in enumerate(
            zip(saes, cfgs, optimizers)
        ):
            sae_output = sae(batch)
            loss = sae_output["loss"]

            # Log metrics to appropriate wandb process
            log_dict = {
                k: (
                    v.item()
                    if isinstance(v, torch.Tensor) and v.dim() == 0
                    else v
                )
                for k, v in sae_output.items()
                if isinstance(v, (int, float))
                or (isinstance(v, torch.Tensor) and v.dim() == 0)
            }
            log_queues[idx].put(log_dict)

            if i % cfg["checkpoint_freq"] == 0:
                # Save checkpoint and send artifact info to wandb process
                save_dir, _, _ = save_checkpoint_mp(sae, cfg, i)
                log_queues[idx].put(
                    {"checkpoint": True, "step": i, "save_dir": save_dir}
                )

            pbar.set_postfix(
                {
                    f"Loss_{idx}": f"{loss.item():.4f}",
                    f"L0_{idx}": f"{sae_output['l0_norm']:.4f}",
                    f"L2_{idx}": f"{sae_output['l2_loss']:.4f}",
                    f"L1_{idx}": f"{sae_output['l1_loss']:.4f}",
                }
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                sae.parameters(), cfg["max_grad_norm"]
            )

            optimizer.step()
            optimizer.zero_grad()

    # Final checkpoints
    for idx, (sae, cfg) in enumerate(zip(saes, cfgs)):
        save_dir, _, _ = save_checkpoint_mp(sae, cfg, i)
        log_queues[idx].put(
            {"checkpoint": True, "step": i, "save_dir": save_dir}
        )

    # Clean up wandb processes
    for queue in log_queues:
        queue.put("DONE")
    for process in wandb_processes:
        process.join()
