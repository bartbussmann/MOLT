import torch
import torch.nn as nn
import torch.nn.functional as F


class MOLT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        torch.manual_seed(cfg.get("seed", 0))
        self.cfg = cfg

        # ---------------- core dims ------------------------------------
        self.d_model = cfg["act_size"]
        self.rank_groups = cfg["rank_groups"]
        self.n_transforms = sum([n for n, _ in self.rank_groups])

        # ---------------- encoder (gating) -----------------------------
        self.b_enc = nn.Parameter(torch.zeros(self.n_transforms))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(self.d_model, self.n_transforms)
            )
        )

        # ---------- assign transform indexes to same-rank groups --------
        counts = [n_t for n_t, _ in self.rank_groups]
        full_idx = torch.arange(self.n_transforms, dtype=torch.long)
        self.group_indices: List[torch.LongTensor] = list(torch.split(full_idx, counts))

        # ---------------- initialize transform parameters ---------------
        self.U_groups = nn.ParameterList()
        self.V_groups = nn.ParameterList()
        for n, rank in self.rank_groups:
            U = nn.Parameter(
                torch.nn.init.kaiming_uniform_(
                    torch.empty(n, self.d_model, rank)
                )
            )

            V = nn.Parameter(
                torch.nn.init.kaiming_uniform_(
                    torch.empty(n, rank, self.d_model)
                )
            )

            self.U_groups.append(U)
            self.V_groups.append(V)

        self.to(cfg["device"]).to(cfg["dtype"])


    def encode(self, model_acts):
        x, target = model_acts
        x_flat = x.view(-1, self.d_model)
        acts = F.relu(x_flat @ self.W_enc - self.b_enc)  # (B, T)
        return acts
            
            
    def forward(self, model_acts):
        x, target = model_acts
        orig_shape = x.shape
        x_flat = x.view(-1, self.d_model)
        out_flat = torch.zeros_like(x_flat)

        acts = F.relu(x_flat @ self.W_enc - self.b_enc)  # (B, T)

        sparsity_loss = 0.0
        total_l0 = 0.0
        for group_idx, ((n_t, k_t), idx, U_t, V_t) in enumerate(zip(
            self.rank_groups, self.group_indices, self.U_groups, self.V_groups
        )):
            acts_g = acts[:, idx]  # (B, n_t)
            inner = torch.einsum("bd,Gkd->bGk", x_flat, V_t) * acts_g.unsqueeze(-1)
            out_flat += torch.einsum("bGk,Gdk->bd", inner, U_t)

            group_sparsity = self.group_sparsity_penalty(acts_g, U_t, V_t)
            sparsity_loss += group_sparsity
            
            group_l0 = (acts_g > 0).float().sum(dim=1)
            total_l0 += group_l0

        out = out_flat.view(orig_shape)
        
        recon_loss = F.mse_loss(out.squeeze(), target.squeeze(), reduction="mean")
        
        total_loss = recon_loss + sparsity_loss
        
        l0_mean = total_l0.mean()

        target_flat = target.view(-1, self.d_model)
        frac_variance_unexplained = torch.var(target_flat - out_flat, dim=0, unbiased=False).mean() / torch.var(target_flat, dim=0, unbiased=False).mean()

        result = {
            "sae_out": out,
            "loss": total_loss,
            "l2_loss": recon_loss,
            "l1_loss": sparsity_loss,
            "l0_norm": l0_mean,
            "frac_variance_unexplained": frac_variance_unexplained,
        }
        return result

    def group_sparsity_penalty(self, acts, U, V):
        # ‖U_t V_t‖_F  via trace(UᵀU · V Vᵀ)
        UtU   = torch.bmm(U.transpose(1, 2), U)        # (n, k, k)
        VtV   = torch.bmm(V, V.transpose(1, 2))        # (n, k, k)
        frob2 = torch.einsum('gii->g', UtU @ VtV)      # trace per transform
        frob  = torch.sqrt(frob2 + 1e-8)

        mean_act     = acts.abs().mean(0)              # average over batch
        sparsity_pen = torch.dot(frob, mean_act)
        return self.cfg["l1_coeff"] * sparsity_pen



class Transcoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        torch.manual_seed(cfg.get("seed", 0))
        self.cfg = cfg

        # core dims
        self.d_model     = cfg["act_size"]
        self.hidden_size = cfg["hidden_size"]

        # ----- encoder -------------------------------------------------
        self.b_enc = nn.Parameter(torch.zeros(self.hidden_size))
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(self.d_model, self.hidden_size))
        )

        # ----- decoder -------------------------------------------------

        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(torch.empty(self.hidden_size, self.d_model))
        )
        self.b_dec = nn.Parameter(torch.zeros(self.d_model))

        # send everything to right device / dtype
        self.to(cfg.get("device", "cpu")).to(cfg.get("dtype", torch.float32))

    # ------------------------------------------------------------------
    def forward(self, model_acts):
        x, target = model_acts                       # both tensors shaped like activations
        orig_shape = x.shape

        x_flat   = x.view(-1, self.d_model)          # (B*…, D)
        hidden   = F.relu(x_flat @ self.W_enc - self.b_enc)
        out_flat = hidden @ self.W_dec + self.b_dec  # (B*…, D)
        out      = out_flat.view(orig_shape)

        # ---------- losses -------------------------------------------
        recon_loss   = F.mse_loss(out.squeeze(), target.squeeze(), reduction="mean")
        sparsity_pen = self.sparsity_penalty(hidden)
        total_loss   = recon_loss + sparsity_pen
        l0_mean      = (hidden > 0).float().sum(dim=1).mean()

        return {
            "sae_out":  out,
            "loss":     total_loss,
            "l2_loss":  recon_loss,
            "l1_loss":  sparsity_pen,
            "l0_norm":  l0_mean,
        }


    def sparsity_penalty(self, hidden: torch.Tensor) -> torch.Tensor:
        """
        MOLT‑style sparsity penalty for a rank‑1 Transcoder.

            penalty = λ · Σ_i  (‖W_dec[i]‖₂ · ‖W_enc[:,i]‖₂) · E_b[|hidden_{b,i}|]

        where E_b is the batch mean.
        """
        # Frobenius norm of each rank‑1 transform T_i = u_i v_i^T
        dec_norm = torch.norm(self.W_dec, dim=1)      # (H,)   ‖u_i‖
        transform_norm = dec_norm          # (H,)
        mean_act = hidden.abs().mean(dim=0)           # (H,)   E_b[|h_{b,i}|]

        return self.cfg.get("l1_coeff", 0.0) * torch.dot(transform_norm, mean_act)