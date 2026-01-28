#!/usr/bin/env python
import gymnasium as gym
import itertools
import numpy as np
from collections import namedtuple, deque
import random
import torch
from torch import nn
import copy
import h5py
# device = torch.device("cpu") 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'mps'
import warnings
from torch.distributions import Categorical
from FactualEncoder import TSTransformerWithMask, train_epoch_efficient_online, validate_epoch_efficient_online
import matplotlib.pyplot as plt
import torch.nn.functional as F

import os
import subprocess

def auto_select_gpu():
    """
    Automatically select the GPU with the most free memory.
    """
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,nounits,noheader"],
            encoding="utf-8"
        )
        free_memories = [int(x) for x in result.strip().split("\n")]
        best_gpu = int(max(range(len(free_memories)), key=lambda i: free_memories[i]))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu)
        print(f"[Auto GPU] Using GPU {best_gpu} with {free_memories[best_gpu]} MiB free")
    except Exception as e:
        print("[Auto GPU] Failed to auto-select GPU, fallback to default.", e)

auto_select_gpu()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))
# This is the change environment version of the agent class

class memory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class neural_network(nn.Module):
    '''
    Feedforward neural network with variable number
    of hidden layers and ReLU nonlinearites
    '''

    def __init__(self,
                layers=[14,64,32,4],# layers[i] = # of neurons at i-th layer
                # layers[0] = input layer
                # layers[-1] = output layer
                dropout=False,
                p_dropout=0.5,
                ):
        super(neural_network,self).__init__()

        self.network_layers = []
        n_layers = len(layers)
        for i,neurons_in_current_layer in enumerate(layers[:-1]):
            #
            self.network_layers.append(nn.Linear(neurons_in_current_layer, 
                                                layers[i+1]) )
            #
            if dropout:
                self.network_layers.append( nn.Dropout(p=p_dropout) )
            #
            if i < n_layers - 2:
                self.network_layers.append( nn.ReLU() )
        #
        self.network_layers = nn.Sequential(*self.network_layers)
        #

    def forward(self,x):
        for layer in self.network_layers:
            x = layer(x)
        return x

class TwoBranchDQN(nn.Module):
    def __init__(self, state_dim=8, env_dim=3, context_dim=3, n_actions=4):
        super(TwoBranchDQN, self).__init__()

        # Branch 1: Processing the original state (8)
        self.state_branch = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )

        # Branch 2: Processing environment vectors (3)
        self.env_branch = nn.Sequential(
            nn.Linear(env_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        # Branch 3: Processing context vectors (3)
        self.context_branch = nn.Sequential(
            nn.Linear(context_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 16)
        )

        # Merge and output Q value
        self.q_head = nn.Sequential(
            nn.Linear(16 + 16 + 16, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.Linear(32, n_actions)
        )

    def forward(self, x):
        # Split input: The first 8 dimensions are the original state, and the last 3 dimensions are environment variables
        state_input = x[:, :8]
        env_input = x[:, 8:11]
        context_input = x[:, 11:]

        state_feat = self.state_branch(state_input)
        env_feat = self.env_branch(env_input)
        context_feat = self.context_branch(context_input)

        combined = torch.cat((state_feat, env_feat, context_feat), dim=1)
        q_values = self.q_head(combined)

        return q_values

import csv
import os

def save_validation_metrics_to_csv(val_ap_list, val_an_list, stride=1, filename="validation_metrics.csv"):
    os.makedirs("logs", exist_ok=True)  
    filepath = os.path.join("logs", filename)

    with open(filepath, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Avg_Pos_Dist", "Avg_Neg_Dist"])
        for i, (ap, an) in enumerate(zip(val_ap_list, val_an_list)):
            writer.writerow([i * stride, ap, an])
    print(f"[Saved] Validation metrics written to {filepath}")

def save_returns_to_csv(returns, filename="episode_returns.csv"):
    os.makedirs("logs", exist_ok=True) 
    filepath = os.path.join("logs", filename)
    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode_return"])  # header
        for r in returns:
            writer.writerow([r])
    print(f"[Saved] Episode returns to {filename}")

def save_episode_stats_to_csv(
    episode_returns,
    episode_durations,
    successful_episodes,
    filename="episode_stats.csv"
):
    os.makedirs("logs", exist_ok=True)
    filepath = os.path.join("logs", filename)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode_return",
            "episode_duration",
            "successful_episode"
        ])
        for r, d, s in zip(
            episode_returns,
            episode_durations,
            successful_episodes
        ):
            writer.writerow([r, d, int(s)])

    print(f"[Saved] Episode stats to {filepath}")

def save_context_metrics_to_csv(context_logs,
                                filename="context_contrastive_metrics.csv"):
    os.makedirs("logs", exist_ok=True)
    filepath = os.path.join("logs", filename)

    with open(filepath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "step",
            "loss_ctx",
            "risk_mean",
            "risk_std",
            "overlap@10",
            "p_entropy",
            "kl_t_p",
        ])
        for row in context_logs:
            writer.writerow([
                row["loss_ctx"],
                row["risk_mean"],
                row["risk_std"],
                row["overlap@10"],
                row["p_entropy"],
                row["kl_t_p"],
            ])

    print(f"[Saved] Context contrastive metrics to {filepath}")

def sample_phi(mode="mix", device="cpu", dtype=torch.float32):
        """
        mode:
        - "neutral": always [1,1]  (your benchmark)
        - "mix": randomly choose goal or safety (50/50)
        - "goal": always goal
        - "safety": always safety
        """
        if mode == "neutral":
            phi = torch.tensor([1.0, 1.0], device=device, dtype=dtype)
        elif mode == "goal":
            phi = torch.tensor([0.9, 0.1], device=device, dtype=dtype)
        elif mode == "safety":
            phi = torch.tensor([0.1, 0.9], device=device, dtype=dtype)
        else:  # "mix"
            if torch.rand(1).item() < 0.5:
                phi = torch.tensor([0.9, 0.1], device=device, dtype=dtype)
            else:
                phi = torch.tensor([0.1, 0.9], device=device, dtype=dtype)
        return phi

class TrackEmbeddingWrapper:
    def __init__(self, transformer_model_path, device='cuda', max_len=50):
        self.device = device
        self.max_len = max_len
        self.track = []
        self.mask = []

        # Initialization of the Encoder model
        self.model = TSTransformerWithMask()
        self.model.load_state_dict(torch.load(transformer_model_path, map_location=device))
        self.model.to(self.device)
        self.model.eval()

    def reset(self):
        self.track = []
        self.mask = []

    def update(self, state, action):
        self.track.append(np.concatenate([state[:8], [action]]))
        self.mask.append(1)
        if len(self.track) > self.max_len:
            self.track.pop(0)
            self.mask.pop(0)

    def get_embedding(self):
        track = np.zeros((self.max_len, 9), dtype=np.float32)  # [50, 9]
        mask = np.zeros((self.max_len,), dtype=np.bool_)       # [50]

        valid_len = len(self.track)
        track[-valid_len:] = self.track
        mask[-valid_len:] = self.mask

        track_tensor = torch.tensor(track, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask_tensor = torch.tensor(mask, dtype=torch.bool).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(track_tensor, mask=mask_tensor)  # [1, 3]
        return embedding.squeeze(0)

class ContextBeliefNet(nn.Module):
    """
    Contextual belief inference network.
    Input:
        e_seq: (B, T, e_dim)  factual embedding sequence (e.g., e_dim=3)
        r_seq: (B, T) or (B, T, 1) reward sequence
        phi:   (B, phi_dim) or (phi_dim,) personality vector (constant per episode)
    Output:
        logits: (B, 3)  raw logits for [danger, shifting, safe]
        z:      (B, 3)  soft belief (softmax)
    """
    def __init__(
        self,
        e_dim: int = 3,
        phi_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.e_dim = e_dim
        self.phi_dim = phi_dim

        in_dim =  e_dim + 1 + phi_dim # e + r + phi

        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # φ-conditioned feature modulation (FiLM-style)
        self.phi_film = nn.Sequential(
            nn.Linear(phi_dim, 2 * in_dim),
        )
        self.embed_dim = 16
        self.tar_embed_dim = 3  # [danger, shifting, safe]
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.embed_dim),  # [danger, shifting, safe] x now high dimension
            nn.Linear(self.embed_dim, self.tar_embed_dim),
            nn.LayerNorm(self.tar_embed_dim)
        )

    @staticmethod
    def _ensure_3d_seq(x: torch.Tensor) -> torch.Tensor:
        """
        Ensure x is (B, T, 1) if given (B, T).
        """
        if x.dim() == 2:
            return x.unsqueeze(-1)
        return x
    
    def get_embedding(self, e_seq, r_seq, phi):
        """
        Return continuous embedding for contrastive learning: [B, embed_dim]
        """
        # reuse the same preprocessing as forward()
        B, T, _ = e_seq.shape
        if r_seq.dim() == 2:
            r_in = r_seq.unsqueeze(-1)  # [B,T,1]
        else:
            r_in = r_seq

        phi_seq = phi.unsqueeze(1).expand(-1, T, -1)      # [B,T,2]
        x = torch.cat([e_seq, r_in, phi_seq], dim=-1)     # [B,T,in_dim]

        # apply FiLM if you have it
        if hasattr(self, "phi_film"):
            film = self.phi_film(phi)                     # [B, 2*in_dim]
            gamma, beta = film.chunk(2, dim=-1)           # [B,in_dim], [B,in_dim]
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
            x = (1.0 + gamma) * x + beta

        out, h = self.gru(x)
        h_last = h[-1]                                    # [B,hidden_dim]
        z = self.head(h_last)                             # [B,embed_dim]
        return z

    def forward(
        self,
        e_seq: torch.Tensor,   # (B, T, e_dim)
        r_seq: torch.Tensor,   # (B, T) or (B, T, 1)
        phi: torch.Tensor      # (B, phi_dim) or (phi_dim,)
    ):
        assert e_seq.dim() == 3, f"e_seq must be (B,T,e_dim), got {e_seq.shape}"

        B, T, e_dim = e_seq.shape
        assert e_dim == self.e_dim, f"Expected e_dim={self.e_dim}, got {e_dim}"

        # r: (B, T, 1)
        r_seq = self._ensure_3d_seq(r_seq).to(e_seq.dtype)
        assert r_seq.shape[:2] == (B, T), f"r_seq must match (B,T), got {r_seq.shape}"

        # phi: (B, phi_dim)
        if phi.dim() == 1:
            # (phi_dim,) -> (B, phi_dim)
            phi = phi.unsqueeze(0).expand(B, -1)
        assert phi.shape == (B, self.phi_dim), f"phi must be (B,phi_dim), got {phi.shape}"

        # -----------------------------
        def _delta_lag(e: torch.Tensor, k: int) -> torch.Tensor:
            """
            e: (B,T,e_dim). Return delta^(k): (B,T,e_dim)
            delta[:, :k] = 0
            delta[:, k:] = e[:, k:] - e[:, :-k]
            """
            B_, T_, D_ = e.shape
            d = torch.zeros_like(e)
            if T_ > k:
                d[:, k:, :] = e[:, k:, :] - e[:, :-k, :]
            return d

        phi_seq = phi.unsqueeze(1).expand(B, T, self.phi_dim).to(e_seq.dtype)

        x = torch.cat([e_seq, r_seq, phi_seq], dim=-1)

        # x: (B,T,in_dim)
        film = self.phi_film(phi)          # (B, 2*in_dim)
        gamma, beta = torch.chunk(film, 2, dim=-1)  # each (B,in_dim)
        gamma = gamma.unsqueeze(1)         # (B,1,in_dim)
        beta  = beta.unsqueeze(1)          # (B,1,in_dim)

        # modulate
        x = (1.0 + gamma) * x + beta

        # GRU encode
        out, h_n = self.gru(x)              # out: (B,T,H), h_n: (num_layers,B,H)
        h_last = h_n[-1]                    # (B,H)

        logits = self.head(h_last)          # (B,3)
        z = torch.softmax(logits, dim=-1)   # (B,3)

        return logits, z
    
class ContextBeliefTracker(nn.Module):
    """
    Online tracker for contextual belief z_t using ContextBeliefNet.

    Maintains a sliding window of e_t and r_t (length T_context).
    Handles early steps by left-padding with zeros to length T_context.
    Optionally returns a default belief until valid_len >= min_valid.

    z_t meaning: [danger, shifting, safe]
    """
    def __init__(
        self,
        context_net: nn.Module,
        T_context: int = 5,
        e_dim: int = 3,
        phi_dim: int = 2,
        min_valid: int = 5,
        default_z: torch.Tensor = None,
        device: torch.device = device,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.context_net = context_net
        self.T = int(T_context)
        self.e_dim = int(e_dim)
        self.phi_dim = int(phi_dim)
        self.min_valid = int(min_valid)
        self.device = device
        self.dtype = dtype

        self.e_hist = deque(maxlen=self.T)
        self.r_hist = deque(maxlen=self.T)
        # train-only
        self.shift_buf = deque(maxlen=self.T)  # store is_shift (0/1)
        self.window_bank = deque(maxlen=1000)

        if default_z is None:
            # default to "safe"
            default_z = torch.tensor([0.33, 0.33, 0.34], dtype=self.dtype)
        self.default_z = default_z.to(self.device, dtype=self.dtype)

        # --- phi-conditioned gates ---
        self.phi2gate = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 2)   # [g_goal, g_safe]
        ).to(self.device)

        # --- phi-conditioned allocation inside volatility group ---
        self.phi2alpha = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 4)   # [alpha_sigma, alpha_C, alpha_E, alpha_shift]
        ).to(self.device)

    def reset(self):
        """Call at episode reset."""
        self.e_hist.clear()
        self.r_hist.clear()
        self.shift_buf.clear()
    @staticmethod
    def _pad_list(seq_list, T, pad_value: torch.Tensor):
        """
        Left-pad a list to length T with pad_value (cloned).
        If longer than T, keep the last T.
        """
        if len(seq_list) >= T:
            return list(seq_list)[-T:]
        pad_count = T - len(seq_list)
        pads = [pad_value.clone() for _ in range(pad_count)]
        return pads + list(seq_list)

    def _build_batch_inputs(self, phi: torch.Tensor):
        """
        Build e_seq and r_seq tensors with shape:
          e_seq: (1, T, e_dim)
          r_seq: (1, T)  (ContextBeliefNet also accepts (1,T,1), but we keep (1,T))
        """
        pad_e = torch.zeros(self.e_dim, device=self.device, dtype=self.dtype)
        pad_r = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        e_list = self._pad_list(self.e_hist, self.T, pad_e)
        r_list = self._pad_list(self.r_hist, self.T, pad_r)

        e_seq = torch.stack(e_list, dim=0).unsqueeze(0)  # (1,T,e_dim)
        r_seq = torch.stack(r_list, dim=0).unsqueeze(0)  # (1,T)

        # phi can be (phi_dim,) or (1,phi_dim); ContextBeliefNet handles both.
        if isinstance(phi, np.ndarray):
            phi = torch.from_numpy(phi)
        phi = phi.to(self.device, dtype=self.dtype)

        return e_seq, r_seq, phi

    def update(self, e_t, r_t, phi):
        """
        Update buffers with the latest e_t and r_t, then compute z_t.

        Args:
            e_t: torch.Tensor or np.ndarray, shape (e_dim,)
            r_t: float or torch scalar
            phi: torch.Tensor or np.ndarray, shape (phi_dim,) or (1,phi_dim)

        Returns:
            z_t: torch.Tensor, shape (3,)  (on self.device)
        """
        # to tensor
        if isinstance(e_t, np.ndarray):
            e_t = torch.from_numpy(e_t)
        e_t = e_t.to(self.device, dtype=self.dtype).view(-1)
        assert e_t.numel() == self.e_dim, f"e_t dim mismatch: got {e_t.numel()}, expected {self.e_dim}"

        if isinstance(r_t, torch.Tensor):
            r_val = r_t.detach().to(self.device, dtype=self.dtype).view(())
        else:
            r_val = torch.tensor(float(r_t), device=self.device, dtype=self.dtype)

        self.e_hist.append(e_t)
        self.r_hist.append(r_val)

        valid_len = len(self.e_hist)

        # Early steps: return default belief before we have enough evidence
        if valid_len < self.min_valid:
            return self.default_z.clone()

        # Build padded sequences and run net
        e_seq, r_seq, phi = self._build_batch_inputs(phi)

        # IMPORTANT: do NOT wrap with torch.no_grad() here, because we may
        # later choose to enable gradient-based training paths.
        logits, z = self.context_net(e_seq, r_seq, phi)  # z: (1,3)

        return z.squeeze(0).detach()
    
    def push_shift(self, is_shift: int):
        """
        train-only: call this in train loop right after you decide is_shift
        """
        self.shift_buf.append(torch.tensor([float(is_shift)], device=self.device))
    
    
    def push_window_to_bank(self, phi):
        if len(self.e_hist) < self.T or len(self.r_hist) < self.T or len(self.shift_buf) < self.T:
            return
        e_seq = torch.stack(list(self.e_hist), dim=0).detach()
        r_seq = torch.stack([x.view(-1) for x in self.r_hist], dim=0).squeeze(-1).detach()
        s_seq = torch.stack([x.view(-1) for x in self.shift_buf], dim=0).squeeze(-1).detach()
        phi_t = phi.detach().to(self.device, dtype=self.dtype)
        self.window_bank.append((e_seq, r_seq, s_seq, phi_t))
    
    def _window_stats(self, e_seq, r_seq, s_seq):
        device = e_seq.device

        # 1) e-related change magnitude
        if e_seq.shape[0] >= 2:
            de = e_seq[1:] - e_seq[:-1]
            C = torch.norm(de, dim=-1).mean()
        else:
            C = torch.tensor(0.0, device=device)

        e_norm = torch.norm(e_seq, dim=-1)                 # shape [T]
        e_norm_std = e_norm.std(unbiased=False)            # scalar

        # 2) reward stats
        mu = r_seq.mean()
        sigma = r_seq.std(unbiased=False)

        # 3) shift: keep old binary flag for compatibility
        shift_flag = (s_seq.max() > 0.5).float()

        # 4) NEW: continuous shift strength based on position of most recent shift in the window
        T = int(s_seq.shape[0])  # should be 5
        idx = (s_seq > 0.5).nonzero(as_tuple=False).flatten()  # indices: 0..T-1
        if idx.numel() == 0:
            shift_strength = torch.tensor(0.0, device=device)
        else:
            k_star = int(idx.max().item()) + 1  # 1..T, most recent shift
            shift_strength = torch.tensor(k_star / T, device=device)

        return C, e_norm_std, mu, sigma, shift_flag, shift_strength
    
    def _robust_norm01(self, x, q10, q90, eps=1e-6):
        # x, q10, q90 are torch scalars
        denom = (q90 - q10).clamp_min(eps)
        y = (x - q10) / denom
        return y.clamp(0.0, 1.0)
    
    def _compute_feature_quantiles_from_bank(self, max_items=400):
        """
        Return quantiles for C, e_std, mu, sigma, s_strength based on recent windows in bank.
        If bank too small, return None to fallback to simple normalization.
        """
        if len(self.window_bank) < 30:
            return None

        # take last max_items windows
        items = list(self.window_bank)[-max_items:]

        C_list, e_list, mu_list, sig_list, s_list = [], [], [], [], []
        for (e_seq, r_seq, s_seq, phi) in items:
            # ensure tensors
            if not torch.is_tensor(e_seq):
                e_seq = torch.as_tensor(e_seq)
            if not torch.is_tensor(r_seq):
                r_seq = torch.as_tensor(r_seq)
            if not torch.is_tensor(s_seq):
                s_seq = torch.as_tensor(s_seq)

            # make sure shapes are [T], [T], [T]
            if r_seq.dim() > 1:
                r_seq = r_seq.squeeze(-1)
            if s_seq.dim() > 1:
                s_seq = s_seq.squeeze(-1)

            C, e_norm_std, mu, sigma, _, s_strength = self._window_stats(e_seq, r_seq, s_seq)

            C_list.append(C.detach().cpu())
            e_list.append(e_norm_std.detach().cpu())
            mu_list.append(mu.detach().cpu())
            sig_list.append(sigma.detach().cpu())
            s_list.append(s_strength.detach().cpu())

        C_t = torch.stack(C_list)
        e_t = torch.stack(e_list)
        mu_t = torch.stack(mu_list)
        sig_t = torch.stack(sig_list)
        s_t = torch.stack(s_list)

        qs = {}
        for name, t in [("C", C_t), ("e", e_t), ("mu", mu_t), ("sig", sig_t), ("s", s_t)]:
            qs[name] = {
                "q10": torch.quantile(t, 0.10),
                "q90": torch.quantile(t, 0.90),
            }
        return qs
    
    def compute_risk_score(self, C, e_norm_std, mu, sigma, s_strength, phi, quantiles=None):
        """
        Continuous risk in [0,1].
        phi: tensor shape [2] -> [goal, safety] assumed.
        quantiles: dict from _compute_feature_quantiles_from_bank()
        """
        device = C.device
        goal = float(phi[0].item()) if torch.is_tensor(phi) else float(phi[0])
        safety = float(phi[1].item()) if torch.is_tensor(phi) else float(phi[1])

        # Normalize features to [0,1]
        if quantiles is not None:
            Cn   = self._robust_norm01(C,         quantiles["C"]["q10"].to(device),   quantiles["C"]["q90"].to(device))
            En   = self._robust_norm01(e_norm_std, quantiles["e"]["q10"].to(device),   quantiles["e"]["q90"].to(device))
            Mun  = self._robust_norm01(mu,        quantiles["mu"]["q10"].to(device),  quantiles["mu"]["q90"].to(device))
            Sign = self._robust_norm01(sigma,     quantiles["sig"]["q10"].to(device), quantiles["sig"]["q90"].to(device))
            Sn   = self._robust_norm01(s_strength,quantiles["s"]["q10"].to(device),   quantiles["s"]["q90"].to(device))
        else:
            # fallback: simple clamps (crude but safe)
            Cn, En = C.clamp(0, 1), e_norm_std.clamp(0, 1)
            Mun = mu.clamp(-200, 200) / 400.0 + 0.5  # map roughly to [0,1]
            Mun = Mun.clamp(0, 1)
            Sign = (sigma.clamp(0, 200) / 200.0).clamp(0, 1)
            Sn = s_strength.clamp(0, 1)

        # "danger evidence": low mean reward -> high risk
        mu_danger = 1.0 - Mun

        # ===== phi-conditioned gates =====
        raw_gate = self.phi2gate(phi.unsqueeze(0)).squeeze(0)   # [2]
        g_goal, g_safe = torch.nn.functional.softplus(raw_gate)

        # optional: squash to (0,1) to avoid saturation
        g_goal = g_goal / (1.0 + g_goal)
        g_safe = g_safe / (1.0 + g_safe)

        # ===== phi-conditioned allocation inside volatility group =====
        alpha_logits = self.phi2alpha(phi.unsqueeze(0)).squeeze(0)  # [4]
        alpha = torch.softmax(alpha_logits, dim=0)                  # sum to 1

        # unpack alpha
        a_sig, a_C, a_E, a_sh = alpha

        # volatility block (learned composition)
        vol_block = (
            a_sig * Sign +
            a_C   * Cn   +
            a_E   * En   +
            a_sh  * Sn
        )

        # ===== final risk =====
        risk = (g_goal * mu_danger + g_safe * vol_block).clamp(0.0, 1.0)
        return risk

    def _assign_bucket(self, C, mu, sigma, shift_flag, phi,
                       mu_low=-0.2, mu_high=0.2,
                       sigma_high=0.8, C_high=0.25):
        """
        return bucket id: 0=danger, 1=shift, 2=safe
        """
        # phi: [2]  (goal, safety)
        # allow neutral [1,1] too
        phi = phi.detach().float().view(-1)
        phi_sum = float(phi.sum().item()) if phi.numel() > 0 else 1.0
        phi_norm = phi / max(phi_sum, 1e-6)

        goal_w   = float(phi_norm[0].item()) if phi_norm.numel() >= 1 else 0.5
        safety_w = float(phi_norm[1].item()) if phi_norm.numel() >= 2 else 0.5

        # safety -> more sensitive to volatility/change => lower thresholds
        sigma_high_eff = sigma_high * (1.15 - 0.6 * safety_w)   # safety_w=1 => *0.55 ; goal_w=1 => *1.15
        C_high_eff     = C_high     * (1.15 - 0.6 * safety_w)

        # goal -> more tolerant to volatility/change, easier to call safe by mean reward
        mu_high_eff = mu_high - 0.15 * goal_w   # goal => lower safe threshold
        mu_low_eff  = mu_low  - 0.10 * goal_w   # goal => less likely to call danger by low mu

        mu_high = mu_high_eff
        mu_low = mu_low_eff
        sigma_high = sigma_high_eff
        C_high = C_high_eff
        if shift_flag.item() > 0.5:
            return 1  # shift
        # danger: low mean reward OR high reward volatility
        if (mu.item() < mu_low) or (sigma.item() > sigma_high):
            return 0  # danger
        # safe: high mean reward AND low volatility AND low change
        if (mu.item() > mu_high) and (sigma.item() <= sigma_high) and (C.item() < C_high):
            return 2  # safe
        # fallback: if uncertain, treat as danger (or you can randomize/choose nearest class)
        return 0

    @torch.no_grad()
    def _build_window(self):
        T = len(self.e_hist)
        e_seq = torch.stack(list(self.e_hist), dim=0)                    # [T,3]
        r_seq = torch.stack([x.view(-1) for x in self.r_hist], dim=0).squeeze(-1)  # [T]
        s_seq = torch.stack([x.view(-1) for x in self.shift_buf], dim=0).squeeze(-1) if len(self.shift_buf)>0 \
                else torch.zeros(T, device=self.device)
        return e_seq, r_seq, s_seq

    def compute_contrastive_loss(self, batch_size=32, tau=0.2, tau_g=0.15, tau_phi=1.0, return_metrics=False):
        """
        Soft-weighted contrastive based on continuous risk score.
        tau: temperature for embedding similarity
        tau_g: bandwidth for risk distance -> soft positives
        tau_phi: bandwidth for phi distance (if phi varies; if fixed, has little effect)
        """
        if len(self.window_bank) < batch_size:
            return None

        device = self.device if hasattr(self, "device") else next(self.context_net.parameters()).device

        # 1) sample windows
        idxs = np.random.choice(len(self.window_bank), size=batch_size, replace=False)
        batch = [self.window_bank[i] for i in idxs]

        # 2) build batch tensors (pad to T)
        e_seqs, r_seqs, s_seqs, phis = [], [], [], []
        for (e_seq, r_seq, s_seq, phi) in batch:
            # to tensors on device
            e_seq = torch.as_tensor(e_seq, device=device, dtype=torch.float32)
            r_seq = torch.as_tensor(r_seq, device=device, dtype=torch.float32)
            s_seq = torch.as_tensor(s_seq, device=device, dtype=torch.float32)
            phi   = torch.as_tensor(phi,   device=device, dtype=torch.float32)

            if r_seq.dim() > 1:
                r_seq = r_seq.squeeze(-1)
            if s_seq.dim() > 1:
                s_seq = s_seq.squeeze(-1)

            # ensure length T (self.T)
            T = self.T
            if e_seq.shape[0] < T:
                pad_n = T - e_seq.shape[0]
                e_pad = torch.zeros((pad_n, e_seq.shape[-1]), device=device)
                r_pad = torch.zeros((pad_n,), device=device)
                s_pad = torch.zeros((pad_n,), device=device)
                e_seq = torch.cat([e_pad, e_seq], dim=0)
                r_seq = torch.cat([r_pad, r_seq], dim=0)
                s_seq = torch.cat([s_pad, s_seq], dim=0)
            elif e_seq.shape[0] > T:
                e_seq = e_seq[-T:]
                r_seq = r_seq[-T:]
                s_seq = s_seq[-T:]

            e_seqs.append(e_seq)
            r_seqs.append(r_seq)
            s_seqs.append(s_seq)
            phis.append(phi)

        e_seqs = torch.stack(e_seqs, dim=0)               # [B,T,3]
        r_seqs = torch.stack(r_seqs, dim=0)               # [B,T]
        phis   = torch.stack(phis, dim=0)                 # [B,2]

        # 3) compute risk for each window (using bank-based quantiles)
        quantiles = self._compute_feature_quantiles_from_bank()
        risks = []
        for b in range(batch_size):
            C, e_norm_std, mu, sigma, _, s_strength = self._window_stats(
                e_seqs[b], r_seqs[b], s_seqs[b]
            )
            risk_b = self.compute_risk_score(C, e_norm_std, mu, sigma, s_strength, phis[b], quantiles)
            risks.append(risk_b)
        risks = torch.stack(risks, dim=0).view(-1)        # [B]

        # 4) get embeddings from context_net
        # IMPORTANT: use a higher-dim embedding than 3-d logits to avoid collapse
        # We will use GRU last hidden state if you exposed it; if not, add a projection head.
        z_embed = self.context_net.get_embedding(e_seqs, r_seqs, phis)  # [B, D]
        z_embed = torch.nn.functional.normalize(z_embed, dim=-1)

        # 5) similarity-based prediction distribution p_ij
        sim = (z_embed @ z_embed.T) / tau                 # [B,B]
        sim = sim - torch.eye(batch_size, device=device) * 1e9   # mask diagonal
        p = torch.softmax(sim, dim=1)                     # [B,B]

        # 6) build soft target distribution t_ij based on |risk_i-risk_j| and phi distance
        dr = (risks.view(-1,1) - risks.view(1,-1)).abs()  # [B,B]
        dphi = torch.norm(phis.view(batch_size,1,-1) - phis.view(1,batch_size,-1), dim=-1)  # [B,B]

        # weights (exclude diagonal)
        w = torch.exp(-dr / tau_g) * torch.exp(-dphi / tau_phi)
        w = w * (1.0 - torch.eye(batch_size, device=device))

        # normalize to distribution
        t = w / (w.sum(dim=1, keepdim=True) + 1e-8)       # [B,B]

        # 7) cross-entropy between target distribution t and predicted distribution p
        loss = -(t * torch.log(p + 1e-8)).sum(dim=1).mean()

        # -------------------------
        # Diagnostics / metrics
        # -------------------------
        metrics = {}

        # (A) risk stats
        metrics["risk_mean"] = float(risks.mean().item())
        metrics["risk_std"]  = float(risks.std(unbiased=False).item())
        metrics["risk_min"]  = float(risks.min().item())
        metrics["risk_max"]  = float(risks.max().item())

        # (B) neighbor overlap@K: compare top-K neighbors by embedding distance vs risk distance
        # embedding distance: smaller is closer -> use d_z = 1 - cosine_sim
        d_z = 1.0 - (z_embed @ z_embed.T)           # [B,B], cosine distance since z_embed normalized
        d_z = d_z + torch.eye(batch_size, device=device) * 1e9  # mask diagonal
        d_r = dr + torch.eye(batch_size, device=device) * 1e9   # risk distance, mask diagonal

        def _overlap_at_k(d1, d2, k=10):
            # d1/d2: [B,B], smaller = closer
            nn1 = torch.topk(d1, k=k, dim=1, largest=False).indices   # [B,k]
            nn2 = torch.topk(d2, k=k, dim=1, largest=False).indices   # [B,k]
            # compute overlap per row
            overlap = []
            for i in range(d1.shape[0]):
                a = set(nn1[i].tolist())
                b = set(nn2[i].tolist())
                overlap.append(len(a & b) / k)
            return float(np.mean(overlap))

        metrics["overlap@5"]  = _overlap_at_k(d_z, d_r, k=5)
        metrics["overlap@10"] = _overlap_at_k(d_z, d_r, k=10)

        # (C) distribution sharpness: entropy of p (lower -> more peaky / potential collapse)
        p_safe = p.clamp_min(1e-12)
        entropy = -(p_safe * torch.log(p_safe)).sum(dim=1).mean()
        metrics["p_entropy"] = float(entropy.item())

        # (D) optional: KL(t||p) as an alignment score (lower is better, equals loss up to constant)
        # since loss = CE(t,p), CE = H(t) + KL(t||p), KL is CE - H(t)
        t_safe = t.clamp_min(1e-12)
        Ht = -(t_safe * torch.log(t_safe)).sum(dim=1).mean()
        metrics["t_entropy"] = float(Ht.item())
        metrics["kl_t_p"] = float((loss - Ht).item())

        if return_metrics:
            return loss, metrics
        return loss



class TrajectoryBuffer:
    def __init__(self, max_size=100):
        self.buffer = []
        self.env_vectors = []

    def add_episode(self, traj, env_vector):
        """traj: list of (state, action), env_vector: np.array of shape (3,)"""
        if len(self.buffer) >= 100:
            self.buffer.pop(0)
            self.env_vectors.pop(0)
        self.buffer.append(traj)
        self.env_vectors.append(env_vector)

    def build_transformer_dataset(self, clip_len=50, min_clip=30):
        """Return: [100, 50, 9], mask [100, 50], env_target [100, 3]"""
        data, masks, targets = [], [], []

        for traj, env in zip(self.buffer, self.env_vectors):
            traj_len = len(traj)
            clip = np.random.randint(min_clip, clip_len+1)

            if traj_len >= clip:
                start = np.random.randint(0, traj_len - clip + 1)
                clipped = traj[start: start + clip]
            else:
                clipped = traj  # use all available

            # Pad to 50
            padded = clipped + [(np.zeros(8), 0)] * (clip_len - len(clipped))
            mask = [1] * len(clipped) + [0] * (clip_len - len(clipped))

            state_action = [np.concatenate([s, [a]]) for s, a in padded]
            data.append(state_action)
            masks.append(mask)
            targets.append(env)

        return (
            np.array(data, dtype=np.float32),      # [100, 50, 9]
            np.array(masks, dtype=np.float32),     # [100, 50]
            np.array(targets, dtype=np.float32)    # [100, 3]
        )

    
class agent_base():

    def __init__(self,parameters):
        """
        Initializes the agent class

        Keyword arguments:
        parameters -- dictionary with parameters for the agent

        There are two mandatory keys for the dictionary:
        - N_state (int): dimensionality of the (continuous) state space
        - N_actions (int): number of actions available to the agent

        All other arguments are optional, for a list see the class methods 
        get_default_parameters(self,parameters)
        set_parameters(self,parameters)

        """
        #
        parameters = self.make_dictionary_keys_lowercase(parameters)
        #
        # set parameters that are mandatory and can only be set at 
        # initializaton of a class instance
        self.set_initialization_parameters(parameters=parameters)
        #
        # get dictionary with default parameters
        default_parameters = self.get_default_parameters()
        # for all parameters not set by the input dictionary, add the 
        # respective default parameter
        parameters = self.merge_dictionaries(dict1=parameters,
                                             dict2=default_parameters)
        # set all parameters (except for those already set above in 
        # self.set_initialization_parameters())
        self.set_parameters(parameters=parameters)
        #
        # for future reference, each instance of a class carries a copy of 
        # the parameters as internal variable
        self.parameters = copy.deepcopy(parameters)
        #
        # intialize neural networks 
        self.initialize_neural_networks(neural_networks=\
                                            parameters['neural_networks'])
        # initialize the optimizer and loss function used for training
        self.initialize_optimizers(optimizers=parameters['optimizers'])
        self.initialize_losses(losses=parameters['losses'])
        #
        self.in_training = False

    def make_dictionary_keys_lowercase(self,dictionary):
        output_dictionary = {}
        for key, value in dictionary.items():
            output_dictionary[key.lower()] = value
        return output_dictionary

    def merge_dictionaries(self,dict1,dict2):
        '''
        Merge two dictionaries and return the merged dictionary

        If a key "key" exists in both dict1 and dict2, then the value from
        dict1 is used for the returned dictionary
        '''
        #
        return_dict = copy.deepcopy(dict1)
        #
        dict1_keys = return_dict.keys()
        for key, value in dict2.items():
            # we just add those entries from dict2 to dict1
            # that do not already exist in dict1
            if key not in dict1_keys:
                return_dict[key] = value
        #
        return return_dict

    def get_default_parameters(self):
        '''
        Create and return dictionary with the default parameters of the class
        '''
        #
        parameters = {
            'neural_networks':
                {
                'policy_net':{
                    # 'layers':[self.n_state,128,32,self.n_actions],
                    'custom_model': TwoBranchDQN(state_dim=8, env_dim=3, context_dim=3, n_actions=4)
                            }
                },
            'optimizers':
                {
                'policy_net':{
                    'optimizer':'RMSprop',
                     'optimizer_args':{'lr':1e-3}, # learning rate
                            }
                },
            'losses':
                {
                'policy_net':{            
                    'loss':'MSELoss',
                }
                },
            #
            'n_memory':200000,
            'training_stride':5,
            'batch_size':32,
            'saving_stride':100,
            #
            'n_episodes_max':10000,
            'n_solving_episodes':20,
            'solving_threshold_min':200,
            'solving_threshold_mean':230,
            #
            'discount_factor':0.99,
            }
        #
        # in case at some point the above dictionary is edited and an upper
        # case key is added:
        parameters = self.make_dictionary_keys_lowercase(parameters)
        #
        return parameters


    def set_initialization_parameters(self,parameters):
        '''Set those class parameters that are required at initialization'''
        #
        try: # set mandatory parameter N_state
            self.n_state = parameters['n_state']
        except KeyError:
            raise RuntimeError("Parameter N_state (= # of input"\
                         +" nodes for neural network) needs to be supplied.")
        #
        try: # set mandatory parameter N_actions
            self.n_actions = parameters['n_actions']
        except KeyError:
            raise RuntimeError("Parameter N_actions (= # of output"\
                         +" nodes for neural network) needs to be supplied.")

    def set_parameters(self,parameters):
        """Set training parameters"""
        #
        parameters = self.make_dictionary_keys_lowercase(parameters)
        #
        ########################################
        # Discount factor for Bellman equation #
        ########################################
        try: # 
            self.discount_factor = parameters['discount_factor']
        except KeyError:
            pass
        #
        #################################
        # Experience replay memory size #
        #################################
        try: # 
            self.n_memory = int(parameters['n_memory'])
            self.memory = memory(self.n_memory)
        except KeyError:
            pass
        #
        ###############################
        # Parameters for optimization #
        ###############################
        try: # number of simulation timesteps between optimization steps
            self.training_stride = parameters['training_stride']
        except KeyError:
            pass
        #
        try: # size of mini-batch for each optimization step
            self.batch_size = int(parameters['batch_size'])
        except KeyError:
            pass
        #
        try: # IO during training: every saving_stride episodes, the 
            # current status of the training is saved to disk
            self.saving_stride = parameters['saving_stride']
        except KeyError:
            pass
        #
        ##############################################
        # Parameters for training stopping criterion #
        ##############################################
        try: # maximal number of episodes until the training is stopped 
            # (if stopping criterion is not met before)
            self.n_episodes_max = parameters['n_episodes_max']
        except KeyError:
            pass
        #
        try: # # of the last N_solving episodes that need to fulfill the
            # stopping criterion for minimal and mean episode return
            self.n_solving_episodes = parameters['n_solving_episodes']
        except KeyError:
            pass
        #
        try: # minimal return over last N_solving_episodes
            self.solving_threshold_min = parameters['solving_threshold_min']
        except KeyError:
            pass
        #
        try: # mean return over last N_solving_episodes
            self.solving_threshold_mean = parameters['solving_threshold_mean']
        except KeyError:
            pass
        #

    def get_parameters(self):
        """Return dictionary with parameters of the current agent instance"""

        return self.parameters

    def initialize_neural_networks(self,neural_networks):
        """Initialize all neural networks"""

        self.neural_networks = {}
        for key, value in neural_networks.items():
            if 'custom_model' in value:
                self.neural_networks[key] = value['custom_model'].to(device)
            else:
                self.neural_networks[key] = neural_network(value['layers']).to(device)
        
    def initialize_optimizers(self,optimizers):
        """Initialize optimizers"""

        self.optimizers = {}
        for key, value in optimizers.items():
            self.optimizers[key] = torch.optim.RMSprop(
                        self.neural_networks[key].parameters(),
                            **value['optimizer_args'])
    
    def initialize_losses(self,losses):
        """Instantiate loss functions"""

        self.losses = {}
        for key, value in losses.items():
            self.losses[key] = nn.MSELoss()

    def get_number_of_model_parameters(self,name='policy_net'): 
        """Return the number of trainable neural network parameters"""
        # from https://stackoverflow.com/a/49201237
        return sum(p.numel() for p in self.neural_networks[name].parameters() \
                                    if p.requires_grad)


    def get_state(self):
        '''Return dictionary with current state of neural net and optimizer'''
        #
        state = {'parameters':self.get_parameters()}
        #
        for name,neural_network in self.neural_networks.items():
            state[name] = copy.deepcopy(neural_network.state_dict())
        #
        for name,optimizer in (self.optimizers).items():
            #
            state[name+'_optimizer'] = copy.deepcopy(optimizer.state_dict())
        #
        return state
    

    def load_state(self,state):
        '''
        Load given states for neural networks and optimizer

        The argument "state" has to be a dictionary with the following 
        (key, value) pairs:

        1. state['parameters'] = dictionary with the agents parameters
        2. For every neural network, there should be a state dictionary:
            state['$name'] = state dictionary of neural_network['$name']
        3. For every optimizer, there should be a state dictionary:
            state['$name_optimizer'] = state dictionary of optimizers['$name']
        '''
        #
        parameters=state['parameters']
        #
        self.check_parameter_dictionary_compatibility(parameters=parameters)
        #
        self.__init__(parameters=parameters)
        #
        #
        for name,state_dict in (state).items():
            if name == 'parameters':
                continue
            elif 'optimizer' in name:
                name = name.replace('_optimizer','')
                self.optimizers[name].load_state_dict(state_dict)
            else:
                self.neural_networks[name].load_state_dict(state_dict)
        #


    def check_parameter_dictionary_compatibility(self,parameters):
        """Check compatibility of provided parameter dictionary with class"""

        error_string = ("Error loading state. Provided parameter {0} = {1} ",
                    "is inconsistent with agent class parameter {0} = {2}. ",
                    "Please instantiate a new agent class with parameters",
                    " matching those of the model you would like to load.")
        try: 
            n_state =  parameters['n_state']
            if n_state != self.n_state:
                raise RuntimeError(error_string.format('n_state',n_state,
                                                self.n_state))
        except KeyError:
            pass
        #
        try: 
            n_actions =  parameters['n_actions']
            if n_actions != self.n_actions:
                raise RuntimeError(error_string.format('n_actions',n_actions,
                                                self.n_actions))
        except KeyError:
            pass


    def evaluate_stopping_criterion(self,list_of_returns):
        """ Evaluate stopping criterion """
        # if we have run at least self.N_solving_episodes, check
        # whether the stopping criterion is met
        if len(list_of_returns) < self.n_solving_episodes:
            return False, 0., 0.
        #
        # get numpy array with recent returns
        recent_returns = np.array(list_of_returns)
        recent_returns = recent_returns[-self.n_solving_episodes:]
        #
        # calculate minimal and mean return over the last
        # self.n_solving_episodes epsiodes 
        minimal_return = np.min(recent_returns)
        mean_return = np.mean(recent_returns)
        #
        # check whether stopping criterion is met
        if minimal_return > self.solving_threshold_min:
            if mean_return > self.solving_threshold_mean:
                return True, minimal_return, mean_return
        # if stopping crtierion is not met:
        return False, minimal_return, mean_return


    def act(self,state):
        """
        Select an action for the current state
        """
        #
        # This typically uses the policy net. See the child classes below
        # for examples:
        # - dqn: makes decisions using an epsilon-greedy algorithm
        # - actor_critic: draws a stochastic decision with probabilities given
        #                 by the current stochastic policy
        #
        # As an example, we here draw a fully random action:
        return np.random.randint(self.n_actions) 


    def add_memory(self,memory):
        """Add current experience tuple to the memory"""
        self.memory.push(*memory)

    def get_samples_from_memory(self):
        '''
        Get a tuple (states, actions, next_states, rewards, episode_end? ) 
        from the memory, as appopriate for experience replay
        '''
        #
        # get random sample of transitions from memory
        current_transitions = self.memory.sample(batch_size=self.batch_size)
        #
        # convert list of Transition elements to Transition element with lists
        # (see https://stackoverflow.com/a/19343/3343043)
        batch = Transition(*zip(*current_transitions))
        #
        # convert lists of current transitions to torch tensors
        state_batch = torch.cat( [s.unsqueeze(0) for s in batch.state],
                                        dim=0).to(device)
        # state_batch.shape = [batch_size, N_states]
        next_state_batch = torch.cat(
                         [s.unsqueeze(0) for s in batch.next_state],dim=0)
        action_batch = torch.cat(batch.action).to(device)
        # action_batch.shape = [batch_size]
        reward_batch = torch.cat(batch.reward).to(device)
        done_batch = torch.tensor(batch.done).float().to(device)
        #
        return state_batch, action_batch, next_state_batch, \
                        reward_batch, done_batch

          

    def train(self,environment,
                    verbose=True,
                    model_filename=None,
                    training_filename=None,
                    track_embedder=None
                ):
        """
        Train the agent on a provided environment

        Keyword arguments:
        environment -- environment used by the agent to train. This should be
                       an instance of a class with methods "reset" and "step".
                       - environment.reset() should reset the environment to
                         an initial state and return a tuple,
                            current_state, info = environment.reset(),
                         such current_state is an initial state of the with
                         np.shape(current_state) = (self.N_state,)
                       - environment.set(action) should take an integer in 
                         {0, ..., self.N_action-1} and return a tuple, 
                            s, r, te, tr, info = environment.step(action),
                         where s is the next state with shape (self.N_state,),
                         r is the current reward (a float), and where te and
                         tr are two Booleans that tell us whether the episode
                         has terminated (te == True) or has been truncated 
                         (tr == True)
        verbose (Bool) -- Print progress of training to terminal. Defaults to
                          True
        model_filename (string) -- Output filename for final trained model and
                                   periodic snapshots of the model during 
                                   training. Defaults to None, in which case
                                   nothing is not written to disk
        training_filename (string) -- Output filename for training data, 
                                      namely lists of episode durations, 
                                      episode returns, number of training 
                                      epochs, and total number of steps 
                                      simulated. Defaults to None, in which 
                                      case no training data is written to disk
        """
        self.in_training = True
        #
        training_complete = False
        step_counter = 0 # total number of simulated environment steps
        epoch_counter = 0 # number of training epochs 
        #
        # lists for documenting the training
        model = TSTransformerWithMask().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        init_model_path = f"./model_file/FactualEncoder.pt"
        if not os.path.exists(init_model_path):
            torch.save(model.state_dict(), init_model_path)
        # torch.save(model.state_dict(), f"./model_file/model_LL_epoch_bothtrain.pt")
        # track_embedder = TrackEmbeddingWrapper(f"./model_file/model_LL_epoch_bothtrain.pt")
        track_embedder = TrackEmbeddingWrapper(init_model_path, device=device)

        # traj_buffer = TrajectoryBuffer(max_size=200)
        best_mean = float("inf")

        val_ap_list = []
        val_an_list = []
        os.makedirs("logs", exist_ok=True)
        episode_durations = [] # duration of each training episodes
        episode_returns = [] # return of each training episode
        steps_simulated = [] # total number of steps simulated at the end of
                             # each training episode
        training_epochs = [] # total number of training epochs at the end of 
                             # each training episode
        successful_episodes = [] # counter for number of successful episodes                
        #
        output_state_dicts = {} # dictionary in which we will save the status
                                # of the neural networks and optimizer
                                # every self.saving_stride steps epochs during
                                # training. 
                                # We also store the final neural network
                                # resulting from our training in this 
                                # dictionary
        context_state_dicts = {} 
        context_logs = []                      
        # ========== Context belief modules (NEW) ==========
        self.phi_dim = 2                      # 你的人格向量Dimension（先默认2）
        self.T_context = 5                   # context窗口长度（先默认5）
        self.min_valid = 5                    # 前min_valid步Output默认belief
        warmup_episodes = 1000              # 前warmup_episodes只在不变化Environment里Trainingtrack_embedder
        #
        
        self.context_net = ContextBeliefNet(
            e_dim=3,
            phi_dim=self.phi_dim,
            hidden_dim=64,
            num_layers=1,
            dropout=0.0,
        ).to(device)

        # online tracker（维护e/r序列并Outputz_t）
        self.context_tracker = ContextBeliefTracker(
            context_net=self.context_net,
            T_context=self.T_context,
            e_dim=3,
            phi_dim=self.phi_dim,
            min_valid=self.min_valid,
            device=device,
            dtype=torch.float32,
        )
        self.context_optimizer = torch.optim.Adam(
            self.context_tracker.parameters(),
            lr=1e-3
        )
        self.phi = torch.ones(self.phi_dim, device=device, dtype=torch.float32)


        #
        if verbose:
            training_progress_header = (
                "| episode | return          | minimal return    "
                    "  | mean return        |\n"
                "|         | (this episode)  | (last {0} episodes)  "
                    "| (last {0} episodes) |\n"
                "|---------------------------------------------------"
                    "--------------------")
            print(training_progress_header.format(self.n_solving_episodes))
            #
            status_progress_string = ( # for outputting status during training
                        "| {0: 7d} |   {1: 10.3f}    |     "
                        "{2: 10.3f}      |    {3: 10.3f}      |")
        #
        gravity_range = np.linspace(-11.99, -6.665, 4)
        wind_power_range = np.linspace(0.01, 19.99, 5)
        turbulence_power_range = np.linspace(0.01, 1.99, 5)

        param_combinations = np.array(np.meshgrid(gravity_range, wind_power_range, turbulence_power_range)).T.reshape(-1, 3)
        
        for n_episode in range(self.n_episodes_max):
            #
            # reset environment and reward of current episode
            
            gravity, wind_power, turbulence_power = random.choice(param_combinations)
            
            environment = gym.make(
                "LunarLander-v3",
                gravity=gravity,
                enable_wind=True,  # 开启风
                wind_power=wind_power,
                turbulence_power=turbulence_power,
            )
            
            state, info = environment.reset()
            track_embedder.reset()
            self.context_tracker.reset()

            self.phi = sample_phi(mode="neutral", device=self.device, dtype=self.dtype)

            # --- Construct 3D environment vectors e ---
            def normalize(value, vmin, vmax):
                return (value - vmin) / (vmax - vmin)

            env_vector = np.array([
                normalize(gravity, np.min(gravity_range), np.max(gravity_range)),
                normalize(wind_power, np.min(wind_power_range), np.max(wind_power_range)),
                normalize(turbulence_power, np.min(turbulence_power_range), np.max(turbulence_power_range)),
            ], dtype=np.float32)

            state_tensor = torch.tensor(state[:8], dtype=torch.float32, device=device)
            zero_tensor = torch.zeros(3, dtype=torch.float32, device=device)
            z_t = self.context_tracker.default_z.clone()
            aug_state = torch.cat([state_tensor, zero_tensor, z_t]).detach()
            action = self.act(state=aug_state)
            track_embedder.update(state, action)
            with torch.no_grad():
                    e_vector = track_embedder.get_embedding()
            
            # state = np.concatenate((state, e))         
            
            current_total_reward = 0.
            reward = 0.
            #
            episode_trajectory = []  # episode (state, action)
            for i in itertools.count(): # timesteps of environment
                #
                if i % 20 == 0 and n_episode >= warmup_episodes:
                    gravity, wind_power, turbulence_power = random.choice(param_combinations)

                    u = environment.unwrapped

                    # modify wind / turbulence
                    if hasattr(u, "wind_power"):
                        u.wind_power = wind_power
                    if hasattr(u, "turbulence_power"):
                        u.turbulence_power = turbulence_power

                    # modify gravity (Box2D)
                    if hasattr(u, "gravity"):
                        u.gravity = gravity

                    # record shift (for ContextNet semi-supervised use)
                    is_shift = 1
                else:
                    is_shift = 0
                info["is_shift"] = is_shift
                info["env_params"] = (gravity, wind_power, turbulence_power)
                
                # aug_state = np.concatenate([state[:8], e_vector])
                state_tensor = torch.tensor(state[:8], dtype=torch.float32).to(device)
                aug_state = torch.cat([state_tensor, e_vector, z_t]).detach()  
                action = self.act(state=aug_state)
                episode_trajectory.append((state[:8].copy(), action))
                #
                # perform action
                next_state, reward, terminated, truncated, info = \
                                        environment.step(action)

                track_embedder.update(next_state, action)
                with torch.no_grad():
                    e_vector = track_embedder.get_embedding()  # (3,)
                
                z_t = self.context_tracker.update(e_vector, reward, self.phi)
                self.context_tracker.push_shift(is_shift)
                self.context_tracker.push_window_to_bank(self.phi)
                
                # next_state = np.concatenate((next_state, e_vector))
                #
                step_counter += 1 # increase total steps simulated
                done = terminated or truncated # did the episode end?
                current_total_reward += reward # add current reward to total
                #
                # store the transition in memory
                reward = torch.tensor([np.float32(reward)], device=device)
                action = torch.tensor([action], device=device)

                next_aug_state = torch.cat([
                        torch.tensor(next_state[:8], dtype=torch.float32, device=device),
                        e_vector.clone().detach(), z_t.detach()
                    ]).detach()
                self.add_memory([aug_state,
                                 action,
                                 next_aug_state,
                                 reward,
                                 done])


                #
                state = next_state
                #
                if step_counter % self.training_stride == 0:
                    # train model
                    self.run_optimization_step(epoch=epoch_counter) # optimize
                    epoch_counter += 1 # increase count of optimization steps
                    out = self.context_tracker.compute_contrastive_loss(return_metrics=True)
                    if out is not None:
                        loss_ctx, metrics_ctx = out
                        self.context_optimizer.zero_grad()
                        loss_ctx.backward()
                        self.context_optimizer.step()
                        context_logs.append({
                            "loss_ctx": float(loss_ctx.item()),
                            "risk_mean": metrics_ctx["risk_mean"],
                            "risk_std": metrics_ctx["risk_std"],
                            "overlap@10": metrics_ctx["overlap@10"],
                            "p_entropy": metrics_ctx["p_entropy"],
                            "kl_t_p": metrics_ctx["kl_t_p"],
                        })
                #
                if done: # if current episode ended
                    #
                    # update training statistics
                    episode_durations.append(i + 1)
                    episode_returns.append(current_total_reward)
                    steps_simulated.append(step_counter)
                    training_epochs.append(epoch_counter)
                    if terminated and reward > 0:
                        safe_landing = 1
                    else:
                        safe_landing = 0
                    successful_episodes.append(safe_landing)
                    # traj_buffer.add_episode(episode_trajectory, env_vector)
                    #
                    # check whether the stopping criterion is met
                    training_complete, min_ret, mean_ret = \
                            self.evaluate_stopping_criterion(\
                                list_of_returns=episode_returns)
                    if verbose:
                            # print training stats
                            if n_episode % 100 == 0 and n_episode > 0:
                                end='\n'
                            else:
                                end='\r'
                            if min_ret > self.solving_threshold_min:
                                if mean_ret > self.solving_threshold_mean:
                                    end='\n'
                            #
                            print(status_progress_string.format(n_episode,
                                    current_total_reward,
                                   min_ret,mean_ret),
                                        end=end)
                    break
            #
            # Save model and training stats to disk
            if (n_episode % self.saving_stride == 0) \
                    or training_complete \
                    or n_episode == self.n_episodes_max-1:
                #
                # print("bank size =", len(self.context_tracker.window_bank))
                # print("loss_ctx is None (warmup / no positives)") if loss_ctx is None else f"loss_ctx = {loss_ctx.item():.4f}"
                # print("z_t =", z_t.detach().cpu().numpy(), "sum=", float(z_t.sum()))
                if model_filename != None:
                    output_state_dicts[n_episode] = self.get_state()
                    torch.save(output_state_dicts, model_filename)
                    context_state_dicts[n_episode] = copy.deepcopy(self.context_tracker.state_dict())
                    context_ckpt_path = model_filename.replace(".pth", "_context_tracker.pth")
                    torch.save(context_state_dicts, context_ckpt_path)

                #
                training_results = {'episode_durations':episode_durations,
                            'epsiode_returns':episode_returns,
                            'n_training_epochs':training_epochs,
                            'n_steps_simulated':steps_simulated,
                            'training_completed':False,
                            }
                if training_filename != None:
                    self.save_dictionary(dictionary=training_results,
                                        filename=training_filename)
            #
            # train Transformer model on collected trajectories
            if n_episode!= 0 and n_episode % 200 == 0:
                model = TSTransformerWithMask().to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                newdataset, mask, real_envs = traj_buffer.build_transformer_dataset()
                real_envs = np.array(real_envs)
                for epoch in range(300):
                    train_epoch_efficient_online(model, newdataset, mask, real_envs, optimizer, batch_size=32)
                    avg_loss, avg_ap, avg_an = validate_epoch_efficient_online(model, newdataset, mask, real_envs, batch_size=32)
                    val_ap_list.append(avg_ap)
                    val_an_list.append(avg_an)
                save_validation_metrics_to_csv(val_ap_list, val_an_list, stride=1, filename=f"validation_metrics_epoch_{n_episode}_Encoder.csv")
                torch.save(model.state_dict(), f"./model_file/FactualEncoder.pt")
                torch.save(optimizer.state_dict(), f"./model_file/optimizer_FactualEncoder.pt")
                track_embedder.model.load_state_dict(torch.load("./model_file/FactualEncoder.pt"))
                
                
            if n_episode!= 0 and n_episode % 1000 == 0:
                save_episode_stats_to_csv(
                    episode_returns,
                    episode_durations,
                    successful_episodes,
                    filename=f"episode_stats_epoch_{n_episode}.csv"
                )
                save_context_metrics_to_csv(context_logs, filename=f"contrastive_metrics_epoch_{n_episode}.csv")

                # save_validation_metrics_to_csv(val_ap_list, val_an_list, stride=200, filename="validation_metrics_ourbothtrain.csv")
                # # n_save = 2000 + n_episode
                # n_save = n_episode
                # torch.save(model.state_dict(), f"./model_file/model_LL_epoch_{n_save}_ourbothtrain.pt")

            #
            if training_complete:
                # we stop if the stopping criterion was met at the end of
                # the current episode
                training_results['training_completed'] = True
                break

            try:
                environment.close()
            except Exception:
                pass
        #

        if not training_complete:
            # if we stopped the training because the maximal number of
            # episodes was reached, we throw a warning
            warning_string = ("Warning: Training was stopped because the "
            "maximum number of episodes, {0}, was reached. But the stopping "
            "criterion has not been met.")
            warnings.warn(warning_string.format(self.n_episodes_max))
        #
        self.in_training = False
        #
        return training_results

    def save_dictionary(self,dictionary,filename):
        """Save a dictionary in hdf5 format"""

        with h5py.File(filename, 'w') as hf:
            self.save_dictionary_recursively(h5file=hf,
                                            path='/',
                                            dictionary=dictionary)
                
    def save_dictionary_recursively(self,h5file,path,dictionary):
        #

        for key, value in dictionary.items():
            if isinstance(value, dict):
                self.save_dictionary_recursively(h5file, 
                                                path + str(key) + '/',
                                                value)
            else:
                h5file[path + str(key)] = value

    def load_dictionary(self,filename):
        with h5py.File(filename, 'r') as hf:
            return self.load_dictionary_recursively(h5file=hf,
                                                    path='/')

    def load_dictionary_recursively(self,h5file, path):

        return_dict = {}
        for key, value in h5file[path].items():
            if isinstance(value, h5py._hl.dataset.Dataset):
                return_dict[key] = value.value
            elif isinstance(value, h5py._hl.group.Group):
                return_dict[key] = self.load_dictionary_recursively(\
                                            h5file=h5file, 
                                            path=path + key + '/')
        return return_dict



class dqn(agent_base):

    def __init__(self,parameters):
        super().__init__(parameters=parameters)
        self.in_training = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32

    def get_default_parameters(self):
        '''
        Create and return dictionary with the default parameters of the dqn
        algorithm
        '''
        #
        default_parameters = super().get_default_parameters()
        #
        # add default parameters specific to the dqn algorithm
        default_parameters['neural_networks']['target_net'] = {}
        if 'layers' in default_parameters['neural_networks']['policy_net']:
            default_parameters['neural_networks']['target_net']['layers'] = \
                copy.deepcopy(default_parameters['neural_networks']['policy_net']['layers'])
        elif 'custom_model' in default_parameters['neural_networks']['policy_net']:
            default_parameters['neural_networks']['target_net']['custom_model'] = \
                copy.deepcopy(default_parameters['neural_networks']['policy_net']['custom_model'])

        #
        #
        # soft update stride for target net:
        default_parameters['target_net_update_stride'] = 1 
        # soft update parameter for target net:
        default_parameters['target_net_update_tau'] = 1e-2 
        #
        # Parameters for epsilon-greedy policy with epoch-dependent epsilon
        default_parameters['epsilon'] = 1.0 # initial value for epsilon
        default_parameters['epsilon_1'] = 0.1 # final value for epsilon
        default_parameters['d_epsilon'] = 0.00005 # decrease of epsilon
            # after each training epoch
        #
        default_parameters['doubledqn'] = False
        #
        return default_parameters


    def set_parameters(self,parameters):
        #
        super().set_parameters(parameters=parameters)
        #
        ##################################################
        # Use deep Q-learning or double deep Q-learning? #
        ##################################################
        try: # False -> use DQN; True -> use double DQN
            self.doubleDQN = parameters['doubledqn']
        except KeyError:
            pass
        #
        ##########################################
        # Parameters for updating the target net #
        ##########################################
        try: # after how many training epochs do we update the target net?
            self.target_net_update_stride = \
                                    parameters['target_net_update_stride']
        except KeyError:
            pass
        #
        try: # tau for soft update of target net (value 1 means hard update)
            self.target_net_update_tau = parameters['target_net_update_tau']
            # check if provided parameter is within bounds
            error_msg = ("Parameter 'target_net_update_tau' has to be "
                    "between 0 and 1, but value {0} has been passed.")
            error_msg = error_msg.format(self.target_net_update_tau)
            if self.target_net_update_tau < 0:
                raise RuntimeError(error_msg)
            elif self.target_net_update_tau > 1:
                raise RuntimeError(error_msg)
        except KeyError:
            pass
        #
        #
        ########################################
        # Parameters for epsilon-greedy policy #
        ########################################
        try: # probability for random action for epsilon-greedy policy
            self.epsilon = \
                    parameters['epsilon']
        except KeyError:
            pass
        #
        try: # final probability for random action during training 
            #  for epsilon-greedy policy
            self.epsilon_1 = \
                    parameters['epsilon_1']
        except KeyError:
            pass
        # 
        try: # amount by which epsilon decreases during each training epoch
            #  until the final value self.epsilon_1 is reached
            self.d_epsilon = \
                    parameters['d_epsilon']
        except KeyError:
            pass

    def act(self,state,epsilon=0.):
        """
        Use policy net to select an action for the current state
        
        We use an epsilon-greedy algorithm: 
        - With probability epsilon we take a random action (uniformly drawn
          from the finite number of available actions)
        - With probability 1-epsilon we take the optimal action (as predicted
          by the policy net)

        By default epsilon = 0, which means that we actually use the greedy 
        algorithm for action selection
        """
        #
        if self.in_training:
            epsilon = self.epsilon

        if torch.rand(1).item() > epsilon:
            # 
            policy_net = self.neural_networks['policy_net']
            #
            with torch.no_grad():
                policy_net.eval()
                # action = policy_net(torch.tensor(state).unsqueeze(0)).argmax(1).item()
                # action = policy_net(torch.tensor(state).unsqueeze(0).float()).argmax(1).item()
                state_tensor = state.clone().detach().unsqueeze(0).to(device)
                action = policy_net(state_tensor).argmax(1).item()

                # action = policy_net(torch.tensor(state)).argmax(0).item()
                policy_net.train()
                return action
        else:
            # perform random action
            return torch.randint(low=0,high=self.n_actions,size=(1,)).item()
        
    def update_epsilon(self):
        """
        Update epsilon for epsilon-greedy algorithm
        
        For training we assume that 
        epsilon(n) = max{ epsilon_0 - d_epsilon * n ,  epsilon_1 },
        where n is the number of training epochs.

        For epsilon_0 > epsilon_1 the function epsilon(n) is piecewise linear.
        It first decreases from epsilon_0 to epsilon_1 with a slope d_epsilon,
        and then becomes constant at the value epsilon_1.
        
        This ensures that during the initial phase of training the neural 
        network explores more randomly, and in later stages of the training
        follows more the policy learned by the neural net.
        """
        self.epsilon = max(self.epsilon - self.d_epsilon, self.epsilon_1)

    def run_optimization_step(self,epoch):
        """Run one optimization step for the policy net"""
        #
        # if we have less sample transitions than we would draw in an 
        # optimization step, we do nothing
        if len(self.memory) < self.batch_size:
            return
        #
        state_batch, action_batch, next_state_batch, \
                        reward_batch, done_batch = self.get_samples_from_memory()
        #
        policy_net = self.neural_networks['policy_net']
        target_net = self.neural_networks['target_net']
        #
        optimizer = self.optimizers['policy_net']
        loss = self.losses['policy_net']
        #
        policy_net.train() # turn on training mode
        #
        # Evaluate left-hand side of the Bellman equation using policy net
        LHS = policy_net(state_batch.to(device)).gather(dim=1,
                                 index=action_batch.unsqueeze(1))
        # LHS.shape = [batch_size, 1]
        #
        # Evaluate right-hand side of Bellman equation
        if self.doubleDQN:
            #
            # in double deep Q-learning, we use the policy net for choosing
            # the action on the right-hand side of the Bellman equation. We 
            # then use the target net to evaluate the Q-function on the 
            # chosen action
            argmax_next_state = policy_net(next_state_batch).argmax(
                                                                    dim=1)
            # argmax_next_state.shape = [batch_size]
            #
            Q_next_state = target_net(next_state_batch).gather(
                dim=1,index=argmax_next_state.unsqueeze(1)).squeeze(1)
            # shapes of the various tensor appearing in the previous line:
            # self.target_net(next_state_batch).shape = [batch_size,N_actions]
            # self.target_net(next_state_batch).gather(dim=1,
            #   index=argmax_next_state.unsqueeze(1)).shape = [batch_size, 1]
            # Q_next_state.shape = [batch_size]
        else:
            # in deep Q-learning, we use the target net both for choosing
            # the action on the right-hand side of the Bellman equation, and 
            # for evaluating the Q-function on that action
            Q_next_state = target_net(next_state_batch\
                                                ).max(1)[0].detach()
            # Q_next_state.shape = [batch_size]
        RHS = Q_next_state * self.discount_factor * (1.-done_batch) \
                            + reward_batch
        RHS = RHS.unsqueeze(1) # RHS.shape = [batch_size, 1]
        #
        # optimize the model
        loss_ = loss(LHS, RHS)
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
        #
        policy_net.eval() # turn off training mode
        #
        self.update_epsilon() # for epsilon-greedy algorithm
        #
        if epoch % self.target_net_update_stride == 0:
            self.soft_update_target_net() # soft update target net
        #
        
    def soft_update_target_net(self):
        """Soft update parameters of target net"""
        #
        params1 = self.neural_networks['policy_net'].named_parameters()
        params2 = self.neural_networks['target_net'].named_parameters()

        dict_params2 = dict(params2)

        for name1, param1 in params1:
            if name1 in dict_params2:
                dict_params2[name1].data.copy_(\
                    self.target_net_update_tau*param1.data\
                + (1-self.target_net_update_tau)*dict_params2[name1].data)
        self.neural_networks['target_net'].load_state_dict(dict_params2)
