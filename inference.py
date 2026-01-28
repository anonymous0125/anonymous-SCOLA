import os
import csv
import gymnasium as gym
import numpy as np
import torch
from tqdm import tqdm

# === import your classes ===
from agent_class import (
    dqn,
    TwoBranchDQN,              # only needed if your ckpt params require it
    TrackEmbeddingWrapper,
    ContextBeliefTracker,
    ContextBeliefNet
)

# ============================
# Config
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DQN_CKPT_PATH = "./trained_agents/ContextualNet.pth"
CTX_CKPT_PATH = "./trained_agents/dContextualNet_tracker.pth"
EMBEDDER_PATH = "./model_file/FactualEncoder.pt"

episode_to_use = 3000  # choose which saved episode checkpoint to evaluate

# evaluation protocol
EPISODES_PER_ENV = 100         # 100 envs * 100 = 1000 episodes
EP_STEPS = 120                # each episode runs exactly 120 steps
SWITCH_EVERY = 20             # change env every 20 steps
MODES = ["mild", "severe"]    # run both

# context window length (must match how tracker expects in your build)
# If your tracker in training used T_context=5, keep it 5 here.
T_CONTEXT = 5
MIN_VALID = 5

# phi (safe)
# Adjust if your phi meaning is reversed; here we assume phi=[0,1] = safe
PHI_SAFE = torch.tensor([0.1, 0.9], device=device, dtype=torch.float32)
PHI_NEUTRAL = torch.tensor([0.5, 0.5], device=device)
PHI_RISKY   = torch.tensor([0.9, 0.1], device=device)

# output
OUT_CSV = "./logs/Logs.csv"


# ============================
# Utils
# ============================
def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def append_row_csv(filepath, header, row):
    ensure_dir(filepath)
    write_header = (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0)
    with open(filepath, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)

def set_eval_mode(agent, context_tracker):
    # DQN
    agent.neural_networks["policy_net"].eval()
    agent.neural_networks["target_net"].eval()

    # context
    context_tracker.eval()
    if hasattr(context_tracker, "context_net"):
        context_tracker.context_net.eval()

def disable_epsilon(agent):
    if hasattr(agent, "epsilon"):
        agent.epsilon = 0.0
    if hasattr(agent, "eps"):
        agent.eps = 0.0

def apply_env_params(env, gravity, wind, turbulence):
    """
    Best-effort in-place parameter change for LunarLander.
    Some gym versions differ; this tries common attributes.
    """
    u = env.unwrapped

    # gravity
    try:
        if hasattr(u, "world") and u.world is not None:
            # Box2D world gravity is a vector (x,y)
            u.world.gravity = (0.0, float(gravity))
        if hasattr(u, "gravity"):
            u.gravity = float(gravity)
    except Exception:
        pass

    # wind / turbulence
    for name, val in [("wind_power", wind), ("turbulence_power", turbulence)]:
        try:
            if hasattr(u, name):
                setattr(u, name, float(val))
        except Exception:
            pass

def step_index_move(idx, n, delta):
    """move index by ±delta with clipping"""
    return int(np.clip(idx + delta, 0, n - 1))

def random_move(delta):
    return int(np.random.choice([-delta, delta]))

def env_id_to_indices(env_id, n_g=4, n_w=5, n_t=5):
    """
    Map env_id in [0, 99] to (gi, wi, ti) consistent with meshgrid flattening.
    We will define ordering as: gi major -> wi -> ti (or adjust if you want).
    Here: env_id = gi*(n_w*n_t) + wi*n_t + ti
    """
    gi = env_id // (n_w * n_t)
    rem = env_id % (n_w * n_t)
    wi = rem // n_t
    ti = rem % n_t
    return gi, wi, ti


# ============================
# Build env grid (100 envs)
# ============================
gravity_range = np.linspace(-11.99, -6.665, 4)
wind_power_range = np.linspace(0.01, 19.99, 5)
turbulence_power_range = np.linspace(0.01, 1.99, 5)

N_G, N_W, N_T = len(gravity_range), len(wind_power_range), len(turbulence_power_range)
assert N_G * N_W * N_T == 100, "Grid should be 100 environments"


# ============================
# Load agent checkpoint
# ============================
saved_models = torch.load(DQN_CKPT_PATH, map_location="cpu", weights_only=False)
ckpt = saved_models[episode_to_use]

# Initialize agent from ckpt parameters to ensure consistency
agent = dqn(parameters=ckpt["parameters"])
agent.load_state(ckpt)

# Frozen embedder
track_embedder = TrackEmbeddingWrapper(EMBEDDER_PATH, device=device)

# Build context_net + tracker, then load tracker state
context_net = ContextBeliefNet(e_dim=3, phi_dim=2, hidden_dim=64).to(device)
context_tracker = ContextBeliefTracker(
    context_net=context_net,
    T_context=T_CONTEXT,
    e_dim=3,
    phi_dim=2,
    min_valid=MIN_VALID,
    device=device,
    dtype=torch.float32
)

ctx_ckpts = torch.load(CTX_CKPT_PATH, map_location=device)
context_tracker.load_state_dict(ctx_ckpts[episode_to_use])

# IMPORTANT: add get_current_embedding if you haven't added it in codebase
# (If you already added it, you can delete this block.)
if not hasattr(context_tracker, "get_current_embedding"):
    @torch.no_grad()
    def get_current_embedding(phi, normalize=True):
        valid_len = len(context_tracker.e_hist)
        if valid_len < context_tracker.min_valid:
            return None

        # 1) build window e_seq, r_seq from tracker buffers (avoid phi shape bugs)
        T = context_tracker.T
        device_ = context_tracker.device

        e_list = list(context_tracker.e_hist)[-T:]
        r_list = list(context_tracker.r_hist)[-T:]

        e_seq = torch.stack(e_list, dim=0).to(device_).float()           # [T,3]
        r_seq = torch.stack([x.view(()) for x in r_list], dim=0).to(device_).float()  # [T]

        e_seq = e_seq.unsqueeze(0)                                       # [1,T,3]
        r_seq = r_seq.unsqueeze(0)                                       # [1,T]

        # 2) force phi to shape [1,2]
        phi_t = torch.as_tensor(phi, device=device_, dtype=torch.float32)
        phi_t = phi_t.view(-1)                                           # [2]
        phi_t = phi_t.unsqueeze(0)                                       # [1,2]

        # 3) embedding
        z = context_tracker.context_net.get_embedding(e_seq, r_seq, phi_t)  # [1,D]
        if normalize:
            z = torch.nn.functional.normalize(z, dim=-1)
        return z.squeeze(0).detach()

    context_tracker.get_current_embedding = get_current_embedding

# eval mode
set_eval_mode(agent, context_tracker)
disable_epsilon(agent)


# ============================
# Logging
# ============================
header = [
    "mode",
    "env_id",
    "trial",
    "t",
    "g", "w", "turb",
    "shift_flag",
    "rt",
    "ct_norm", "dct_norm",
    "et_norm", "det_norm",
]

# reset output file
ensure_dir(OUT_CSV)
if os.path.exists(OUT_CSV):
    os.remove(OUT_CSV)


# ============================
# Main evaluation
# ============================
def run_one_episode(env, mode, env_id, trial, C_DIM):
    """
    Run exactly EP_STEPS steps. If env terminates early, immediately reset and continue
    to reach exactly EP_STEPS (to keep alignment around switch points).
    """
    # start indices fixed by env_id
    gi0, wi0, ti0 = env_id_to_indices(env_id, N_G, N_W, N_T)
    gi, wi, ti = gi0, wi0, ti0

    gravity = float(gravity_range[gi])
    wind = float(wind_power_range[wi])
    turb = float(turbulence_power_range[ti])

    # reset env and apply initial params
    apply_env_params(env, gravity, wind, turb)
    state, _ = env.reset()
    apply_env_params(env, gravity, wind, turb)  # apply again after reset, more robust
    # C_DIM = context_tracker.context_net.head.out_features  # embedding dim, e.g., 16/32
    # infer embedding dim robustly (works for Linear or Sequential head)
    
    c_prev = None
    e_prev = None

    # reset modules
    track_embedder.reset()
    context_tracker.reset()

    reward_prev = 0.0
    e_vector = torch.zeros(3, device=device, dtype=torch.float32)

    c_prev = None
    e_prev = None

    for t in range(EP_STEPS):

        # switch parameters every 20 steps (t=20,40,60,...)
        shift_flag = 0
        if (t > 0) and (t % SWITCH_EVERY == 0):
            shift_flag = 1
            if mode == "mild":
                which = np.random.randint(0, 3)
                step = random_move(1)
                if which == 0:
                    gi = step_index_move(gi, N_G, step)
                elif which == 1:
                    wi = step_index_move(wi, N_W, step)
                else:
                    ti = step_index_move(ti, N_T, step)
            elif mode == "severe":
                # all three params move by 2
                gi = step_index_move(gi, N_G, random_move(2))
                wi = step_index_move(wi, N_W, random_move(2))
                ti = step_index_move(ti, N_T, random_move(2))
            else:
                raise ValueError(f"Unknown mode: {mode}")

            gravity = float(gravity_range[gi])
            wind = float(wind_power_range[wi])
            turb = float(turbulence_power_range[ti])
            apply_env_params(env, gravity, wind, turb)

        # context update (belief)
        z_t = context_tracker.update(e_vector, reward_prev, PHI_RISKY)  # [3]

        # context embedding (c_t)
        c_t = context_tracker.get_current_embedding(PHI_RISKY)
        # infer embedding dim robustly (works for Linear or Sequential head)
        if c_t is None:
            c_t = torch.zeros(C_DIM, device=device, dtype=torch.float32)

        # DQN input: [state(8), env(3), z(3)] -> 14
        state_tensor = torch.tensor(state[:8], dtype=torch.float32, device=device)
        env_tensor = torch.tensor([gravity, wind, turb], dtype=torch.float32, device=device)
        aug_state = torch.cat([state_tensor, env_tensor, z_t], dim=0)

        # act
        action = agent.act(state=aug_state)

        # step
        next_state, reward, terminated, truncated, _ = env.step(action)

        # update embedder -> next e_vector
        track_embedder.update(next_state, action)
        e_vector = track_embedder.get_embedding()
        if isinstance(e_vector, np.ndarray):
            e_vector = torch.from_numpy(e_vector).to(device=device, dtype=torch.float32)
        else:
            e_vector = e_vector.to(device=device, dtype=torch.float32)
        e_vector = e_vector.view(-1)

        # norms
        ct_norm = float(torch.norm(c_t).item())
        et_norm = float(torch.norm(e_vector).item())

        # dct_norm = float(torch.norm(c_t - c_prev).item()) if (c_prev is not None) else 0.0
        if (c_prev is not None) and (c_prev.shape == c_t.shape):
            dct_norm = float(torch.norm(c_t - c_prev).item())
        else:
            dct_norm = 0.0
        det_norm = float(torch.norm(e_vector - e_prev).item()) if (e_prev is not None) else 0.0

        # log row
        append_row_csv(
            OUT_CSV, header,
            [
                mode, env_id, trial, t,
                gravity, wind, turb,
                shift_flag,
                float(reward),
                ct_norm, dct_norm,
                et_norm, det_norm
            ]
        )

        # advance
        c_prev = c_t.detach()
        e_prev = e_vector.detach()
        reward_prev = float(reward)
        state = next_state

        # if terminated early, reset immediately but keep same params
        if terminated or truncated:
            state, _ = env.reset()
            apply_env_params(env, gravity, wind, turb)


def main():
    # detect embedding dim once
    with torch.no_grad():
        Ttmp = context_tracker.T
        e_dummy = torch.zeros((1, Ttmp, 3), device=device, dtype=torch.float32)
        r_dummy = torch.zeros((1, Ttmp), device=device, dtype=torch.float32)
        phi_dummy = PHI_RISKY.view(1, -1)
        C_DIM = int(context_tracker.context_net.get_embedding(e_dummy, r_dummy, phi_dummy).shape[-1])
    print("Detected C_DIM =", C_DIM)

    # create a single env instance and reuse
    env = gym.make(
        "LunarLander-v3",
        gravity=float(gravity_range[0]),
        enable_wind=True,
        wind_power=float(wind_power_range[0]),
        turbulence_power=float(turbulence_power_range[0]),
    )

    total_runs = len(MODES) * 100 * EPISODES_PER_ENV
    pbar = tqdm(total=total_runs, desc="Infer shift reaction")

    for mode in MODES:
        for env_id in range(100):
            for trial in range(EPISODES_PER_ENV):
                run_one_episode(env, mode, env_id, trial, C_DIM)
                pbar.update(1)

    pbar.close()
    env.close()
    print("[Done] Saved:", OUT_CSV)


if __name__ == "__main__":
    main()