"""One-shot actor behavior-cloning warmup; never called during PPO updates."""
from __future__ import annotations

import numpy as np
import torch


def warm_actor(agent, expert_npz: str, epochs: int = 200, batch_size: int = 256):
    z = np.load(expert_npz)
    obs = torch.tensor(z["obs"], dtype=torch.float32, device=agent.device)
    priv = torch.tensor(z["priv_info"], dtype=torch.float32, device=agent.device)
    act = torch.tensor(z["actions"], dtype=torch.float32, device=agent.device)
    assert obs.shape[1] == agent.obs_shape[0]
    assert act.shape[1] == agent.actions_num
    # Fit normalization statistics once on the successful demonstration.
    agent.running_mean_std.train()
    with torch.no_grad(): agent.running_mean_std(obs)
    agent.running_mean_std.eval()
    params = list(agent.model.actor_mlp.parameters()) + list(agent.model.mu.parameters())
    if hasattr(agent.model, "env_mlp"): params += list(agent.model.env_mlp.parameters())
    opt = torch.optim.Adam(params, lr=3e-4)
    rng = np.random.RandomState(0)
    final = None
    for epoch in range(int(epochs)):
        order = rng.permutation(len(obs))
        losses = []
        for lo in range(0, len(order), batch_size):
            ids = torch.tensor(order[lo:lo+batch_size], device=agent.device)
            mu, *_ = agent.model._actor_critic({
                "obs": agent.running_mean_std(obs[ids]), "priv_info": priv[ids]})
            loss = (mu - act[ids]).square().mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            losses.append(float(loss.detach()))
        final = float(np.mean(losses))
        if epoch % 25 == 0 or epoch == epochs - 1:
            print(f"[BC] epoch={epoch:03d} mse={final:.6f}", flush=True)
    agent.running_mean_std.train()
    return final
