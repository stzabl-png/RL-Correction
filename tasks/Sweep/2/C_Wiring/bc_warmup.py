"""One-shot expert warmup; never called during on-policy PPO updates."""
from __future__ import annotations

import numpy as np
import torch


def _load(path: str, device: str):
    z = np.load(path)
    required = {"obs", "priv_info", "actions", "return_target"}
    assert required.issubset(z.files), (path, required - set(z.files))
    return {key: torch.tensor(z[key], dtype=torch.float32, device=device)
            for key in required}


def warm_actor_critic(agent, expert25_npz: str, expert40_npz: str,
                      failure_npz: str, actor_epochs: int = 300,
                      critic_epochs: int = 200, batch_size: int = 256):
    """BC the actor on 40 mm, then fit the critic on 25/40/failure returns."""
    d25 = _load(expert25_npz, agent.device)
    d40 = _load(expert40_npz, agent.device)
    dfail = _load(failure_npz, agent.device)
    obs = d40["obs"]
    priv = d40["priv_info"]
    act = d40["actions"]
    assert obs.shape[1] == agent.obs_shape[0]
    assert priv.shape[1] == agent.priv_info_dim
    assert act.shape[1] == agent.actions_num
    # Fit observation normalization on both near-success demonstrations and the
    # canonical failure, even though only 40 mm supervises the Actor.
    agent.running_mean_std.train()
    with torch.no_grad():
        agent.running_mean_std(torch.cat([d25["obs"], d40["obs"], dfail["obs"]], 0))
    agent.running_mean_std.eval()
    nonzero40 = d40["actions"].abs().amax(1) > 1.0e-6
    # The more assertive 40 mm trajectory is the sole Actor demonstration.
    # Keep its reference-only prefix at low weight so it cannot drown the short
    # non-zero correction window.
    weight = torch.where(
        nonzero40, torch.ones_like(nonzero40, dtype=torch.float32),
        torch.full_like(nonzero40, 0.05, dtype=torch.float32))
    params = list(agent.model.actor_mlp.parameters()) + list(agent.model.mu.parameters())
    if hasattr(agent.model, "env_mlp"): params += list(agent.model.env_mlp.parameters())
    opt = torch.optim.Adam(params, lr=3e-4)
    rng = np.random.RandomState(0)
    actor_final = None
    for epoch in range(int(actor_epochs)):
        order = rng.permutation(len(obs))
        losses = []
        for lo in range(0, len(order), batch_size):
            ids = torch.tensor(order[lo:lo+batch_size], device=agent.device)
            mu, *_ = agent.model._actor_critic({
                "obs": agent.running_mean_std(obs[ids]), "priv_info": priv[ids]})
            per = (mu - act[ids]).square().mean(1)
            loss = (per * weight[ids]).sum() / weight[ids].sum().clamp_min(1.0e-8)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            losses.append(float(loss.detach()))
        actor_final = float(np.mean(losses))
        if epoch % 25 == 0 or epoch == actor_epochs - 1:
            print(f"[BC actor] epoch={epoch:03d} weighted_mse={actor_final:.6f}", flush=True)

    c_obs = torch.cat([d25["obs"], d40["obs"], dfail["obs"]], 0)
    c_priv = torch.cat([d25["priv_info"], d40["priv_info"], dfail["priv_info"]], 0)
    c_ret_raw = torch.cat([d25["return_target"], d40["return_target"],
                           dfail["return_target"]], 0).reshape(-1, 1)
    agent.value_mean_std.train()
    with torch.no_grad(): agent.value_mean_std(c_ret_raw)
    agent.value_mean_std.eval()
    c_ret = agent.value_mean_std(c_ret_raw)
    critic_params = list(agent.model.critic_mlp.parameters()) + list(agent.model.value.parameters())
    critic_param_ids = {id(p) for p in critic_params}
    actor_params = [p for p in agent.model.parameters() if id(p) not in critic_param_ids]
    for param in actor_params:
        param.requires_grad_(False)
    copt = torch.optim.Adam(critic_params, lr=3e-4)
    critic_final = None
    for epoch in range(int(critic_epochs)):
        order = rng.permutation(len(c_obs)); losses = []
        for lo in range(0, len(order), batch_size):
            ids = torch.tensor(order[lo:lo+batch_size], device=agent.device)
            _, _, value, *_ = agent.model._actor_critic({
                "obs": agent.running_mean_std(c_obs[ids]),
                "priv_info": c_priv[ids]})
            loss = torch.nn.functional.smooth_l1_loss(value, c_ret[ids])
            copt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(critic_params, 1.0); copt.step()
            losses.append(float(loss.detach()))
        critic_final = float(np.mean(losses))
        if epoch % 25 == 0 or epoch == critic_epochs - 1:
            print(f"[BC critic] epoch={epoch:03d} huber={critic_final:.6f}", flush=True)
    for param in actor_params:
        param.requires_grad_(True)
    agent.running_mean_std.train()
    agent.value_mean_std.train()
    return {"actor_weighted_mse": actor_final, "critic_huber": critic_final,
            "actor_samples": len(obs), "critic_samples": len(c_obs)}
