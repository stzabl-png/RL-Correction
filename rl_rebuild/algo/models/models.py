# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class PointNetEncoder(nn.Module):
    """Lightweight permutation-invariant point encoder: per-point MLP -> max-pool.
    Input (..., N, in_dim) [xyz(3) + mask(2)] -> feature (..., feat_dim)."""
    def __init__(self, in_dim=5, feat_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ELU(),
            nn.Linear(64, 128), nn.ELU(),
            nn.Linear(128, feat_dim), nn.ELU(),
        )

    def forward(self, pc):
        x = self.mlp(pc)                 # (..., N, feat_dim)
        x = torch.max(x, dim=-2).values  # (..., feat_dim)  permutation-invariant pool
        return x


class ProprioAdaptTConv(nn.Module):
    def __init__(self, frame_shape): # frame_shape: 42 or 47_w_tactile
        super(ProprioAdaptTConv, self).__init__()
        self.channel_transform = nn.Sequential(
            nn.Linear(frame_shape, frame_shape),
            nn.ReLU(inplace=True),
            nn.Linear(frame_shape, frame_shape),
            nn.ReLU(inplace=True),
        )
        self.temporal_aggregation = nn.Sequential(
            nn.Conv1d(frame_shape, frame_shape, (9,), stride=(2,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(frame_shape, frame_shape, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
            nn.Conv1d(frame_shape, frame_shape, (5,), stride=(1,)),
            nn.ReLU(inplace=True),
        )
        self.low_dim_proj = nn.Linear(frame_shape * 3, 8)

    def forward(self, x):
        x = self.channel_transform(x)  # (N, 30, frame_shape)
        x = x.permute((0, 2, 1))  # (N, frame_shape, 30)
        x = self.temporal_aggregation(x)  # (N, frame_shape, 3)
        x = self.low_dim_proj(x.flatten(1))
        return x


class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.units = kwargs.pop('actor_units')
        self.critic_units = kwargs.pop('critic_units', self.units)
        self.priv_mlp = kwargs.pop('priv_mlp_units')
        mlp_input_shape = input_shape[0]

        out_size = self.units[-1]
        self.priv_info = kwargs['priv_info']
        self.priv_info_stage2 = kwargs['proprio_adapt']
        # --- v3net (2026-07-02, network_capacity_audit): all default to legacy behavior ---
        # separate_critic: independent critic trunk on (raw obs + FULL raw priv) — true asymmetric
        #   critic (Dactyl-style); the actor keeps its extrinsics embedding and stays deployable.
        # actor_priv_dim: the actor's env_mlp embeds only the FIRST K priv dims (the HORA extrinsics);
        #   dims beyond K (live object state, contacts) are critic-only.
        # sigma_floor: min action std (clamps the learned state-independent log-std from below).
        self.separate_critic = kwargs.get('separate_critic', False)
        self.actor_priv_dim = int(kwargs.get('actor_priv_dim') or kwargs['priv_info_dim'])
        self.sigma_floor = kwargs.get('sigma_floor', None)
        if self.priv_info:
            mlp_input_shape += self.priv_mlp[-1]
            self.env_mlp = MLP(units=self.priv_mlp, input_size=self.actor_priv_dim)

            if self.priv_info_stage2:
                self.adapt_tconv = ProprioAdaptTConv(input_shape[0]//3)
        # REBUILD STAGE 2: point-cloud branch (PointNet), shared by actor AND critic.
        # 必须建在 critic_mlp 之前 —— critic 也要吃形状特征, 否则多物体时 critic
        # 无法区分物体, 价值估计对所有形状是同一个函数, 会拖慢甚至误导学习.
        self.use_pc = kwargs.get('pointcloud', False)
        self.pc_feat_dim = 0
        if self.use_pc:
            self.pc_feat_dim = kwargs.get('pc_feat_dim', 128)
            self.pc_encoder = PointNetEncoder(kwargs.get('pc_in_dim', 5), self.pc_feat_dim)
            mlp_input_shape += self.pc_feat_dim
        if self.separate_critic:
            assert self.critic_units[-1] == out_size, (
                "actor and critic trunks must end at the shared value-head width")
            self.critic_mlp = MLP(units=self.critic_units,
                                  input_size=input_shape[0] + kwargs['priv_info_dim'] + self.pc_feat_dim)

        self.actor_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        self.value = torch.nn.Linear(out_size, 1)
        self.mu = torch.nn.Linear(out_size, actions_num)
        self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)

        # REBUILD STAGE 3: world-model head — predict object pose (pos[3] + quat[4]) from the actor latent
        self.use_wm = kwargs.get('world_model', False)
        if self.use_wm:
            self.wm_head = nn.Sequential(nn.Linear(out_size, 128), nn.ELU(), nn.Linear(128, 7))

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
        # sigma_init: initial action std (NOT log-std). Absent -> legacy exp(0)=1.0.
        # For residual-correction tasks a std of 1.0 x the residual bound is a violent
        # per-step perturbation that destroys contact before any gradient can form.
        _sig0 = kwargs.get('sigma_init', None)
        nn.init.constant_(self.sigma, 0 if _sig0 is None else float(np.log(_sig0)))
        # mu_init_scale: shrink the policy head so mu ~= 0 at init. For residual
        # correction this makes the initial policy the *reference itself* (a verified
        # working trajectory) instead of a random projection of the obs. Absent -> legacy.
        _mu0 = kwargs.get('mu_init_scale', None)
        if _mu0 is not None:
            with torch.no_grad():
                self.mu.weight.mul_(float(_mu0))

    @torch.no_grad()
    def act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value, _, _, _ = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1), # self.neglogp(selected_action, mu, sigma, logstd),
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        mu, logstd, value, _, _, _ = self._actor_critic(obs_dict)
        return mu

    def _actor_critic(self, obs_dict):
        obs = obs_dict['obs']
        obs_raw = obs                                # pre-embedding obs, for the separate critic
        extrin, extrin_gt = None, None
        if self.priv_info:
            if self.priv_info_stage2:
                extrin = self.adapt_tconv(obs_dict['proprio_hist'])
                # during supervised training, extrin has gt label (actor extrinsics = first K priv dims)
                extrin_gt = self.env_mlp(obs_dict['priv_info'][..., :self.actor_priv_dim]) \
                    if 'priv_info' in obs_dict else extrin
                extrin_gt = torch.tanh(extrin_gt)
                extrin = torch.tanh(extrin)
                obs = torch.cat([obs, extrin], dim=-1)
            else:
                extrin = self.env_mlp(obs_dict['priv_info'][..., :self.actor_priv_dim])
                extrin = torch.tanh(extrin)
                obs = torch.cat([obs, extrin], dim=-1)

        pc_feat = None
        if self.use_pc and ('pointcloud' in obs_dict) and (obs_dict['pointcloud'] is not None):
            pc_feat = self.pc_encoder(obs_dict['pointcloud'])
            obs = torch.cat([obs, pc_feat], dim=-1)

        x = self.actor_mlp(obs)
        if self.separate_critic and ('priv_info' in obs_dict):
            _cin = [obs_raw, obs_dict['priv_info']]
            if pc_feat is not None:
                _cin.append(pc_feat)          # critic 与 actor 共享同一个 PointNet 编码
            xc = self.critic_mlp(torch.cat(_cin, dim=-1))
            value = self.value(xc)
        else:
            value = self.value(x)                    # legacy shared-trunk path (and priv-less inference)
        mu = self.mu(x)
        logstd = self.sigma
        if self.sigma_floor is not None:
            logstd = torch.clamp(logstd, min=float(np.log(self.sigma_floor)))
        wm_pred = self.wm_head(x) if self.use_wm else None
        return mu, mu * 0 + logstd, value, extrin, extrin_gt, wm_pred

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        rst = self._actor_critic(input_dict)
        mu, logstd, value, extrin, extrin_gt, wm_pred = rst
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
            'extrin': extrin,
            'extrin_gt': extrin_gt,
            'wm_pred': wm_pred,
        }
        return result
