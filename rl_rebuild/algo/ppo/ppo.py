# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: RLGames
# Copyright (c) 2019 Denys88
# Licence under MIT License
# https://github.com/Denys88/rl_games/
# --------------------------------------------------------

import os
import time
import numpy as np
import torch

from rl_rebuild.algo.ppo.experience import ExperienceBuffer
from rl_rebuild.algo.models.models import ActorCritic
from rl_rebuild.algo.models.running_mean_std import RunningMeanStd

from rl_rebuild.utils.misc import AverageScalarMeter

from tensorboardX import SummaryWriter
from rl_rebuild.utils.wandb_writer import TBWriter


class PPO(object):
    def __init__(self, env, output_dir, full_config, create_output_dir=True):
        self.device = full_config.train["device"]
        self.network_config = full_config.train["network"]
        self.ppo_config = full_config.train["algorithm"]
        # ---- build environment ----
        self.env = env
        self.num_actors = self.ppo_config['num_actors']
        action_space = self.env.action_space
        self.actions_num = action_space.shape[1]
        self.actions_low = torch.from_numpy(action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(action_space.high.copy()).float().to(self.device)
        self.observation_space = self.env.observation_space
        self.obs_shape = (self.observation_space.shape[1],)
        # ---- Priv Info ----
        self.priv_info_dim = self.ppo_config['priv_info_dim']
        self.priv_info = self.ppo_config['priv_info']
        # ---- REBUILD STAGE 2/3: point cloud + world model (read from env cfg) ----
        self.use_pc = bool(getattr(full_config.task, 'enable_pointcloud', False))
        self.use_wm = bool(getattr(full_config.task, 'enable_world_model', False))
        self.wm_coef = float(getattr(full_config.task, 'world_model_coef', 1.0))
        self.pc_in_dim = int(getattr(full_config.task, 'pc_in_dim', 5))
        self.pc_num = 0
        if self.use_pc or self.use_wm:
            _probe = self.env.reset()
            if self.use_pc:
                self.pc_num = int(_probe['pointcloud'].shape[1])
                print(f"[REBUILD] point cloud ON: {self.pc_num} points x {self.pc_in_dim}")
            if self.use_wm:
                print(f"[REBUILD] world-model loss ON (coef={self.wm_coef})")
        # ---- Model ----
        net_config = {
            'actor_units': self.network_config["mlp"]["units"],
            'critic_units': self.network_config.get(
                "critic_mlp", self.network_config["mlp"])["units"],
            'priv_mlp_units': self.network_config["priv_mlp"]["units"],
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'priv_info': self.priv_info,
            'proprio_adapt': False,
            'priv_info_dim': self.priv_info_dim,
            'pointcloud': self.use_pc,
            'pc_in_dim': self.pc_in_dim,
            'pc_feat_dim': int(getattr(full_config.task, 'pc_feat_dim', 128)),
            'world_model': self.use_wm,
            # v3net (network_capacity_audit_2026-07-02): absent keys -> legacy behavior
            'separate_critic': self.network_config.get('separate_critic', False),
            'actor_priv_dim': self.network_config.get('actor_priv_dim', None),
            'sigma_floor': self.network_config.get('sigma_floor', None),
            'sigma_init': self.network_config.get('sigma_init', None),
            'mu_init_scale': self.network_config.get('mu_init_scale', None),
        }
        if net_config['separate_critic']:
            print(f"[v3net] separate critic ON (priv {self.priv_info_dim} raw -> critic; "
                  f"actor embeds first {net_config['actor_priv_dim'] or self.priv_info_dim})")
        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)
        # ---- Output Dir ----
        # allows us to specify a folder where all experiments will reside
        self.output_dir = output_dir
        self.nn_dir = os.path.join(self.output_dir, 'stage1_nn')
        self.tb_dif = os.path.join(self.output_dir, 'stage1_tb')
        if create_output_dir:
            os.makedirs(self.nn_dir, exist_ok=True)
            os.makedirs(self.tb_dif, exist_ok=True)
        # ---- Optim ----
        self.last_lr = float(self.ppo_config['learning_rate'])
        self.weight_decay = self.ppo_config.get('weight_decay', 0.0)
        _betas = tuple(self.ppo_config.get('adam_betas', (0.9, 0.999)))
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.last_lr, betas=_betas, weight_decay=self.weight_decay)
        # ---- PPO Train Param ----
        self.e_clip = self.ppo_config['e_clip']
        self.clip_value = self.ppo_config['clip_value']
        self.entropy_coef = self.ppo_config['entropy_coef']
        self.critic_coef = self.ppo_config['critic_coef']
        self.bounds_loss_coef = self.ppo_config['bounds_loss_coef']
        self.gamma = self.ppo_config['gamma']
        self.tau = self.ppo_config['tau']
        self.truncate_grads = self.ppo_config['truncate_grads']
        self.grad_norm = self.ppo_config['grad_norm']
        self.value_bootstrap = self.ppo_config['value_bootstrap']
        self.normalize_advantage = self.ppo_config['normalize_advantage']
        self.normalize_input = self.ppo_config['normalize_input']
        self.normalize_value = self.ppo_config['normalize_value']
        # ---- PPO Collect Param ----
        self.horizon_length = self.ppo_config['horizon_length']
        self.batch_size = self.horizon_length * self.num_actors
        self.minibatch_size = self.ppo_config['minibatch_size']
        self.mini_epochs_num = self.ppo_config['mini_epochs']
        assert self.batch_size % self.minibatch_size == 0 or full_config.test
        # ---- scheduler ----
        self.kl_threshold = self.ppo_config['kl_threshold']
        self.critic_warmup_iters = int(self.ppo_config.get('critic_warmup_iters', 0))
        # max_lr 设成初始 lr => 调度器只能减速不能加速 (非对称信赖域).
        # 2026-07-21 教训: 对称调度在本任务两侧都会失控 —— kl_threshold 0.02 时 lr 塌到
        # 1e-5 策略冻死; 改 0.05 后 KL 长期低于加速线, lr 每轮 x1.5 冲到 5.1e-3 把已经
        # 学到 task=0.41 的策略打烂. 残差修正需要慢而稳, 不需要自动提速.
        self.scheduler = AdaptiveScheduler(
            self.kl_threshold,
            max_lr=float(self.ppo_config.get('max_lr', 1e-2)),
            min_lr=float(self.ppo_config.get('min_lr', 1e-6)))
        # ---- Snapshot
        self.save_freq = self.ppo_config['save_frequency']
        self.save_best_after = self.ppo_config['save_best_after']
        # ---- TensorBoard + Weights & Biases Logger (W&B on by default; SHARPA_WANDB=0 to disable) ----
        self.extra_info = {}
        if create_output_dir:
            self.writer = TBWriter(self.tb_dif, config=full_config.train)

        self.episode_rewards = AverageScalarMeter(100)
        self.episode_lengths = AverageScalarMeter(100)
        self.obs = None
        self.epoch_num = 0
        # 每个 epoch 边界调用一次 (train.py 挂 gpu_guard 的让出点; 推理路径不设 = 不生效).
        self.epoch_hook = None
        self._in_critic_warmup = False   # 推理路径(record/play)不进 train_epoch, 给个安全缺省
        self.storage = ExperienceBuffer(
            self.num_actors, self.horizon_length, self.batch_size, self.minibatch_size, self.obs_shape[0],
            self.actions_num, self.priv_info_dim, self.device,
            pc_num=self.pc_num, pc_in_dim=self.pc_in_dim, use_wm=self.use_wm,
        )

        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config['max_agent_steps']
        self.best_rewards = -10000
        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls):
        self.writer.add_scalar('performance/RLTrainFPS', self.agent_steps / self.rl_train_time, self.agent_steps)
        self.writer.add_scalar('performance/EnvStepFPS', self.agent_steps / self.data_collect_time, self.agent_steps)

        self.writer.add_scalar('losses/actor_loss', torch.mean(torch.stack(a_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/bounds_loss', torch.mean(torch.stack(b_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/critic_loss', torch.mean(torch.stack(c_losses)).item(), self.agent_steps)
        self.writer.add_scalar('losses/entropy', torch.mean(torch.stack(entropies)).item(), self.agent_steps)

        self.writer.add_scalar('info/last_lr', self.last_lr, self.agent_steps)
        self.writer.add_scalar('info/e_clip', self.e_clip, self.agent_steps)
        self.writer.add_scalar('info/kl', torch.mean(torch.stack(kls)).item(), self.agent_steps)
        if getattr(self, '_clip_fracs', None):
            # PPO 里被裁掉的样本占比: 太低(<2%)=步长白给, 太高(>30%)=更新一直在撞天花板
            self.writer.add_scalar('info/clip_frac',
                                   float(np.mean(self._clip_fracs)), self.agent_steps)
            self._clip_fracs = []

        for k, v in self.extra_info.items():
            self.writer.add_scalar(f'{k}', v, self.agent_steps)

    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()
        if self.normalize_value:
            self.value_mean_std.train()

    def model_act(self, obs_dict):
        processed_obs = self.running_mean_std(obs_dict['obs'])
        input_dict = {
            'obs': processed_obs,
            'priv_info': obs_dict['priv_info'],
        }
        if self.use_pc:
            input_dict['pointcloud'] = obs_dict['pointcloud']
        res_dict = self.model.act(input_dict)
        res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        return res_dict

    def train(self):
        _t = time.time()
        _last_t = time.time()
        self.obs = self.env.reset()
        self.agent_steps = max(
            self.batch_size, int(getattr(self, 'initial_agent_steps', 0)))

        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1
            a_losses, c_losses, b_losses, entropies, kls = self.train_epoch()
            self.storage.data_dict = None

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()

            self.write_stats(a_losses, c_losses, b_losses, entropies, kls)

            mean_rewards = self.episode_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            self.writer.add_scalar('episode_rewards/step', mean_rewards, self.agent_steps)
            self.writer.add_scalar('episode_lengths/step', mean_lengths, self.agent_steps)
            checkpoint_name = f'ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}M_reward_{mean_rewards:.2f}'

            if self.save_freq > 0:
                if self.epoch_num % self.save_freq == 0:
                    self.save(os.path.join(self.nn_dir, checkpoint_name))
                    self.save(os.path.join(self.nn_dir, 'last'))

            if mean_rewards > self.best_rewards and self.epoch_num >= self.save_best_after:
                print(f'save current best reward: {mean_rewards:.2f}')
                self.best_rewards = mean_rewards
                self.save(os.path.join(self.nn_dir, 'best'))

            info_string = f'Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | ' \
                          f'Last FPS: {last_fps:.1f} | ' \
                          f'Collect Time: {self.data_collect_time / 60:.1f} min | ' \
                          f'Train RL Time: {self.rl_train_time / 60:.1f} min | ' \
                          f'Mean Rewards: {mean_rewards:.2f} | ' \
                          f'Current Best: {self.best_rewards:.2f}'
            print(info_string)

            # 录像让出点: 此刻 ckpt 刚落盘, 是挂起的干净位置. 无请求时开销 = 一次 stat.
            # 两个 Isaac 同时满载会把这台机器的电源打到 OCP 整机瞬断, 见 utils/gpu_guard.py.
            if self.epoch_hook is not None:
                self.epoch_hook()

        print('max steps achieved')

    def save(self, name):
        weights = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'agent_steps': int(self.agent_steps),
            'epoch_num': int(self.epoch_num),
            'last_lr': float(self.last_lr),
        }
        if self.running_mean_std:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.value_mean_std:
            weights['value_mean_std'] = self.value_mean_std.state_dict()
        torch.save(weights, f'{name}.pth')

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint['model'])
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.initial_agent_steps = int(checkpoint.get('agent_steps', 0))
        self.epoch_num = int(checkpoint.get('epoch_num', self.epoch_num))
        if 'last_lr' in checkpoint:
            self.last_lr = float(checkpoint['last_lr'])
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.last_lr
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        # Also restore the VALUE normalizer if present (the checkpoint saves it but this method
        # previously ignored it -> on --resume the critic scale reset, causing a reward transient
        # that knocked resumed policies off their behavior). Guarded so non-value-norm ckpts are safe.
        if 'value_mean_std' in checkpoint and hasattr(self, 'value_mean_std') and self.value_mean_std is not None:
            self.value_mean_std.load_state_dict(checkpoint['value_mean_std'])

    def restore_test(self, fn):
        checkpoint = torch.load(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def test(self):
        self.set_eval()
        obs_dict = self.env.reset()
        while True:
            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']),
                'priv_info': obs_dict['priv_info'],
            }
            mu = self.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)

    def train_epoch(self):
        self._in_critic_warmup = self.epoch_num <= self.critic_warmup_iters
        if self.critic_warmup_iters and self.epoch_num == 1:
            print(f'[warmup] critic-only 前 {self.critic_warmup_iters} epoch (actor 冻结, lr 调度暂停)')
        if self.critic_warmup_iters and self.epoch_num == self.critic_warmup_iters + 1:
            print(f'[warmup] 结束 @epoch {self.epoch_num}, actor 解冻, lr={self.last_lr:.2e}')
        # collect minibatch data
        _t = time.time()
        self.set_eval()
        self.play_steps()
        self.data_collect_time += (time.time() - _t)
        # update network
        _t = time.time()
        self.set_train()
        a_losses, b_losses, c_losses = [], [], []
        entropies, kls = [], []
        for _ in range(0, self.mini_epochs_num):
            ep_kls = []
            for i in range(len(self.storage)):
                value_preds, old_action_log_probs, advantage, old_mu, old_sigma, \
                    returns, actions, obs, priv_info, pointcloud, obj_pose, actor_mask = self.storage[i]

                obs = self.running_mean_std(obs)
                batch_dict = {
                    'prev_actions': actions,
                    'obs': obs,
                    'priv_info': priv_info,
                }
                if self.use_pc:
                    batch_dict['pointcloud'] = pointcloud
                if os.environ.get('MEMDBG')=='1' and self.epoch_num<=2:
                    torch.cuda.synchronize(); _m0=torch.cuda.memory_allocated()/2**20
                res_dict = self.model(batch_dict)
                if os.environ.get('MEMDBG')=='1' and self.epoch_num<=2:
                    torch.cuda.synchronize(); _m1=torch.cuda.memory_allocated()/2**20
                    print(f'[mem] ep{self.epoch_num} mb{i} 前向 {_m0:8.1f}->{_m1:8.1f}MB (+{_m1-_m0:7.1f}) 峰值 {torch.cuda.max_memory_allocated()/2**20:8.1f} 保留 {torch.cuda.memory_reserved()/2**20:8.1f}', flush=True)
                action_log_probs = res_dict['prev_neglogp']
                values = res_dict['values']
                entropy = res_dict['entropy']
                mu = res_dict['mus']
                sigma = res_dict['sigmas']

                # actor loss
                ratio = torch.exp(old_action_log_probs - action_log_probs)
                surr1 = advantage * ratio
                surr2 = advantage * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)
                with torch.no_grad():   # 盘面: 被裁样本占比(不参与反传)
                    if not hasattr(self, '_clip_fracs'):
                        self._clip_fracs = []
                    self._clip_fracs.append(
                        ((ratio - 1.0).abs() > self.e_clip).float().mean().item())
                a_loss = torch.max(-surr1, -surr2)
                # critic loss
                value_pred_clipped = value_preds + (values - value_preds).clamp(-self.e_clip, self.e_clip)
                value_losses = (values - returns) ** 2
                value_losses_clipped = (value_pred_clipped - returns) ** 2
                c_loss = torch.max(value_losses, value_losses_clipped)
                # bounded loss
                if self.bounds_loss_coef > 0:
                    soft_bound = 1.1
                    mu_loss_high = torch.clamp_max(mu - soft_bound, 0.0) ** 2
                    mu_loss_low = torch.clamp_max(-mu + soft_bound, 0.0) ** 2
                    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                else:
                    b_loss = 0
                mask = actor_mask.squeeze(-1)
                mask_denom = mask.sum().clamp_min(1.0)
                a_loss = (a_loss * mask).sum() / mask_denom
                entropy = (entropy * mask).sum() / mask_denom
                if torch.is_tensor(b_loss):
                    b_loss = (b_loss * mask).sum() / mask_denom
                c_loss = torch.mean(c_loss)

                # critic warmup: 前 N 个 epoch 只训 critic, actor 完全冻结.
                # 随机初始化的 critic 给出的 advantage 是噪声, 拿它加权策略梯度会让 mu
                # 随机游走 —— 对"初值本身就是可用解"的残差修正任务是致命的.
                # separate_critic=True 时 critic 有独立主干, 所以只留 c_loss 即可隔离 actor.
                if self._in_critic_warmup:
                    loss = 0.5 * c_loss * self.critic_coef
                else:
                    loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef

                # REBUILD STAGE 3: world-model auxiliary loss (predict object pose from the actor latent)
                if self.use_wm and (obj_pose is not None) and (res_dict.get('wm_pred', None) is not None):
                    wm_pred = res_dict['wm_pred']
                    wm_pos_loss = ((wm_pred[:, :3] - obj_pose[:, :3]) ** 2).sum(-1)
                    # quaternion: 1 - <q_pred_normalized, q_true>^2  (sign-invariant orientation error)
                    qn = wm_pred[:, 3:] / (torch.norm(wm_pred[:, 3:], dim=-1, keepdim=True) + 1e-6)
                    qt = obj_pose[:, 3:]
                    wm_rot_loss = 1.0 - (qn * qt).sum(-1) ** 2
                    wm_loss = (wm_pos_loss + wm_rot_loss).mean()
                    loss = loss + self.wm_coef * wm_loss
                    self.extra_info['losses/world_model'] = wm_loss.detach()

                self.optimizer.zero_grad()
                loss.backward()
                if self.truncate_grads:
                    _gn = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                    self.writer.add_scalar('info/grad_norm', float(_gn), self.agent_steps)
                self.optimizer.step()

                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma,
                                        actor_mask)

                kl = kl_dist
                # detach 后再收集: 这些 list 只用于 write_stats 求均值打日志.
                # 不 detach 会把每个 minibatch 的整张 autograd 图钉住 ——
                # mini_epochs(5) x minibatches(4) = 20 张图同时驻留.
                # MLP 时每张图只有几 MB 不显眼; 接上 PointNet 后每张 243MB -> 泄漏 ~4.9GB.
                a_losses.append(a_loss.detach())
                c_losses.append(c_loss.detach())
                ep_kls.append(kl)
                entropies.append(entropy.detach())
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss.detach() if torch.is_tensor(b_loss) else b_loss)

                self.storage.update_mu_sigma(mu.detach(), sigma.detach())

            av_kls = torch.mean(torch.stack(ep_kls))
            # warmup 期 actor 不动 -> KL≈0 -> 调度器会每轮把 lr ×1.5 顶到 max_lr(1e-2),
            # warmup 一结束就用这个爆炸的 lr 砸 actor. 所以 warmup 期跳过调度.
            if not self._in_critic_warmup:
                self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.last_lr
            kls.append(av_kls)

        self.rl_train_time += (time.time() - _t)
        return a_losses, c_losses, b_losses, entropies, kls

    def play_steps(self):
        for n in range(self.horizon_length):
            res_dict = self.model_act(self.obs)
            # collect o_t
            self.storage.update_data('obses', n, self.obs['obs'])
            self.storage.update_data('priv_info', n, self.obs['priv_info'])
            self.storage.update_data(
                'actor_mask', n,
                self.obs.get('actor_mask', torch.ones(
                    self.num_actors, 1, device=self.device)))
            if self.use_pc:
                self.storage.update_data('pointcloud', n, self.obs['pointcloud'])
            if self.use_wm:
                self.storage.update_data('obj_pose', n, self.obs['obj_pose'])
            for k in ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']:
                self.storage.update_data(k, n, res_dict[k])
            # do env step
            actions = torch.clamp(res_dict['actions'], -1.0, 1.0)
            self.obs, rewards, self.dones, infos = self.env.step(actions)
            rewards = rewards.unsqueeze(1)
            # update dones and rewards after env step
            self.storage.update_data('dones', n, self.dones)
            shaped_rewards = 0.01 * rewards.clone()
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * infos['time_outs'].unsqueeze(1).float()
            self.storage.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])

            assert isinstance(infos, dict), 'Info Should be a Dict'
            self.extra_info = {}
            for k, v in infos.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.extra_info[k] = v

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        res_dict = self.model_act(self.obs)
        last_values = res_dict['values']

        self.agent_steps += self.batch_size
        self.storage.computer_return(last_values, self.gamma, self.tau)
        self.storage.prepare_training()

        returns = self.storage.data_dict['returns']
        values = self.storage.data_dict['values']
        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
        self.storage.data_dict['values'] = values
        self.storage.data_dict['returns'] = returns


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma, mask=None):
    c1 = torch.log(p1_sigma/p0_sigma + 1e-5)
    c2 = (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)
    if mask is None:
        return kl.mean()
    weight = mask.squeeze(-1)
    return (kl * weight).sum() / weight.sum().clamp_min(1.0)


# from https://github.com/leggedrobotics/rsl_rl/blob/master/rsl_rl/algorithms/ppo.py
class AdaptiveScheduler(object):
    def __init__(self, kl_threshold=0.008, max_lr=1e-2, min_lr=1e-6):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr
