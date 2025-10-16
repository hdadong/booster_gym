import numpy as np
import torch

from torch.distributions import Normal
import torch
import torch.nn.functional as F
from utils.normalize import minmaxnormalizer


class TanhBijector:
    """Tanh Bijector."""

    def forward(self, x):
        return torch.tanh(x)

    def inverse(self, y):
        # Clamping the input to avoid numerical issues with arctanh
        return torch.arctanh(torch.clamp(y, -0.999999, 0.999999))

    def forward_log_det_jacobian(self, x):
        # Computing the log of the absolute value of the Jacobian determinant
        return 2. * (torch.log(torch.tensor(2.0)) - x - F.softplus(-2. * x))


class NormalTanhDistribution:
    """Normal distribution followed by tanh."""

    def __init__(self, min_std=0.001, max_std=None):
        self.min_std = min_std
        self.max_std = max_std
        self.postprocessor = TanhBijector()

    def create_dist(self, parameters):
        loc, scale = torch.chunk(parameters, 2, dim=-1)
        if self.max_std is None:
            scale = F.softplus(scale) + self.min_std
        else:
            scale = torch.sigmoid(scale)
            scale = self.min_std + (self.max_std - self.min_std) * scale
        return Normal(loc, scale)

    def sample(self, parameters):
        dist = self.create_dist(parameters)
        return self.postprocessor.forward(dist.rsample())

    def mode(self, parameters):
        dist = self.create_dist(parameters)
        return self.postprocessor.forward(dist.mean)


class Policy:
    def __init__(self, cfg, policy_path):
        try:
            self.cfg = cfg
            self.policy = torch.jit.load(policy_path, map_location="cpu")
            self.policy.eval()
            self.wmobs_minmax_normalizer = minmaxnormalizer()
            self.normalizer = NormalTanhDistribution()

        except Exception as e:
            print(f"Failed to load policy: {e}")
            raise
        self._init_inference_variables()

    def get_policy_interval(self):
        return self.policy_interval

    def _init_inference_variables(self):
        self.default_dof_pos = np.array(self.cfg["common"]["default_qpos"], dtype=np.float32)
        self.stiffness = np.array(self.cfg["common"]["stiffness"], dtype=np.float32)
        self.damping = np.array(self.cfg["common"]["damping"], dtype=np.float32)

        self.commands = np.zeros(3, dtype=np.float32)
        self.smoothed_commands = np.zeros(3, dtype=np.float32)

        self.gait_frequency = self.cfg["policy"]["gait_frequency"]
        self.gait_process = 0.0
        self.dof_targets = np.copy(self.default_dof_pos)
        self.obs = np.zeros(self.cfg["policy"]["num_observations"], dtype=np.float32)
        self.wm_obs = np.zeros(self.cfg["policy"]["num_wm_observations"], dtype=np.float32)
        self.priv_obs = np.zeros(self.cfg["policy"]["num_priv_observations"], dtype=np.float32)


        self.actions = np.zeros(self.cfg["policy"]["num_actions"], dtype=np.float32)
        self.policy_interval = self.cfg["common"]["dt"] * self.cfg["policy"]["control"]["decimation"]

    def inference(self, time_now, dof_pos, dof_vel, base_ang_vel, projected_gravity, vx, vy, vyaw, 
                  quat_wxyz, base_lin_vel, body_height, ang_vel_global):
        self.gait_process = np.fmod(time_now * self.gait_frequency, 1.0)
        self.commands[0] = vx
        self.commands[1] = vy
        self.commands[2] = vyaw
        clip_range = (-self.policy_interval, self.policy_interval)
        self.smoothed_commands += np.clip(self.commands - self.smoothed_commands, *clip_range)

        if np.linalg.norm(self.smoothed_commands) < 1e-5:
            self.gait_frequency = 0.0
        else:
            self.gait_frequency = self.cfg["policy"]["gait_frequency"]

        self.obs[0:3] = projected_gravity
        self.obs[3:6] = base_ang_vel
        self.obs[6] = (
            self.smoothed_commands[0]  * (self.gait_frequency > 1.0e-8)
        )
        self.obs[7] = (
            self.smoothed_commands[1] * (self.gait_frequency > 1.0e-8)
        )
        self.obs[8] = (
            self.smoothed_commands[2] * (self.gait_frequency > 1.0e-8)
        )
        self.obs[9] = np.cos(2 * np.pi * self.gait_process) * (self.gait_frequency > 1.0e-8)
        self.obs[10] = np.sin(2 * np.pi * self.gait_process) * (self.gait_frequency > 1.0e-8)
        self.obs[11:23] = dof_pos[11:]
        self.obs[23:35] = dof_vel[11:]
        self.obs[35:47] = self.actions

        self.wm_obs[:self.obs.shape[0]] = self.obs
        self.wm_obs[self.wmobs_minmax_normalizer.wm_rpy_rate_idxs] = base_ang_vel
        self.wm_obs[self.wmobs_minmax_normalizer.wm_gravity_idxs] = projected_gravity
        self.wm_obs[self.wmobs_minmax_normalizer.wm_quat_idxs] = quat_wxyz
        self.wm_obs[self.wmobs_minmax_normalizer.wm_base_vel_idxs] = base_lin_vel
        self.wm_obs[self.wmobs_minmax_normalizer.wm_q_idxs] = dof_pos[11:]
        self.wm_obs[self.wmobs_minmax_normalizer.wm_qd_idxs] = dof_vel[11:]
        self.wm_obs[self.wmobs_minmax_normalizer.wm_height_idx] = body_height
        self.wm_obs[self.wmobs_minmax_normalizer.wm_gait_process_idx] = self.gait_process
        self.wm_obs[self.wmobs_minmax_normalizer.wm_gait_frequency_idx] = self.gait_frequency
        self.wm_obs = self.wmobs_minmax_normalizer.normalize_obs(self.wm_obs)[0]


        self.priv_obs[:self.obs.shape[0]] = self.obs
        self.priv_obs[self.wmobs_minmax_normalizer.priv_rpy_rate_idxs] = base_ang_vel
        self.priv_obs[self.wmobs_minmax_normalizer.priv_gravity_idxs] = projected_gravity
        self.priv_obs[self.wmobs_minmax_normalizer.priv_base_lin_vel_idxs] = base_lin_vel
        self.priv_obs[self.wmobs_minmax_normalizer.priv_global_ang_vel_idxs] = ang_vel_global
        self.priv_obs[self.wmobs_minmax_normalizer.priv_q_idxs] = dof_pos[11:]
        self.priv_obs[self.wmobs_minmax_normalizer.priv_qd_idxs] = dof_vel[11:]
        self.priv_obs[self.wmobs_minmax_normalizer.priv_height_idx] = body_height

        
        obs_torch = torch.tensor(self.obs)
        #obs_torch_minmaxnorm = self.obs_minmax_normalizer.normalize_obs(obs_torch)

        dist = self.policy(obs_torch.unsqueeze(0))        # -> shape [1, 2*A]
        self.actions[:]= self.normalizer.mode(dist).detach().numpy()     # -> shape [1, A]
        self.actions[:] = np.clip(
            self.actions,
            -self.cfg["policy"]["normalization"]["clip_actions"],
            self.cfg["policy"]["normalization"]["clip_actions"],
        )
        self.dof_targets[:] = self.default_dof_pos
        self.dof_targets[11:] += self.cfg["policy"]["control"]["action_scale"] * self.actions

        return self.dof_targets
