import numpy as np
import torch

from torch.distributions import Normal
import torch
import torch.nn.functional as F

class minmaxnormalizer():
    def __init__(self):
        self.device='cpu'
        # Normalization limits
        self.obs_limit_min = torch.full((1, 47), -1.0, device=self.device)  # Min values
        self.obs_limit_max = torch.full((1, 47), 1.0, device=self.device)   # Max values

        indice_dof_pos = [i for i in range(11, 23, 1)]
        self.obs_limit_min[:, indice_dof_pos] = torch.tensor([-1.8, -0.3, -1.0, 0.0, -0.87, -0.44, -1.8, -1.57, -1.0, 0.0, -0.87, -0.44], device=self.device) - 0.25
        self.obs_limit_max[:, indice_dof_pos] = torch.tensor([1.57, 1.57, 1.0, 2.34, 0.35, 0.44, 1.57, 0.3, 1.0, 2.34, 0.35, 0.44], device=self.device) + 0.25

        # indice_dof_pos = [i for i in range(60, 72, 1)]
        # self.obs_limit_min[:, indice_dof_pos] = torch.tensor([-1.8, -0.3, -1.0, 0.0, -0.87, -0.44, -1.8, -1.57, -1.0, 0.0, -0.87, -0.44], device=self.device) - 0.25
        # self.obs_limit_max[:, indice_dof_pos] = torch.tensor([1.57, 1.57, 1.0, 2.34, 0.35, 0.44, 1.57, 0.3, 1.0, 2.34, 0.35, 0.44], device=self.device) + 0.25

        # self._forward_vel_idx = 53
        # self.obs_limit_min[:, self._forward_vel_idx] = 0.0
        # self.obs_limit_max[:, self._forward_vel_idx] = 0.5

        # self._y_vel_idx = 58
        # self.obs_limit_min[:, self._y_vel_idx] = -0.5
        # self.obs_limit_max[:, self._y_vel_idx] = 0.5

        # self._z_vel_idx = 59
        # self.obs_limit_min[:, self._z_vel_idx] = -0.5
        # self.obs_limit_max[:, self._z_vel_idx] = 0.5

        self._roll_rate_idx = 3
        self._pitch_rate_idx = 4
        self._turn_rate_idx = 5
        self._rpy_rate_idxs = [i for i in range(self._roll_rate_idx, self._turn_rate_idx+1, 1)]
        self.obs_limit_min[:, self._rpy_rate_idxs] = torch.tensor([-1.5, -1.5, -1.5], device=self.device)
        self.obs_limit_max[:, self._rpy_rate_idxs] = torch.tensor([1.5, 1.5, 1.5], device=self.device)

        # self._priv_roll_rate_idx = 47
        # self._priv_pitch_rate_idx = 48
        # self._priv_turn_rate_idx = 49
        # self._priv_rpy_rate_idxs = [i for i in range(self._priv_roll_rate_idx, self._priv_turn_rate_idx+1, 1)]
        # self.obs_limit_min[:, self._priv_rpy_rate_idxs] = torch.tensor([-1.5, -1.5, -1.5], device=self.device)
        # self.obs_limit_max[:, self._priv_rpy_rate_idxs] = torch.tensor([1.5, 1.5, 1.5], device=self.device)

        self._qd_idxs = [i for i in range(23, 35, 1)]
        self.obs_limit_min[:, self._qd_idxs] = -20.0
        self.obs_limit_max[:, self._qd_idxs] = 20

        # self._qd_idxs = [i for i in range(72, 84, 1)]
        # self.obs_limit_min[:, self._qd_idxs] = -20.0
        # self.obs_limit_max[:, self._qd_idxs] = 20

        # indice_height = 84
        # self.obs_limit_min[:, indice_height] = 0.0
        # self.obs_limit_max[:, indice_height] = 0.8

    def normalize_obs(self, obs):
        # Normalize observation
        normalized_obs = 2 * (obs - self.obs_limit_min) / (self.obs_limit_max - self.obs_limit_min) - 1
        return normalized_obs

    def denormalize_obs(self, normalize_obs):
        obs = (normalize_obs + 1)*(self.obs_limit_max - self.obs_limit_min)/2 + self.obs_limit_min
        return obs
        

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
    def __init__(self, cfg):
        try:
            self.cfg = cfg
            self.policy = torch.jit.load(self.cfg["policy"]["policy_path"], map_location="cpu")
            self.policy.eval()
            self.obs_minmax_normalizer = minmaxnormalizer()
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
        self.actions = np.zeros(self.cfg["policy"]["num_actions"], dtype=np.float32)
        self.policy_interval = self.cfg["common"]["dt"] * self.cfg["policy"]["control"]["decimation"]

    def inference(self, time_now, dof_pos, dof_vel, base_ang_vel, projected_gravity, vx, vy, vyaw):
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
        # self.obs[11:23] = (dof_pos - self.default_dof_pos)[11:] * self.cfg["policy"]["normalization"]["dof_pos"]
        # self.obs[23:35] = dof_vel[11:] * self.cfg["policy"]["normalization"]["dof_vel"]
        self.obs[35:47] = self.actions
        
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
