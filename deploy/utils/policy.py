import numpy as np
import torch

from torch.distributions import Normal
import torch
import torch.nn.functional as F

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
            self.policy = torch.jit.load(self.cfg["policy"]["policy_path"])
            self.policy.eval()
            self.obs_mean = np.array([3.0357199e-02, -9.8874830e-03, -9.9206269e-01, -1.6514524e-03,
       -4.7709481e-04,  4.9726330e-03, -8.0190925e-03,  1.9653260e-03,
        9.3764812e-04,  1.0440621e-03, -8.6771068e-04, -2.8857273e-01,
        1.0588542e-01, -4.2704106e-03,  4.4549558e-01, -9.7549878e-02,
       -9.3825005e-02, -3.7460104e-01, -6.9362082e-02,  1.1318316e-01,
        4.6367481e-01, -5.4922726e-02,  9.6331209e-02, -1.8595024e-03,
        4.2197056e-04,  4.2550630e-04,  4.4615846e-03, -2.5539331e-03,
       -2.4634821e-04, -2.2772530e-03, -1.4257306e-04,  1.6598843e-04,
        4.5325644e-03, -1.0988067e-03,  2.8125438e-04, -2.9834160e-01,
        2.0448740e-01, -4.7801007e-02,  2.2000553e-01, -1.3439683e-02,
       -7.8917868e-02, -3.7099758e-01, -1.6228448e-01,  1.5532517e-01,
        2.3750339e-01, -1.8590566e-02,  8.0623433e-02], dtype=np.float32)
            self.obs_std = np.array([0.07724574, 0.07971249, 0.05202492, 0.60347027, 0.5575331 ,
       0.7170951 , 0.55858004, 0.4437893 , 0.5515236 , 0.6730663 ,
       0.67216027, 0.1643191 , 0.11209376, 0.11647055, 0.2240846 ,
       0.17157657, 0.15722944, 0.17174241, 0.1140605 , 0.11706857,
       0.20969345, 0.17119823, 0.15554875, 0.19883834, 0.15762271,
       0.19464833, 0.296113  , 0.40685037, 0.4213234 , 0.203172  ,
       0.15646084, 0.1969148 , 0.2980072 , 0.41440678, 0.4232227 ,
       0.25161865, 0.2592898 , 0.19867057, 0.48128015, 0.4113223 ,
       0.34822154, 0.26590976, 0.25794333, 0.20328748, 0.47240415,
       0.40385327, 0.33891624 ], dtype=np.float32)
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

        self.obs[0:3] = projected_gravity * self.cfg["policy"]["normalization"]["gravity"]
        self.obs[3:6] = base_ang_vel * self.cfg["policy"]["normalization"]["ang_vel"]
        self.obs[6] = (
            self.smoothed_commands[0] * self.cfg["policy"]["normalization"]["lin_vel"] * (self.gait_frequency > 1.0e-8)
        )
        self.obs[7] = (
            self.smoothed_commands[1] * self.cfg["policy"]["normalization"]["lin_vel"] * (self.gait_frequency > 1.0e-8)
        )
        self.obs[8] = (
            self.smoothed_commands[2] * self.cfg["policy"]["normalization"]["ang_vel"] * (self.gait_frequency > 1.0e-8)
        )
        self.obs[9] = np.cos(2 * np.pi * self.gait_process) * (self.gait_frequency > 1.0e-8)
        self.obs[10] = np.sin(2 * np.pi * self.gait_process) * (self.gait_frequency > 1.0e-8)
        self.obs[11:23] = (dof_pos - self.default_dof_pos)[11:] * self.cfg["policy"]["normalization"]["dof_pos"]
        self.obs[23:35] = dof_vel[11:] * self.cfg["policy"]["normalization"]["dof_vel"]
        self.obs[35:47] = self.actions
        self.obs = (self.obs - self.obs_mean) / self.obs_std
        dist = self.policy(torch.tensor(self.obs).unsqueeze(0))        # -> shape [1, 2*A]
        self.actions[:]= self.normalizer.mode(dist).detach().numpy()     # -> shape [1, A]
        self.actions[:] = np.clip(
            self.actions,
            -self.cfg["policy"]["normalization"]["clip_actions"],
            self.cfg["policy"]["normalization"]["clip_actions"],
        )
        self.dof_targets[:] = self.default_dof_pos
        self.dof_targets[11:] += self.cfg["policy"]["control"]["action_scale"] * self.actions

        return self.dof_targets
