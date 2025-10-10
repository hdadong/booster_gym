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
            self.policy = torch.jit.load(self.cfg["policy"]["policy_path"], map_location="cpu")
            self.policy.eval()
            self.obs_mean = np.array([ 2.21614912e-02, -6.62935758e-03, -9.89770353e-01, -1.84123067e-03,
       -1.05458626e-03, -3.48594203e-03, -1.35526909e-02,  5.11697785e-04,
        1.54562516e-03,  1.50413706e-03, -7.06638210e-04, -3.33513528e-01,
        7.78222755e-02, -4.17490155e-02,  5.26312172e-01, -1.20869458e-01,
       -1.03857972e-01, -4.04644668e-01, -7.41391554e-02,  7.49955326e-02,
        5.31088114e-01, -9.97376591e-02,  1.02060944e-01, -2.47895136e-03,
        3.20627441e-04,  8.47006449e-04,  4.78907768e-03, -2.23732460e-03,
        8.21477210e-04, -2.81094946e-03, -2.26590651e-04,  3.80213809e-04,
        4.62141959e-03, -1.09391217e-03, -7.45175115e-04, -3.37301910e-01,
        1.75762802e-01, -9.18873921e-02,  2.82788008e-01, -3.28020975e-02,
       -1.21201731e-01, -4.13269728e-01, -1.69798374e-01,  1.24133684e-01,
        2.68622637e-01, -9.55357030e-02,  1.15391515e-01], dtype=np.float32)
            self.obs_std = np.array([0.0956538 , 0.09019899, 0.07090962, 0.62016124, 0.6689715 ,
       0.76529443, 0.5644678 , 0.4491155 , 0.55393225, 0.6727647 ,
       0.671542  , 0.16333263, 0.11988021, 0.15301014, 0.2176434 ,
       0.17444059, 0.1672211 , 0.17699417, 0.12580965, 0.14642641,
       0.22388743, 0.17983808, 0.16669112, 0.20540868, 0.16352113,
       0.21124631, 0.29779342, 0.42500436, 0.44682735, 0.21028578,
       0.16216691, 0.2101046 , 0.3005955 , 0.44228187, 0.44563502,
       0.2673667 , 0.2923741 , 0.23901007, 0.5012833 , 0.43059075,
       0.40789968, 0.2820594 , 0.2805095 , 0.24342726, 0.507204  ,
       0.42896807, 0.40686968], dtype=np.float32)
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
