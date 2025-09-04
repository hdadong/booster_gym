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
            self.obs_mean = np.array([0.02070615254342556, -0.003988724201917648, -0.99067622423172, -0.0026212516240775585, 0.0008963909931480885, -0.0007227226160466671, -0.008572239428758621, 0.001339587732218206, 0.0010094484314322472, 0.0011648574145510793, -0.0008712905109860003, -0.2765815556049347, 0.11100117862224579, 0.0036469735205173492, 0.3967839479446411, -0.06536762416362762, -0.08178460597991943, -0.32444992661476135, -0.06049433723092079, 0.10475979000329971, 0.4057130813598633, -0.042936891317367554, 0.08518074452877045, -0.0013346055056899786, 0.0004165724094491452, 0.0005608489154838026, 0.004062098450958729, -0.002379050711169839, -0.0006663111271336675, -0.001763607608154416, -0.0001818236632971093, 0.00011723612260539085, 0.0041565741412341595, -0.0018892193911597133, 0.0004011081764474511, -0.2856055498123169, 0.22007901966571808, -0.0471050962805748, 0.1700993925333023, -0.007412885781377554, -0.053164441138505936, -0.3298279047012329, -0.16256412863731384, 0.15531836450099945, 0.17889828979969025, -0.014540123753249645, 0.05804278701543808], dtype=np.float32)
            self.obs_std = np.array([0.08304113894701004, 0.07659906893968582, 0.05850343406200409, 0.5974513292312622, 0.6095466017723083, 0.7913908362388611, 0.5596552491188049, 0.44567710161209106, 0.5536420345306396, 0.6739206314086914, 0.6729293465614319, 0.19840645790100098, 0.11114466190338135, 0.12484659999608994, 0.26809003949165344, 0.175664484500885, 0.16365687549114227, 0.18303155899047852, 0.11394775658845901, 0.12808428704738617, 0.24776805937290192, 0.1745113581418991, 0.16207487881183624, 0.21933116018772125, 0.16685688495635986, 0.21554428339004517, 0.3250788152217865, 0.42114999890327454, 0.4420636296272278, 0.21787294745445251, 0.16565747559070587, 0.21723207831382751, 0.31946757435798645, 0.42436477541923523, 0.4439714848995209, 0.2975117564201355, 0.2875019311904907, 0.22364921867847443, 0.5251239538192749, 0.4116112291812897, 0.37620288133621216, 0.2896808087825775, 0.2927449643611908, 0.23319095373153687, 0.5099384188652039, 0.40513837337493896, 0.3728852868080139], dtype=np.float32)
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
