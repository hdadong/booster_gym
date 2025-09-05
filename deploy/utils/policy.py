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
            self.obs_mean = np.array([0.022186677902936935, -0.0006525471108034253, -0.9909012317657471, 0.0005314740119501948, 0.0035995023790746927, -0.002715923124924302, -0.01117838267236948, 0.002077909652143717, -0.001199345919303596, 0.0013910839334130287, -0.0007864394574426115, -0.31336620450019836, 0.09931553900241852, -0.03169013559818268, 0.47570329904556274, -0.09791312366724014, -0.10695111006498337, -0.35719576478004456, -0.04998337849974632, 0.0739758238196373, 0.47193002700805664, -0.07997670024633408, 0.09410803765058517, -0.001827322063036263, 0.0004997671931050718, 0.0007723793969489634, 0.004091240465641022, -0.0016792748356238008, 0.0008124201558530331, -0.0019413065165281296, -0.00033273216104134917, 0.0005212977412156761, 0.004297134932130575, -0.0010172909824177623, -0.0003894645196851343, -0.3188897967338562, 0.20384632050991058, -0.07987704128026962, 0.24874968826770782, -0.0272750873118639, -0.11538292467594147, -0.3651694357395172, -0.164378821849823, 0.1294659525156021, 0.22204749286174774, -0.0762973353266716, 0.10642662644386292], dtype=np.float32)
            self.obs_std = np.array([0.09313985705375671, 0.08376897871494293, 0.066642165184021, 0.6048629283905029, 0.6819936633110046, 0.7758254408836365, 0.5551866292953491, 0.4420708417892456, 0.5494083762168884, 0.6703413128852844, 0.6691873669624329, 0.17932972311973572, 0.11380848288536072, 0.14323994517326355, 0.24519357085227966, 0.17862896621227264, 0.17272882163524628, 0.18420162796974182, 0.11466478556394577, 0.14316870272159576, 0.24743086099624634, 0.17938797175884247, 0.17003004252910614, 0.21688435971736908, 0.16632884740829468, 0.21417978405952454, 0.3275333344936371, 0.44217121601104736, 0.4494328498840332, 0.22164849936962128, 0.16401715576648712, 0.2126847505569458, 0.331369549036026, 0.4487495422363281, 0.44665467739105225, 0.2887703478336334, 0.2882082760334015, 0.23130400478839874, 0.5290573239326477, 0.4270688593387604, 0.4131907820701599, 0.29240086674690247, 0.29121553897857666, 0.24406205117702484, 0.5409182906150818, 0.43175259232521057, 0.41589459776878357], dtype=np.float32)
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
