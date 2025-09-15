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
            self.obs_mean = np.array([0.009031126275658607, -0.016382431611418724, -0.9891802072525024, -0.00872659683227539, 0.013422805815935135, 0.006895416881889105, -0.009572474285960197, 0.004453505389392376, 0.0025231600739061832, 0.0016315599204972386, -0.0007586395367980003, -0.15197741985321045, 0.08152750879526138, 0.03252032771706581, 0.2603713572025299, -0.03540874272584915, -0.10718420147895813, -0.280624121427536, -0.07401532679796219, 0.11326964944601059, 0.3509751558303833, -0.023814240470528603, 0.11530981212854385, -0.001267215353436768, 0.0003544245264492929, 0.0012045601615682244, 0.0025766307953745127, -0.0011140760034322739, 0.0007113712490536273, -0.001744314911775291, -0.0006333124474622309, 0.0002338740014238283, 0.003113699145615101, -0.0012560066534206271, -0.00023043151304591447, -0.17397478222846985, 0.18533770740032196, -0.007495585363358259, 0.05048811063170433, 0.02727731503546238, -0.14823181927204132, -0.28480008244514465, -0.15944524109363556, 0.15037065744400024, 0.14384780824184418, -0.008190838620066643, 0.1560172438621521], dtype=np.float32)
            self.obs_std = np.array([0.10424286872148514, 0.08647273480892181, 0.07300852239131927, 0.6295334696769714, 0.7084785103797913, 0.7537087798118591, 0.5610443353652954, 0.4461487829685211, 0.5548593997955322, 0.6724987030029297, 0.6711719036102295, 0.22257505357265472, 0.11295650899410248, 0.13051511347293854, 0.3227432668209076, 0.19360722601413727, 0.172745481133461, 0.19197791814804077, 0.11043481528759003, 0.12991833686828613, 0.28507000207901, 0.18615572154521942, 0.16784152388572693, 0.2103573977947235, 0.1641899198293686, 0.21939510107040405, 0.3255978226661682, 0.4150502681732178, 0.4362732768058777, 0.22102493047714233, 0.16131311655044556, 0.22060595452785492, 0.3336244821548462, 0.43883219361305237, 0.4367797374725342, 0.3019019663333893, 0.3033663332462311, 0.2266508936882019, 0.5368690490722656, 0.42949849367141724, 0.4196316599845886, 0.2959946095943451, 0.28700393438339233, 0.22380200028419495, 0.5229847431182861, 0.42911967635154724, 0.41014716029167175], dtype=np.float32)
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
