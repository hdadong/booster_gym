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
            self.obs_mean = np.array([-0.0073019214905798435, -0.014048554934561253, -0.9756643176078796, -0.02841738425195217, 0.06251026690006256, 0.019795427098870277, -0.012802304700016975, 0.013627313077449799, -0.005408404860645533, 0.0031912841368466616, -0.00035221216967329383, -0.03359220176935196, 0.07374877482652664, -0.03155253827571869, 0.09759850054979324, -0.004320826847106218, -0.1747054159641266, -0.1461101919412613, -0.1312132030725479, 0.02854268066585064, 0.17478466033935547, 0.04533753916621208, 0.1889352947473526, -0.0029237177222967148, 0.00034472934203222394, 0.0022236763034015894, 0.001353084808215499, -0.00018812654889188707, 0.0031960206106305122, -0.0019885385408997536, -0.0007625644793733954, 0.00048210425302386284, 0.000944962608627975, -0.0008301177877001464, -0.002693735295906663, -0.07475382089614868, 0.13747434318065643, -0.049524515867233276, -0.10813529789447784, 0.017400842159986496, -0.3302297294139862, -0.15412189066410065, -0.18510174751281738, 0.04425284266471863, -0.023171441629529, 0.08653438091278076, 0.3440905511379242], dtype=np.float32)
            self.obs_std = np.array([0.15334376692771912, 0.12010559439659119, 0.1112213134765625, 0.7597456574440002, 0.9751726388931274, 0.8421534299850464, 0.5700262784957886, 0.4529842436313629, 0.5590410828590393, 0.6729281544685364, 0.6704978346824646, 0.1754000037908554, 0.1112506315112114, 0.13943631947040558, 0.27514657378196716, 0.1915281116962433, 0.16213157773017883, 0.21742625534534454, 0.11230470985174179, 0.14582817256450653, 0.3097410202026367, 0.20012924075126648, 0.15941372513771057, 0.21490293741226196, 0.16623646020889282, 0.24808825552463531, 0.33670657873153687, 0.4176212549209595, 0.42961254715919495, 0.24341876804828644, 0.1660662591457367, 0.25404515862464905, 0.36896011233329773, 0.44548529386520386, 0.42354902625083923, 0.2815631628036499, 0.3278980553150177, 0.24299441277980804, 0.5027923583984375, 0.4549441933631897, 0.423033207654953, 0.3241385817527771, 0.3077942132949829, 0.2513227164745331, 0.5480690002441406, 0.4808087646961212, 0.4197498857975006], dtype=np.float32)
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
