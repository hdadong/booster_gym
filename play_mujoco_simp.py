import os
os.environ["MUJOCO_GL"] = 'egl'
import sys
import glob
import yaml
import select
import argparse
import numpy as np
import torch
import mujoco, mujoco.viewer
from utils.model import *
from torch.distributions import Normal

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


def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
    parser.add_argument("--checkpoint", type=str, help="Path of model checkpoint to load. Overrides config file if provided.")
    args = parser.parse_args()
    cfg_file = os.path.join("envs", "{}.yaml".format(args.task))
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
    if args.checkpoint is not None:
        cfg["basic"]["checkpoint"] = args.checkpoint
    policy = torch.jit.load(cfg["basic"]["checkpoint"])

    #policy = load_policy(cfg["basic"]["checkpoint"])
    #policy = torch.compile(policy)

    mj_model = mujoco.MjModel.from_xml_path(cfg["asset"]["mujoco_file"])
    mj_model.opt.timestep = cfg["sim"]["dt"]
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    default_dof_pos = np.zeros(mj_model.nu, dtype=np.float32)
    dof_stiffness = np.zeros(mj_model.nu, dtype=np.float32)
    dof_damping = np.zeros(mj_model.nu, dtype=np.float32)
    for i in range(mj_model.nu):
        found = False
        for name in cfg["init_state"]["default_joint_angles"].keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                default_dof_pos[i] = cfg["init_state"]["default_joint_angles"][name]
                found = True
        if not found:
            default_dof_pos[i] = cfg["init_state"]["default_joint_angles"]["default"]

        found = False
        for name in cfg["control"]["stiffness"].keys():
            if name in mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i):
                dof_stiffness[i] = cfg["control"]["stiffness"][name]
                dof_damping[i] = cfg["control"]["damping"][name]
                found = True
        if not found:
            raise ValueError(f"PD gain of joint {mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)} were not defined")
    mj_data.qpos = np.concatenate(
        [
            np.array(cfg["init_state"]["pos"], dtype=np.float32),
            np.array(cfg["init_state"]["rot"][3:4] + cfg["init_state"]["rot"][0:3], dtype=np.float32),
            default_dof_pos,
        ]
    )
    mujoco.mj_forward(mj_model, mj_data)
    normalizer = NormalTanhDistribution()
    actions = np.zeros((cfg["env"]["num_actions"]), dtype=np.float32)
    dof_targets = np.zeros(default_dof_pos.shape, dtype=np.float32)
    gait_frequency = gait_process = 0.0
    lin_vel_x = lin_vel_y = ang_vel_yaw = 0.0
    it = 0
    obs_mean = np.array([ 1.37172379e-02,  3.53139057e-03, -9.94338930e-01, -1.84248842e-03,
  2.19509332e-03,  1.61452889e-02, -2.15455610e-02,  1.72599647e-02,
  2.25300975e-02,  1.21383031e-03,  1.26357860e-04, -3.57616156e-01,
  1.28460124e-01,  1.29287096e-03,  5.63962400e-01, -2.10249782e-01,
 -1.29617706e-01, -3.64434123e-01, -6.98471442e-02,  6.12215213e-02,
  5.03367066e-01, -1.39685035e-01,  1.11739248e-01, -4.81613533e-04,
  7.59325642e-03,  1.22257574e-02,  9.53489356e-03,  1.83292909e-03,
  1.41830593e-02, -3.63218668e-03, -1.51777023e-03, -5.86699834e-03,
  8.39451049e-03,  1.24912960e-02, -1.54671837e-02, -1.71308175e-01,
  1.63175151e-01, -1.19749513e-02, -6.25355542e-02, -1.17908381e-02,
 -1.55036300e-01, -1.70031399e-01, -1.19875036e-01,  7.87013546e-02,
 -1.21435225e-01,  3.54985036e-02,  1.06634386e-01], dtype=np.float32)
    obs_std = np.array([0.07606246, 0.07354303, 0.04887975, 0.72584885, 0.71435946, 0.6610478 ,
 0.5406346 , 0.44328502, 0.5325911 , 0.6731312 , 0.6722322 , 0.22061047,
 0.11689425, 0.12047654, 0.36127594, 0.22395581, 0.14686853, 0.22083732,
 0.13550289, 0.12429432, 0.34629634, 0.20072974, 0.14736265, 2.3733208 ,
 1.693957  , 2.1185708 , 3.9716487 , 4.4172387 , 4.1508403 , 2.3845325 ,
 1.7062409 , 2.1494184 , 3.7765918 , 4.2339725 , 4.126802  , 0.32139105,
 0.33337796, 0.20529182, 0.60802674, 0.43839827, 0.4217969 , 0.32762653,
 0.36651224, 0.21787657, 0.5718411 , 0.42906114, 0.40963235], dtype=np.float32)
    obs_minmax_normalizer = minmaxnormalizer()
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.elevation = -20
        print(f"Set command (x, y, yaw): ")
        while viewer.is_running():
            if select.select([sys.stdin], [], [], 0)[0]:
                try:
                    parts = sys.stdin.readline().strip().split()
                    if len(parts) == 3:
                        lin_vel_x, lin_vel_y, ang_vel_yaw = map(float, parts)
                        if lin_vel_x == 0 and lin_vel_y == 0 and ang_vel_yaw == 0:
                            gait_frequency = 0
                        else:
                            gait_frequency = np.average(cfg["commands"]["gait_frequency"])
                        print(
                            f"Updated command to: x={lin_vel_x}, y={lin_vel_y}, yaw={ang_vel_yaw}\nSet command (x, y, yaw): ",
                            end="",
                        )
                    else:
                        raise ValueError
                except ValueError:
                    print("Invalid input. Enter three numeric values.\nSet command (x, y, yaw): ", end="")
            dof_pos = mj_data.qpos.astype(np.float32)[7:]
            dof_vel = mj_data.qvel.astype(np.float32)[6:]
            quat = mj_data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
            base_ang_vel = mj_data.sensor("angular-velocity").data.astype(np.float32)
            projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
            if it % cfg["control"]["decimation"] == 0:
                obs = np.zeros(cfg["env"]["num_observations"], dtype=np.float32)
                obs[0:3] = projected_gravity
                obs[3:6] = base_ang_vel
                obs[6] = lin_vel_x 
                obs[7] = lin_vel_y
                obs[8] = ang_vel_yaw
                obs[9] = np.cos(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
                obs[10] = np.sin(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
                obs[11:23] = dof_pos
                obs[23:35] = dof_vel
                obs[35:47] = actions
                obs = (obs - obs_mean) / obs_std
                obs_torch = torch.tensor(obs)

                #obs_torch_minmaxnorm = obs_minmax_normalizer.normalize_obs(obs_torch)
                dist = policy(obs_torch.unsqueeze(0))        # -> shape [1, 2*A]
                #actions = dist.detach().numpy()     # -> shape [1, A]

                actions = normalizer.mode(dist).detach().numpy()     # -> shape [1, A]
                actions[:] = np.clip(actions, -cfg["normalization"]["clip_actions"], cfg["normalization"]["clip_actions"])
                dof_targets[:] = default_dof_pos + cfg["control"]["action_scale"] * actions
            mj_data.ctrl = np.clip(
                dof_stiffness * (dof_targets - dof_pos) - dof_damping * dof_vel,
                mj_model.actuator_ctrlrange[:, 0],
                mj_model.actuator_ctrlrange[:, 1],
            )
            mujoco.mj_step(mj_model, mj_data)
            viewer.cam.lookat[:] = mj_data.qpos.astype(np.float32)[0:3]
            viewer.sync()
            it += 1
            gait_process = np.fmod(gait_process + cfg["sim"]["dt"] * gait_frequency, 1.0)
