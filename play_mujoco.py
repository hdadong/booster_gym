import os
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
    obs_mean = np.array([3.0357199e-02, -9.8874830e-03, -9.9206269e-01, -1.6514524e-03,
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
    obs_std = np.array([0.07724574, 0.07971249, 0.05202492, 0.60347027, 0.5575331 ,
       0.7170951 , 0.55858004, 0.4437893 , 0.5515236 , 0.6730663 ,
       0.67216027, 0.1643191 , 0.11209376, 0.11647055, 0.2240846 ,
       0.17157657, 0.15722944, 0.17174241, 0.1140605 , 0.11706857,
       0.20969345, 0.17119823, 0.15554875, 0.19883834, 0.15762271,
       0.19464833, 0.296113  , 0.40685037, 0.4213234 , 0.203172  ,
       0.15646084, 0.1969148 , 0.2980072 , 0.41440678, 0.4232227 ,
       0.25161865, 0.2592898 , 0.19867057, 0.48128015, 0.4113223 ,
       0.34822154, 0.26590976, 0.25794333, 0.20328748, 0.47240415,
       0.40385327, 0.33891624 ], dtype=np.float32)
    
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
                obs[0:3] = projected_gravity * cfg["normalization"]["gravity"]
                obs[3:6] = base_ang_vel * cfg["normalization"]["ang_vel"]
                obs[6] = lin_vel_x * cfg["normalization"]["lin_vel"]
                obs[7] = lin_vel_y * cfg["normalization"]["lin_vel"]
                obs[8] = ang_vel_yaw * cfg["normalization"]["ang_vel"]
                obs[9] = np.cos(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
                obs[10] = np.sin(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
                obs[11:23] = (dof_pos - default_dof_pos) * cfg["normalization"]["dof_pos"]
                obs[23:35] = dof_vel * cfg["normalization"]["dof_vel"]
                obs[35:47] = actions
                obs = (obs - obs_mean) / obs_std
                dist = policy(torch.tensor(obs).unsqueeze(0))        # -> shape [1, 2*A]
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
