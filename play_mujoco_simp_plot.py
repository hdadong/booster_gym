import os
os.environ["MUJOCO_GL"] = 'egl'
import sys
import glob
import yaml
import select
import argparse
import numpy as np
import torch
from utils.model import *
from torch.distributions import Normal
from datetime import datetime
import time
from typing import Optional
from pathlib import Path
import re

def setup_gl_backend(render: bool, gl: str | None):
    """根据参数设置 MuJoCo 的 GL 后端，必须在 import mujoco 前调用。"""
    if gl:                    # 手动覆盖
        os.environ["MUJOCO_GL"] = gl
    elif render:              # 需要窗口渲染
        os.environ["MUJOCO_GL"] = "glfw"
    else:                     # 纯 headless
        os.environ.setdefault("MUJOCO_GL", "egl")


def get_latest_policy_path(policy_dir: str) -> Optional[str]:
    """
    从目录中查找形如 policy_<number>.pt 的权重文件，按<number>数值取最新。
    若目录不存在或无匹配文件，则返回 None。
    """
    # 目录不存在
    try:
        names = os.listdir(policy_dir)
    except FileNotFoundError:
        return None

    pattern = re.compile(r'^policy_(\d+)\.pt$')
    candidates = []

    for name in names:
        m = pattern.match(name)
        if m:
            step = int(m.group(1))
            candidates.append((step, name))

    if not candidates:
        return None

    # 按数值排序，取最大
    candidates.sort(key=lambda x: x[0])
    latest_name = candidates[-1][1]
    return os.path.join(policy_dir, latest_name)

class minmaxnormalizer():
    def __init__(self):
        self.device='cpu'
        # Normalization limits
        self.obs_limit_min = torch.full((1, 87), -1.0, device=self.device)  # Min values
        self.obs_limit_max = torch.full((1, 87), 1.0, device=self.device)   # Max values

        self.q_idxs = [i for i in range(11, 23, 1)]
        self.obs_limit_min[:, self.q_idxs] = torch.tensor([-1.8, -0.3, -1.0, 0.0, -0.87, -0.44, -1.8, -1.57, -1.0, 0.0, -0.87, -0.44], device=self.device) - 0.25
        self.obs_limit_max[:, self.q_idxs] = torch.tensor([1.57, 1.57, 1.0, 2.34, 0.35, 0.44, 1.57, 0.3, 1.0, 2.34, 0.35, 0.44], device=self.device) + 0.25

        self.wm_q_idxs = [i for i in range(60, 72, 1)]
        self.obs_limit_min[:, self.wm_q_idxs] = torch.tensor([-1.8, -0.3, -1.0, 0.0, -0.87, -0.44, -1.8, -1.57, -1.0, 0.0, -0.87, -0.44], device=self.device) - 0.25
        self.obs_limit_max[:, self.wm_q_idxs] = torch.tensor([1.57, 1.57, 1.0, 2.34, 0.35, 0.44, 1.57, 0.3, 1.0, 2.34, 0.35, 0.44], device=self.device) + 0.25

        self.forward_vel_idx = 57
        self.obs_limit_min[:, self.forward_vel_idx] = -0.2
        self.obs_limit_max[:, self.forward_vel_idx] = 2.5

        self.y_vel_idx = 58
        self.obs_limit_min[:, self.y_vel_idx] = -0.5
        self.obs_limit_max[:, self.y_vel_idx] = 0.5

        self.z_vel_idx = 59
        self.obs_limit_min[:, self.z_vel_idx] = -0.5
        self.obs_limit_max[:, self.z_vel_idx] = 0.5

        self.roll_rate_idx = 3
        self.pitch_rate_idx = 4
        self.turn_rate_idx = 5
        self.rpy_rate_idxs = [i for i in range(self.roll_rate_idx, self.turn_rate_idx+1, 1)]
        self.obs_limit_min[:, self.rpy_rate_idxs] = torch.tensor([-1.5, -1.5, -1.5], device=self.device)
        self.obs_limit_max[:, self.rpy_rate_idxs] = torch.tensor([1.5, 1.5, 1.5], device=self.device)

        self.wm_roll_rate_idx = 47
        self.wm_pitch_rate_idx = 48
        self.wm_turn_rate_idx = 49
        self.wm_rpy_rate_idxs = [i for i in range(self.wm_roll_rate_idx, self.wm_turn_rate_idx+1, 1)]
        self.obs_limit_min[:, self.wm_rpy_rate_idxs] = torch.tensor([-1.5, -1.5, -1.5], device=self.device)
        self.obs_limit_max[:, self.wm_rpy_rate_idxs] = torch.tensor([1.5, 1.5, 1.5], device=self.device)

        self.qd_idxs = [i for i in range(23, 35, 1)]
        self.obs_limit_min[:, self.qd_idxs] = -20.0
        self.obs_limit_max[:, self.qd_idxs] = 20

        self.wm_qd_idxs = [i for i in range(72, 84, 1)]
        self.obs_limit_min[:, self.wm_qd_idxs] = -20.0
        self.obs_limit_max[:, self.wm_qd_idxs] = 20

        self.wm_height_idx = 84
        self.obs_limit_min[:, self.wm_height_idx] = 0.0
        self.obs_limit_max[:, self.wm_height_idx] = 0.8


        self.wm_gravity_idxs = [i for i in range(50, 53, 1)]
        self.wm_quat_idxs = [i for i in range(53, 57, 1)]
        self.wm_base_vel_idxs = [i for i in range(57, 60, 1)]
        self.wm_gait_process_idx = 85
        self.wm_gait_frequency_idx = 86
        
        self.priv_rpy_rate_idxs = self.wm_rpy_rate_idxs
        self.priv_gravity_idxs = self.wm_gravity_idxs
        self.priv_base_lin_vel_idxs =  [i for i in range(53, 56, 1)]
        self.priv_global_ang_vel_idxs =  [i for i in range(56, 59, 1)]
        self.priv_q_idxs = [i for i in range(59, 71, 1)]
        self.priv_qd_idxs = [i for i in range(71, 83, 1)]
        self.priv_height_idx = 83


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

def run_mujoco(policy, cfg, render: bool = False):
    import mujoco  # 此时已设置好 MUJOCO_GL
    if render:
        import mujoco.viewer

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
    gait_process = 0.0
    gait_frequency = np.average(cfg["commands"]["gait_frequency"])
    lin_vel_y = ang_vel_yaw = 0.0
    lin_vel_x = 0.1
    it = 0
    step = 0
    num_envs = 1
    obs_minmax_normalizer = minmaxnormalizer()

    data_dict =  {
            'state': [],
            'priv_state': [],
            'wm_state': [],
            'actions': [],
            'torques': [],
            'contacts': [],
            'rewards': [],
            'timestamps': []
    } 
    data_buffers = [
        {key: [] for key in data_dict}  # Create a new dictionary with the same structure
        for _ in range(num_envs)
    ]
    if render:
        viewer =  mujoco.viewer.launch_passive(mj_model, mj_data)
        viewer.cam.elevation = -20
    while True:
        base_pos = mj_data.qpos.astype(np.float32)[:3]
        dof_pos = mj_data.qpos.astype(np.float32)[7:]
        dof_vel = mj_data.qvel.astype(np.float32)[6:]
        quat = mj_data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
        quat_wxyz = mj_data.sensor("orientation").data.astype(np.float32)

        base_ang_vel = mj_data.sensor("angular-velocity").data.astype(np.float32)
        base_lin_vel = mj_data.sensor("linear-velocity").data.astype(np.float32)

        projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        ang_vel_global = rotate(quat, base_ang_vel)
        if it % cfg["control"]["decimation"] == 0:
            if step != 0:
                data_buffers[0]['state'].append(state)
                data_buffers[0]['wm_state'].append(wm_state[0])
                data_buffers[0]['priv_state'].append(priv_state)
                data_buffers[0]['actions'].append(actions[0])
                data_buffers[0]['torques'].append(torque)
                data_buffers[0]['contacts'].append([0.0,0.0])
                data_buffers[0]['rewards'].append(0)
                data_buffers[0]['timestamps'].append(0)
            
            state = np.zeros(cfg["env"]["num_observations"], dtype=np.float32)
            state[0:3] = projected_gravity
            state[3:6] = base_ang_vel
            state[6] = lin_vel_x 
            state[7] = lin_vel_y
            state[8] = ang_vel_yaw
            state[9] = np.cos(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
            state[10] = np.sin(2 * np.pi * gait_process) * (gait_frequency > 1.0e-8)
            state[11:23] = dof_pos
            state[23:35] = dof_vel
            state[35:47] = actions

            obs_torch = torch.tensor(state)

            wm_state = np.zeros(87)
            wm_state[:state.shape[0]] = state
            wm_state[obs_minmax_normalizer.wm_rpy_rate_idxs] = base_ang_vel
            wm_state[obs_minmax_normalizer.wm_gravity_idxs] = projected_gravity
            wm_state[obs_minmax_normalizer.wm_quat_idxs] = quat_wxyz
            wm_state[obs_minmax_normalizer.wm_base_vel_idxs] = base_lin_vel
            wm_state[obs_minmax_normalizer.wm_q_idxs] = dof_pos
            wm_state[obs_minmax_normalizer.wm_qd_idxs] = dof_vel
            wm_state[obs_minmax_normalizer.wm_height_idx] = base_pos[2]
            wm_state[obs_minmax_normalizer.wm_gait_process_idx] = gait_process
            wm_state[obs_minmax_normalizer.wm_gait_frequency_idx] = gait_frequency
            wm_state = torch.tensor(wm_state)
            wm_state = obs_minmax_normalizer.normalize_obs(wm_state)

            priv_state = np.zeros(84)
            priv_state[:state.shape[0]] = state
            priv_state[obs_minmax_normalizer.priv_rpy_rate_idxs] = base_ang_vel
            priv_state[obs_minmax_normalizer.priv_gravity_idxs] = projected_gravity
            priv_state[obs_minmax_normalizer.priv_base_lin_vel_idxs] = base_lin_vel
            priv_state[obs_minmax_normalizer.priv_global_ang_vel_idxs] = ang_vel_global
            priv_state[obs_minmax_normalizer.priv_q_idxs] = dof_pos
            priv_state[obs_minmax_normalizer.priv_qd_idxs] = dof_vel
            priv_state[obs_minmax_normalizer.priv_height_idx] = base_pos[2]
            priv_state = torch.tensor(priv_state)
            dist = policy(obs_torch.unsqueeze(0))        # -> shape [1, 2*A]
            #actions = dist.detach().numpy()     # -> shape [1, A]

            actions = normalizer.mode(dist).detach().numpy()     # -> shape [1, A]
            actions[:] = np.clip(actions, -cfg["normalization"]["clip_actions"], cfg["normalization"]["clip_actions"])
            dof_targets[:] = default_dof_pos + cfg["control"]["action_scale"] * actions
            step += 1
        torque = np.clip(
            dof_stiffness * (dof_targets - dof_pos) - dof_damping * dof_vel,
            mj_model.actuator_ctrlrange[:, 0],
            mj_model.actuator_ctrlrange[:, 1],
        )
        mj_data.ctrl = torque
        mujoco.mj_step(mj_model, mj_data)
        it += 1
        gait_process = np.fmod(gait_process + cfg["sim"]["dt"] * gait_frequency, 1.0)
        if step == 1000 or base_pos[2] < 0.3:
            break
        if render:    
            viewer.cam.lookat[:] = mj_data.qpos.astype(np.float32)[0:3]
            viewer.sync()
    if render:
        viewer.close()
    return step, data_buffers
def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c



def rotate(quat, vec):
  """Rotates a vector vec by a unit quaternion quat.

  Args:
    vec: (3,) a vector
    quat: (4,) a quaternion

  Returns:
    ndarray(3) containing vec rotated by quat.
  """
  s, u = quat[-1], quat[:-1]
  r = 2 * (np.dot(u, vec) * u) + (s * s - np.dot(u, u)) * vec
  r = r + 2 * s * np.cross(u, vec)
  return r

# python play_mujoco_simp_train.py --task=T1 --render
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
    parser.add_argument("--render", action="store_true",
                        help="open gui render")
    args = parser.parse_args()
    cfg_file = os.path.join("envs", "{}.yaml".format(args.task))
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    base_data_dir = '/home/admin123/transformer_worldmodel/ssrl/sac_data/20251020_140739/real_data_dir/data_20251020_145110'
    policy_path = '/home/admin123/booster_gym/policy_13_real.pt'
    env_id = 0

    print("load the latest policy", policy_path)
    policy = torch.jit.load(policy_path)
    policy_path = None

    # 创建新的数据文件夹
    total_step = 0
    episode_num = 0

    episode_step, data_buffers = run_mujoco(policy, cfg, render=args.render)
    # 保存数据到 .npz 文件
    npz_filename = os.path.join(
        base_data_dir,
        f'mujoco_{env_id}_data_{episode_num}.npz',
    )
    state_array = np.array(data_buffers[env_id]['state'], dtype=np.float32)
    wm_state_array = np.array(data_buffers[env_id]['wm_state'], dtype=np.float32)
    priv_state_array = np.array(data_buffers[env_id]['priv_state'], dtype=np.float32)
    actions_array = np.array(data_buffers[env_id]['actions'], dtype=np.float32)
    torques_array = np.array(data_buffers[env_id]['torques'], dtype=np.float32)
    contact_array = np.array(data_buffers[env_id]['contacts'], dtype=np.float32)
    rewards_array = np.array(data_buffers[env_id]['rewards'], dtype=np.float32)
    timestamps_array = np.array(data_buffers[env_id]['timestamps'], dtype=np.float64)
    np.savez_compressed(
        npz_filename,
        states=state_array,
        wm_states=wm_state_array,
        priv_states=priv_state_array,
        actions=actions_array,
        torques=torques_array,
        contacts=contact_array,
        rewards=rewards_array,
        timestamps=timestamps_array,
    )
    print("store the data", npz_filename, "size is ", episode_step)
    # test_load = np.load(npz_filename)
    # for key in test_load:
    #     print(f"{key}:")
    #     print(test_load[key])
    data_buffers[env_id]['state'].clear()
    data_buffers[env_id]['wm_state'].clear()
    data_buffers[env_id]['priv_state'].clear()
    data_buffers[env_id]['actions'].clear()
    data_buffers[env_id]['torques'].clear()
    data_buffers[env_id]['contacts'].clear()
    data_buffers[env_id]['rewards'].clear()
    data_buffers[env_id]['timestamps'].clear()



