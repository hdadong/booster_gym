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
from typing import Any, Dict, Optional, Union, Sequence, Tuple
import jax.numpy as jp
from datetime import datetime
import time
from typing import Optional
from pathlib import Path
import re
import jax
def setup_gl_backend(render: bool, gl: str | None):
    """根据参数设置 MuJoCo 的 GL 后端，必须在 import mujoco 前调用。"""
    if gl:                    # 手动覆盖
        os.environ["MUJOCO_GL"] = gl
    elif render:              # 需要窗口渲染
        os.environ["MUJOCO_GL"] = "glfw"
    else:                     # 纯 headless
        os.environ.setdefault("MUJOCO_GL", "egl")

def quat_rotate_inverse(q, v):
    q_w = q[-1]
    q_vec = q[:3]
    a = v * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, v) * (q_w * 2.0)
    c = q_vec * (np.dot(q_vec, v) * 2.0)
    return a - b + c

def matrix_from_quat(quaternions: jax.Array) -> jax.Array:
    """Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        Rotation matrices. The shape is (..., 3, 3).
    """
    # r, i, j, k = torch.unbind(quaternions, -1)
    r = quaternions[..., 0]
    i = quaternions[..., 1]
    j = quaternions[..., 2]
    k = quaternions[..., 3]

    # two_s = 2.0 / (quaternions * quaternions).sum(-1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    # stack the 9 matrix entries, exactly as in the PyTorch version
    o = jp.stack(
        (
            1.0 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1.0 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1.0 - two_s * (i * i + j * j),
        ),
        axis=-1,
    )

    return o.reshape(quaternions.shape[:-1] + (3, 3))

def subtract_frame_transforms(
    t01: jax.Array,
    q01: jax.Array,
    t02: jax.Array | None = None,
    q02: jax.Array | None = None,
) -> Tuple[jax.Array, jax.Array]:
    r"""Subtract transformations between two reference frames into a stationary frame.

    It performs the following transformation operation: T_12 = T_01^{-1} × T_02,
    where T_AB is the homogeneous transformation matrix from frame A to B.

    Args:
        t01: Position of frame 1 w.r.t. frame 0. Shape (N, 3).
        q01: Quaternion of frame 1 w.r.t. frame 0 in (w, x, y, z). Shape (N, 4).
        t02: Position of frame 2 w.r.t. frame 0. Shape (N, 3) or None.
        q02: Quaternion of frame 2 w.r.t. frame 0 in (w, x, y, z). Shape (N, 4) or None.

    Returns:
        (t12, q12): position and orientation of frame 2 w.r.t. frame 1.
        Shapes: (N, 3), (N, 4).
    """
    # compute orientation: q10 = q01^{-1}, q12 = q10 * q02 (or q10 if q02 is None)
    q10 = quat_inv(q01)
    if q02 is not None:
        q12 = quat_mul(q10, q02)
    else:
        q12 = q10

    # compute translation: t12 = q10 ∘ (t02 - t01)  (or q10 ∘ (-t01) if t02 is None)
    if t02 is not None:
        t12 = quat_apply(q10, t02 - t01)
    else:
        t12 = quat_apply(q10, -t01)

    return t12, q12

def quat_mul(q1, q2):
    # q1, q2: (..., 4)
    w1, x1, y1, z1 = jp.split(q1, 4, axis=-1)
    w2, x2, y2, z2 = jp.split(q2, 4, axis=-1)

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return jp.concatenate([w, x, y, z], axis=-1)

def quat_apply(quat: jax.Array, vec: jax.Array) -> jax.Array:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: (..., 4) quaternion in (w, x, y, z).
        vec:  (..., 3) vector in (x, y, z).

    Returns:
        (..., 3) rotated vector.
    """
    # svec.shape
    shape = vec.shape

    # quat = quat.reshape(-1, 4); vec = vec.reshape(-1, 3)
    quat_flat = quat.reshape(-1, 4)
    vec_flat = vec.reshape(-1, 3)

    # xyz = quat[:, 1:]
    xyz = quat_flat[..., 1:]  # (N, 3)

    # t = xyz.cross(vec, dim=-1) * 2
    t = jp.cross(xyz, vec_flat, axis=-1) * 2.0  # (N, 3)

    # vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)
    w = quat_flat[..., 0:1]
    rotated = vec_flat + w * t + jp.cross(xyz, t, axis=-1)

    return rotated.reshape(shape)

def normalize(q: jax.Array, eps: float = 1e-8) -> jax.Array:
    """模仿 torch 版 normalize(quat_yaw)，对最后一维做 L2 归一化。"""
    norm = jp.linalg.norm(q, axis=-1, keepdims=True)
    return q / jp.maximum(norm, eps)


def yaw_quat(quat: jax.Array) -> jax.Array:
    """Extract the yaw component of a quaternion.

    Args:
        quat: (..., 4) in (w, x, y, z).

    Returns:
        (..., 4) pure yaw quaternion in (w, x, y, z).
    """
    shape = quat.shape

    quat_yaw = quat.reshape(-1, 4)

    qw = quat_yaw[:, 0]
    qx = quat_yaw[:, 1]
    qy = quat_yaw[:, 2]
    qz = quat_yaw[:, 3]

    # yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    yaw = jp.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )

    # quat_yaw[:] = 0.0
    # quat_yaw[:, 3] = torch.sin(yaw / 2)
    # quat_yaw[:, 0] = torch.cos(yaw / 2)
    quat_yaw = jp.zeros_like(quat_yaw)
    quat_yaw = quat_yaw.at[:,3].set(jp.sin(yaw * 0.5))
    quat_yaw = quat_yaw.at[:,0].set(jp.cos(yaw * 0.5))

    # 对应：quat_yaw = normalize(quat_yaw)
    quat_yaw = normalize(quat_yaw)

    # 对应：return quat_yaw.view(shape)
    return quat_yaw.reshape(shape)

def quat_inv(q: jp.ndarray) -> jp.ndarray:
  """Calculates the inverse of quaternion q.

  Args:
    q: (4,) quaternion [w, x, y, z]

  Returns:
    The inverse of q, where qmult(q, inv_quat(q)) = [1, 0, 0, 0].
  """
  return q * jp.array([1, -1, -1, -1])

def rotate(vec: jp.ndarray, quat: jp.ndarray):
  """Rotates a vector vec by a unit quaternion quat.

  Args:
    vec: (3,) a vector
    quat: (4,) a quaternion

  Returns:
    ndarray(3) containing vec rotated by quat.
  """
  if len(vec.shape) != 1:
    raise ValueError('vec must have no batch dimensions.')
  s, u = quat[0], quat[1:]
  r = 2 * (jp.dot(u, vec) * u) + (s * s - jp.dot(u, u)) * vec
  r = r + 2 * s * jp.cross(u, vec)
  return r


def inv_rotate(vec: jp.ndarray, quat: jp.ndarray):
  """Rotates a vector vec by an inverted unit quaternion quat.

  Args:
    vec: (3,) a vector
    quat: (4,) a quaternion

  Returns:
    ndarray(3) containing vec rotated by the inverse of quat.
  """
  return rotate(vec, quat_inv(quat))

def quat_conjugate_jax(q: jax.Array) -> jax.Array:
    """q: (..., 4) in (w, x, y, z)"""
    return jp.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def axis_angle_from_quat_jax(q: jax.Array, eps: float = 1e-6) -> jax.Array:
    sign = jp.where(q[0:1] < 0.0, -1.0, 1.0)
    q = q * sign

    v = q[1:]
    mag = jp.linalg.norm(v, axis=-1)               # |v|
    half_angle = jp.arctan2(mag, q[0])        # θ/2
    angle = 2.0 * half_angle                       # θ

    sin_half = jp.sin(half_angle)
    sin_half_over_angle = jp.where(
        jp.abs(angle) > eps,
        sin_half / angle,
        0.5 - angle * angle / 48.0,
    )

    return v / sin_half_over_angle

def axis_angle_from_quat_batch_jax(q: jax.Array, eps: float = 1e-6) -> jax.Array:
    sign = jp.where(q[:, 0:1] < 0.0, -1.0, 1.0)
    q = q * sign

    v = q[:, 1:]
    mag = jp.linalg.norm(v, axis=-1)               # |v|
    half_angle = jp.arctan2(mag, q[:, 0])        # θ/2
    angle = 2.0 * half_angle                       # θ

    sin_half = jp.sin(half_angle)
    sin_half_over_angle = jp.where(
        jp.abs(angle) > eps,
        sin_half / angle,
        0.5 - angle * angle / 48.0,
    )

    return v / sin_half_over_angle[:, None]

def quat_error_magnitude_jax(q1: jax.Array, q2: jax.Array) -> jax.Array:
    quat_diff = quat_mul(q1, quat_conjugate_jax(q2))
    axis_angle = axis_angle_from_quat_jax(quat_diff)
    return jp.linalg.norm(axis_angle, axis=-1)


def quat_error_magnitude_batch_jax(q1: jax.Array, q2: jax.Array) -> jax.Array:
    quat_diff = quat_mul(q1, quat_conjugate_jax(q2))
    axis_angle = axis_angle_from_quat_batch_jax(quat_diff)
    return jp.linalg.norm(axis_angle, axis=-1)

def quat_mul_torch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions together.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        The product of the two quaternions in (w, x, y, z). Shape is (..., 4).

    Raises:
        ValueError: Input shapes of ``q1`` and ``q2`` are not matching.
    """
    # check input is correct
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    # reshape to (N, 4) for multiplication
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    # extract components from quaternions
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return torch.stack([w, x, y, z], dim=-1).view(shape)

def yaw_quat_torch(quat: torch.Tensor) -> torch.Tensor:
    """Extract the yaw component of a quaternion.

    Args:
        quat: The orientation in (w, x, y, z). Shape is (..., 4)

    Returns:
        A quaternion with only yaw component.
    """
    shape = quat.shape
    quat_yaw = quat.clone().view(-1, 4)
    qw = quat_yaw[:, 0]
    qx = quat_yaw[:, 1]
    qy = quat_yaw[:, 2]
    qz = quat_yaw[:, 3]
    yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    quat_yaw[:] = 0.0
    quat_yaw[:, 3] = torch.sin(yaw / 2)
    quat_yaw[:, 0] = torch.cos(yaw / 2)
    quat_yaw = normalize_torch(quat_yaw)
    return quat_yaw.view(shape)

def normalize_torch(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Normalizes a given input tensor to unit length.

    Args:
        x: Input tensor of shape (N, dims).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        Normalized tensor of shape (N, dims).
    """
    return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)


def quat_inv_torch(q: torch.Tensor) -> torch.Tensor:
    """Compute the inverse of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (N, 4).

    Returns:
        The inverse quaternion in (w, x, y, z). Shape is (N, 4).
    """
    return normalize_torch(quat_conjugate_torch(q))

def quat_conjugate_torch(q: torch.Tensor) -> torch.Tensor:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        The conjugate quaternion in (w, x, y, z). Shape is (..., 4).
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[:, 0:1], -q[:, 1:]), dim=-1).view(shape)

def quat_apply_torch(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)


def compute_body_motion_relative_w_torch(ref_anchor_pos_w, ref_anchor_quat_w, actual_anchor_pos_w, actual_anchor_quat_w, ref_body_quat_w, ref_body_pos_w, num_track_body):
    anchor_pos_w_repeat = ref_anchor_pos_w[None, :].repeat(num_track_body, 1)
    anchor_quat_w_repeat = ref_anchor_quat_w[None, :].repeat(num_track_body, 1)
    robot_anchor_pos_w_repeat = actual_anchor_pos_w[None, :].repeat(num_track_body, 1)
    robot_anchor_quat_w_repeat = actual_anchor_quat_w[None, :].repeat(num_track_body, 1)
    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat_torch(quat_mul_torch(robot_anchor_quat_w_repeat, quat_inv_torch(anchor_quat_w_repeat)))

    body_quat_relative_w = quat_mul_torch(delta_ori_w, ref_body_quat_w)
    body_pos_relative_w = delta_pos_w + quat_apply_torch(delta_ori_w, ref_body_pos_w - anchor_pos_w_repeat)

    return body_pos_relative_w, body_quat_relative_w

def compute_body_motion_relative_w(ref_anchor_pos_w: jax.Array, ref_anchor_quat_w: jax.Array, actual_anchor_pos_w: jax.Array, actual_anchor_quat_w: jax.Array, ref_body_quat_w: jax.Array, ref_body_pos_w: jax.Array, num_track_body: int) -> Tuple[jax.Array, jax.Array]:

    # anchor_pos_w:          (E, 3)   -> (E, 1, 3) -> (E, B, 3)
    anchor_pos_w_repeat = jp.repeat(ref_anchor_pos_w[None, :], num_track_body, axis=0)
    anchor_quat_w_repeat = jp.repeat(ref_anchor_quat_w[None, :], num_track_body, axis=0)
    robot_anchor_pos_w_repeat = jp.repeat(actual_anchor_pos_w[None, :], num_track_body, axis=0)
    robot_anchor_quat_w_repeat = jp.repeat(actual_anchor_quat_w[None, :], num_track_body, axis=0)
    #print(robot_anchor_pos_w_repeat)
    # === delta_pos_w = robot_anchor_pos_w_repeat; delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2] ===
    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w = delta_pos_w.at[..., 2].set(anchor_pos_w_repeat[..., 2])

    # === delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))) ===
    delta_ori_w = yaw_quat(
        quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    # === body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w) ===
    body_quat_relative_w = quat_mul(delta_ori_w, ref_body_quat_w)

    # === body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat) ===
    body_pos_relative_w = delta_pos_w + quat_apply(
        delta_ori_w,
        ref_body_pos_w - anchor_pos_w_repeat,
    )
    return body_pos_relative_w, body_quat_relative_w

class MotionLoader:
    def __init__(self, motion_file: str):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)

        self.fps = float(data["fps"])
        self.isaacsim_to_mujoco_body_indexes = np.array([0, 4, 10, 18, 5, 11, 19, 9, 16, 22, 28, 17, 23, 29])
        self.isaacsim_to_mujoco_joint_indexes = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28])
        self.joint_pos = jp.asarray(data["joint_pos"][:, self.isaacsim_to_mujoco_joint_indexes], dtype=jp.float32)          # [T, n_q]
        self.joint_vel = jp.asarray(data["joint_vel"][:, self.isaacsim_to_mujoco_joint_indexes], dtype=jp.float32)          # [T, n_q]
        self._body_pos_w = jp.asarray(data["body_pos_w"], dtype=jp.float32)       # [T, n_body, 3]
        self._body_quat_w = jp.asarray(data["body_quat_w"], dtype=jp.float32)     # [T, n_body, 4]
        self._body_lin_vel_w = jp.asarray(data["body_lin_vel_w"], dtype=jp.float32)   # [T, n_body, 3]
        self._body_ang_vel_w = jp.asarray(data["body_ang_vel_w"], dtype=jp.float32)   # [T, n_body, 3]

        self.time_step_total = int(self.joint_pos.shape[0])

    @property
    def body_pos_w(self) -> jp.ndarray:
        # torch: self._body_pos_w[:, self._body_indexes]
        return self._body_pos_w[:, self.isaacsim_to_mujoco_body_indexes, :]

    @property
    def body_quat_w(self) -> jp.ndarray:
        return self._body_quat_w[:, self.isaacsim_to_mujoco_body_indexes, :]

    @property
    def body_lin_vel_w(self) -> jp.ndarray:
        return self._body_lin_vel_w[:, self.isaacsim_to_mujoco_body_indexes, :]

    @property
    def body_ang_vel_w(self) -> jp.ndarray:
        return self._body_ang_vel_w[:, self.isaacsim_to_mujoco_body_indexes, :]


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



if __name__ == "__main__":
    import mujoco  # 此时已设置好 MUJOCO_GL
    import mujoco.viewer
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
    args = parser.parse_args()
    cfg_file = os.path.join("envs", "{}.yaml".format(args.task))
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

    TRACK_BODIES = [
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ]

    mj_model = mujoco.MjModel.from_xml_path("/home/admin123/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/mjcf/g1.xml")
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
    # print()
    # mj_data.qpos = np.concatenate(
    #     [
    #         np.array(cfg["init_state"]["pos"], dtype=np.float32),
    #         np.array(cfg["init_state"]["rot"][3:4] + cfg["init_state"]["rot"][0:3], dtype=np.float32),
    #         default_dof_pos,
    #     ]
    # )
    track_id = np.array([mj_model.body(name).id for name in TRACK_BODIES])
    mujoco.mj_forward(mj_model, mj_data)
    normalizer = NormalTanhDistribution()

    actions = np.zeros((cfg["env"]["num_actions"]), dtype=np.float32)
    dof_targets = np.zeros(default_dof_pos.shape, dtype=np.float32)
    gait_frequency = 1.5
    gait_process = 0.0
    lin_vel_x = 0.6
    lin_vel_y = ang_vel_yaw = 0.0
    it = 0
    obs_minmax_normalizer = minmaxnormalizer()
    motion = MotionLoader("/home/admin123/whole_body_tracking/motion.npz")
    step = 0
    print(mj_data.xquat[0])
    log = {
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        viewer.cam.elevation = -20
        print(f"Set command (x, y, yaw): ")
        while viewer.is_running():
            if select.select([sys.stdin], [], [], 0)[0]:
                try:
                    parts = sys.stdin.readline().strip().split()
                    if len(parts) == 3:
                        #lin_vel_x, lin_vel_y, ang_vel_yaw = map(float, parts)
                        # if lin_vel_x == 0 and lin_vel_y == 0 and ang_vel_yaw == 0:
                        #     gait_frequency = 0
                        # else:
                        #     gait_frequency = np.average(cfg["commands"]["gait_frequency"])
                        print(
                            f"Updated command to: x={lin_vel_x}, y={lin_vel_y}, yaw={ang_vel_yaw}\nSet command (x, y, yaw): ",
                            end="",
                        )
                    else:
                        raise ValueError
                except ValueError:
                    print("Invalid input. Enter three numeric values.\nSet command (x, y, yaw): ", end="")
            # dof_pos = mj_data.qpos.astype(np.float32)[7:]
            # dof_vel = mj_data.qvel.astype(np.float32)[6:]
            # #print(mj_data.qpos.astype(np.float32)[2])
            # quat = mj_data.sensor("orientation").data[[1, 2, 3, 0]].astype(np.float32)
            # base_ang_vel = mj_data.sensor("angular-velocity").data.astype(np.float32)
            # base_lin_vel = mj_data.sensor("linear-velocity").data.astype(np.float32)
            # quat_wxyz = mj_data.sensor("orientation").data.astype(np.float32)
            # base_pos = mj_data.qpos.astype(np.float32)[:3]

            #print("base_lin_vel", base_lin_vel)
            # projected_gravity = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
            # ang_vel_global = rotate(quat, base_ang_vel)
            #print("ang_vel_global", ang_vel_global, gait_process, gait_frequency)
            if it % cfg["control"]["decimation"] == 0:
                #print("id", mj_model.body("torso_link").id)
                mj_data.qpos = np.concatenate(
                [
                    np.array(motion.body_pos_w[step][0], dtype=np.float32),
                    np.array(motion.body_quat_w[step][0], dtype=np.float32),
                    motion.joint_pos[step],
                    # motion.joint_pos[step][0:6],
                    # motion.joint_pos[step][12:15],
                    # motion.joint_pos[step][22:29],

                    # motion.joint_pos[step][15:22],

                ]
                )
                mj_data.qvel = np.concatenate(
                [
                    np.array(motion.body_lin_vel_w[step][0], dtype=np.float32),
                    np.array(motion.body_ang_vel_w[step][0], dtype=np.float32),
                    motion.joint_vel[step],
                ]
                )
                mujoco.mj_forward(mj_model, mj_data)

                body_pos_relative_w, body_quat_relative_w = compute_body_motion_relative_w(
                    ref_anchor_pos_w=motion.body_pos_w[step, 7],
                    ref_anchor_quat_w=motion.body_quat_w[step, 7],
                    actual_anchor_pos_w=mj_data.xpos[16],
                    actual_anchor_quat_w=mj_data.xquat[16],
                    ref_body_quat_w=motion.body_quat_w[step],
                    ref_body_pos_w=motion.body_pos_w[step],
                    num_track_body=14,
                )

                body_pos_relative_w_torch, body_quat_relative_w_torch = compute_body_motion_relative_w_torch(
                    ref_anchor_pos_w=torch.tensor(np.array(motion.body_pos_w[step, 7]), dtype=torch.float64),
                    ref_anchor_quat_w=torch.tensor(np.array(motion.body_quat_w[step, 7]), dtype=torch.float64),
                    actual_anchor_pos_w=torch.tensor(np.array(mj_data.xpos[16]), dtype=torch.float64),
                    actual_anchor_quat_w=torch.tensor(np.array(mj_data.xquat[16]), dtype=torch.float64),
                    ref_body_quat_w=torch.tensor(np.array(motion.body_quat_w[step]), dtype=torch.float64),
                    ref_body_pos_w=torch.tensor(np.array(motion.body_pos_w[step]), dtype=torch.float64),
                    num_track_body=14,
                )


                # mj_data.xpos[np.array([[ 1 , 3 , 5,  7,  9 ,11, 13, 16, 18, 20, 23, 25, 27, 30]])] = body_pos_relative_w
                # mj_data.xquat[np.array([[ 1 , 3 , 5,  7,  9 ,11, 13, 16, 18, 20, 23, 25, 27, 30]])] = body_quat_relative_w


                print(body_quat_relative_w[1],body_quat_relative_w_torch[1], mj_data.xquat[3])
                #print(motion.body_lin_vel_w[step, 7], )
                # lin vel of TRACK_BODIES
                global_linvel = np.array([mj_data.sensor(name+"_global_linvel").data for name in TRACK_BODIES])
                global_angvel = np.array([mj_data.sensor(name+"_global_angvel").data for name in TRACK_BODIES])
                log["joint_pos"].append(motion.joint_pos[step])
                log["joint_vel"].append(motion.joint_vel[step])
                log["body_pos_w"].append(mj_data.xpos[track_id])
                log["body_quat_w"].append(mj_data.xquat[track_id])
                log["body_lin_vel_w"].append(global_linvel)
                log["body_ang_vel_w"].append(global_angvel)
                step+=1

                #print()
                #print(global_linvel.shape)
                #print(mj_data.sensor("pelvis_global_linvel").data, motion.body_lin_vel_w[step, 0])
            #print(step, it)
            if step >= motion.body_pos_w.shape[0]:
                break

            viewer.cam.lookat[:] = mj_data.qpos.astype(np.float32)[0:3]
            viewer.sync()
            it += 1
            gait_process = np.fmod(gait_process + cfg["sim"]["dt"] * gait_frequency, 1.0)

    for k in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ):
        log[k] = np.stack(log[k], axis=0)
        print(log[k].shape)
    np.savez("/home/admin123/booster_gym/motion.npz", **log)
