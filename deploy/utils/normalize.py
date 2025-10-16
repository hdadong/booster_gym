import numpy as np

class minmaxnormalizer:

    def __init__(self, dtype=np.float32):
        self.dtype = dtype

        # Normalization limits
        self.obs_limit_min = np.full((1, 87), -1.0, dtype=dtype)
        self.obs_limit_max = np.full((1, 87),  1.0, dtype=dtype)
        self.q_idxs = [i for i in range(11, 23, 1)]
        q_min = np.array([-1.8, -0.3, -1.0, 0.0, -0.87, -0.44,
                          -1.8, -1.57, -1.0, 0.0, -0.87, -0.44], dtype=dtype) - 0.25
        q_max = np.array([ 1.57,  1.57,  1.0, 2.34,  0.35,  0.44,
                           1.57,  0.3 ,  1.0, 2.34,  0.35,  0.44], dtype=dtype) + 0.25
        self.obs_limit_min[:, self.q_idxs] = q_min
        self.obs_limit_max[:, self.q_idxs] = q_max

        self.wm_q_idxs = [i for i in range(60, 72, 1)]
        self.obs_limit_min[:, self.wm_q_idxs] = q_min
        self.obs_limit_max[:, self.wm_q_idxs] = q_max

        self.forward_vel_idx = 57
        self.obs_limit_min[:, self.forward_vel_idx] = -0.2
        self.obs_limit_max[:, self.forward_vel_idx] =  2.5

        self.y_vel_idx = 58
        self.obs_limit_min[:, self.y_vel_idx] = -0.5
        self.obs_limit_max[:, self.y_vel_idx] =  0.5

        self.z_vel_idx = 59
        self.obs_limit_min[:, self.z_vel_idx] = -0.5
        self.obs_limit_max[:, self.z_vel_idx] =  0.5

        self.roll_rate_idx  = 3
        self.pitch_rate_idx = 4
        self.turn_rate_idx  = 5
        self.rpy_rate_idxs = [i for i in range(self.roll_rate_idx, self.turn_rate_idx + 1, 1)]
        rpy_min = np.array([-1.5, -1.5, -1.5], dtype=dtype)
        rpy_max = np.array([ 1.5,  1.5,  1.5], dtype=dtype)
        self.obs_limit_min[:, self.rpy_rate_idxs] = rpy_min
        self.obs_limit_max[:, self.rpy_rate_idxs] = rpy_max

        self.wm_roll_rate_idx  = 47
        self.wm_pitch_rate_idx = 48
        self.wm_turn_rate_idx  = 49
        self.wm_rpy_rate_idxs = [i for i in range(self.wm_roll_rate_idx, self.wm_turn_rate_idx + 1, 1)]
        self.obs_limit_min[:, self.wm_rpy_rate_idxs] = rpy_min
        self.obs_limit_max[:, self.wm_rpy_rate_idxs] = rpy_max

        self.qd_idxs = [i for i in range(23, 35, 1)]
        self.obs_limit_min[:, self.qd_idxs] = -20.0
        self.obs_limit_max[:, self.qd_idxs] =  20.0

        self.wm_qd_idxs = [i for i in range(72, 84, 1)]
        self.obs_limit_min[:, self.wm_qd_idxs] = -20.0
        self.obs_limit_max[:, self.wm_qd_idxs] =  20.0

        self.wm_height_idx = 84
        self.obs_limit_min[:, self.wm_height_idx] = 0.0
        self.obs_limit_max[:, self.wm_height_idx] = 0.8

        self.wm_gravity_idxs     = [i for i in range(50, 53, 1)]
        self.wm_quat_idxs        = [i for i in range(53, 57, 1)]
        self.wm_base_vel_idxs    = [i for i in range(57, 60, 1)]
        self.wm_gait_process_idx = 85
        self.wm_gait_frequency_idx = 86

        self.priv_rpy_rate_idxs      = self.wm_rpy_rate_idxs
        self.priv_gravity_idxs       = self.wm_gravity_idxs
        self.priv_base_lin_vel_idxs  = [i for i in range(53, 56, 1)]
        self.priv_global_ang_vel_idxs= [i for i in range(56, 59, 1)]
        self.priv_q_idxs             = [i for i in range(59, 71, 1)]
        self.priv_qd_idxs            = [i for i in range(71, 83, 1)]
        self.priv_height_idx         = 83
        # -------------------------------------------------------------------

    def normalize_obs(self, obs):
        obs = np.asarray(obs, dtype=self.dtype)
        denom = self.obs_limit_max - self.obs_limit_min
        norm = 2.0 * (obs - self.obs_limit_min) / denom - 1.0
        return norm

    def denormalize_obs(self, normalized_obs):
        normalized_obs = np.asarray(normalized_obs, dtype=self.dtype)
        return (normalized_obs + 1.0) * (self.obs_limit_max - self.obs_limit_min) / 2.0 + self.obs_limit_min
