import numpy as np
import time
import yaml
import logging
import threading
import csv
from pathlib import Path
from datetime import datetime
import re
import os 
from booster_robotics_sdk_python import (
    ChannelFactory,
    B1LocoClient,
    B1LowCmdPublisher,
    B1LowStateSubscriber,
    LowCmd,
    LowState,
    B1JointCnt,
    RobotMode,
    GetModeResponse,
)

from utils.command import create_prepare_cmd, create_first_frame_rl_cmd
from utils.remote_control_service import RemoteControlService
from utils.rotate import rotate_vector_inverse_rpy, rotate_vector_rpy, rpy_zyx_to_quat_wxyz
from utils.timer import TimerConfig, Timer
from utils.policy import Policy
from utils.policy_simp import Policy as Policy_simp
from utils.tcp_server import send_checkpoint_until_success, BackgroundFileServer
from utils.vicon import Vicon
def get_latest_policy_path(policy_dir):
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

class Controller:
    def __init__(self, cfg_file, policy_path, max_episode_length) -> None:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.max_episode_length = max_episode_length
        self.step = 0
        num_envs = 1
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
        self.data_buffers = [
            {key: [] for key in data_dict}  # Create a new dictionary with the same structure
            for _ in range(num_envs)
        ]
        self.last_logged_tick = float('-inf')   # guard to avoid duplicate rows per tick
        self.csv_lock = threading.Lock()
        self.vicon = Vicon()

        # Load config
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        # Initialize components
        self.remoteControlService = RemoteControlService()
        if self.cfg["common"]["use_simp"]:
            print("simp policy")
            self.policy = Policy_simp(cfg=self.cfg, policy_path=policy_path)
        else:
            self.policy = Policy(cfg=self.cfg)

        self._init_timer()
        self._init_low_state_values()
        self._init_communication()
        self.publish_runner = None
        self.running = True

        self.publish_lock = threading.Lock()
        self._init_csv_logger()

    def _init_csv_logger(self):
        # directory + filename
        log_dir = self.cfg.get("logging", {}).get("dir", "logs")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = str(Path(log_dir) / f"b1_run_{ts}.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self._csv_rows_written = 0

        # build header dynamically
        q_cols   = [f"q_{i}" for i in range(B1JointCnt)]
        dq_cols  = [f"dq_{i}" for i in range(B1JointCnt)]
        tgt_cols = [f"target_{i}" for i in range(B1JointCnt)]
        header = (
            ["t",
            "body_height",
            "vx_cmd","vy_cmd","vyaw_cmd",
            "rpy_roll","rpy_pitch","rpy_yaw",
            "acc_x","acc_y","acc_z",
            "gyro_x","gyro_y","gyro_z",
            "proj_gx","proj_gy","proj_gz",
            "base_lin_vx","base_lin_vy","base_lin_vz"]
            + q_cols + dq_cols + tgt_cols
        )
        self.csv_writer.writerow(header)
        self.csv_file.flush()

    def _init_timer(self):
        self.timer = Timer(TimerConfig(time_step=self.cfg["common"]["dt"]))
        self.next_publish_time = self.timer.get_time()
        self.next_inference_time = self.timer.get_time()

    def _init_low_state_values(self):
        self.base_ang_vel = np.zeros(3, dtype=np.float32)
        self.global_ang_vel = np.zeros(3, dtype=np.float32)
        self.global_lin_vel = np.zeros(3, dtype=np.float32)
        self.base_lin_vel = np.zeros(3, dtype=np.float32)
        self.acc = np.zeros(3, dtype=np.float32)
        self.acc_update_time = self.timer.get_time()
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.quat_wxyz = np.array([1, 0, 0, 0], dtype=np.float32)
        self.dof_pos = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_vel = np.zeros(B1JointCnt, dtype=np.float32)
        self.torques = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_target = np.zeros(B1JointCnt, dtype=np.float32)
        self.filtered_dof_target = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_pos_latest = np.zeros(B1JointCnt, dtype=np.float32)
        self.body_height = 0.0
    def _init_communication(self) -> None:
        try:
            self.low_cmd = LowCmd()
            self.low_state_subscriber = B1LowStateSubscriber(self._low_state_handler)
            self.low_cmd_publisher = B1LowCmdPublisher()
            self.client = B1LocoClient()

            self.low_state_subscriber.InitChannel()
            self.low_cmd_publisher.InitChannel()
            self.client.Init()
        except Exception as e:
            self.logger.error(f"Failed to initialize communication: {e}")
            raise

    def _low_state_handler(self, low_state_msg: LowState):
        if abs(low_state_msg.imu_state.rpy[0]) > 0.785 or abs(low_state_msg.imu_state.rpy[1]) > 0.785:
            #self.logger.warning("IMU base rpy values are too large: {}".format(low_state_msg.imu_state.rpy))
            self.running = False


        self.timer.tick_timer_if_sim()
        time_now = self.timer.get_time()
        for i, motor in enumerate(low_state_msg.motor_state_serial):
            self.dof_pos_latest[i] = motor.q
            self.torques[i] = motor.tau_est



        if time_now >= self.next_inference_time:
            r, p, y = low_state_msg.imu_state.rpy
            acc_body = np.array(low_state_msg.imu_state.acc, dtype=np.float32)
            a_world = rotate_vector_rpy(r, p, y, acc_body) + np.array([0.0, 0.0, -9.81], dtype=np.float32)

            # 速度积分（注意 dt、漂移与零偏）
            dt = max(0.0, time_now - self.acc_update_time)
            self.global_lin_vel += a_world * dt
            self.base_lin_vel = rotate_vector_inverse_rpy(
                    low_state_msg.imu_state.rpy[0],
                    low_state_msg.imu_state.rpy[1],
                    low_state_msg.imu_state.rpy[2],
                    self.global_lin_vel,
                )
            self.acc_update_time = time_now
            self.body_height = self.vicon.position[2]

            if self.step > 0:
                if self.body_height < 0.4 or self.body_height > 0.75:
                    self.logger.warning("body height risk: {}".format(self.body_height))
                    self.running = False
                elif abs(self.global_lin_vel[0]) > 10.0 or abs(self.global_lin_vel[1]) > 10.0 or abs(self.global_lin_vel[2]) > 10.0:
                    self.logger.warning("global vel: {}".format(self.global_lin_vel))
                    self.running = False
                elif self.step >= self.max_episode_length: 
                    self.logger.warning("step > max_episode_length")
                    self.running = False
                elif abs(low_state_msg.imu_state.rpy[0]) > 0.785 or abs(low_state_msg.imu_state.rpy[1]) > 0.785:
                    self.logger.warning("IMU base rpy values are too large: {}".format(low_state_msg.imu_state.rpy))
                    self.running = False
            self.projected_gravity[:] = rotate_vector_inverse_rpy(
                low_state_msg.imu_state.rpy[0],
                low_state_msg.imu_state.rpy[1],
                low_state_msg.imu_state.rpy[2],
                np.array([0.0, 0.0, -1.0]),
            )
            self.base_ang_vel[:] = low_state_msg.imu_state.gyro
            self.global_ang_vel[:] = rotate_vector_rpy(
                low_state_msg.imu_state.rpy[0],
                low_state_msg.imu_state.rpy[1],
                low_state_msg.imu_state.rpy[2],
                low_state_msg.imu_state.gyro
            )
            self.quat_wxyz = rpy_zyx_to_quat_wxyz(roll=low_state_msg.imu_state.rpy[0], pitch=low_state_msg.imu_state.rpy[1], yaw=low_state_msg.imu_state.rpy[2])
            for i, motor in enumerate(low_state_msg.motor_state_serial):
                self.dof_pos[i] = motor.q
                self.dof_vel[i] = motor.dq

            # log exactly once per inference tick
            if self.last_logged_tick != self.next_inference_time:
                self._log_one_row(time_now, low_state_msg)
                self.last_logged_tick = self.next_inference_time

    def _log_one_row(self, t: float, low_state_msg: LowState):
        # read values from current state and message
        rpy  = low_state_msg.imu_state.rpy
        gyro = low_state_msg.imu_state.gyro
        acc  = low_state_msg.imu_state.acc

        row = [
            t,
            self.body_height,
            self.remoteControlService.get_vx_cmd(),
            self.remoteControlService.get_vy_cmd(),
            self.remoteControlService.get_vyaw_cmd(),
            rpy[0], rpy[1], rpy[2],
            acc[0], acc[1], acc[2],
            gyro[0], gyro[1], gyro[2],
            self.projected_gravity[0], self.projected_gravity[1], self.projected_gravity[2],
            self.base_lin_vel[0], self.base_lin_vel[1], self.base_lin_vel[2],
        ]

        # joint arrays
        row.extend(self.dof_pos.tolist())
        row.extend(self.dof_vel.tolist())
        # use the latest available targets; filtered or raw depending on your preference
        row.extend(self.filtered_dof_target.tolist())

        if self.step != 0: 
            with self.csv_lock:
                self.csv_writer.writerow(row)
                self._csv_rows_written += 1
                # flush occasionally to avoid data loss but keep IO reasonable
                if self._csv_rows_written % 100 == 0:
                    self.csv_file.flush()

    def _send_cmd(self, cmd: LowCmd):
        self.low_cmd_publisher.Write(cmd)

    def cleanup(self) -> None:
        """Cleanup resources."""
        self.remoteControlService.close()
        if hasattr(self, "low_cmd_publisher"):
            self.low_cmd_publisher.CloseChannel()
        if hasattr(self, "low_state_subscriber"):
            self.low_state_subscriber.CloseChannel()
        if hasattr(self, "publish_runner") and getattr(self, "publish_runner") != None:
            self.publish_runner.join(timeout=1.0)
        if hasattr(self, "csv_file"):
            try:
                self.csv_file.flush()
                self.csv_file.close()
                self.logger.info(f"CSV saved to {self.csv_path}")
            except Exception as e:
                self.logger.warning(f"Error closing CSV: {e}")
        if hasattr(self, "vicon"):
            self.vicon.stop()
    def start_custom_mode_conditionally(self):
        print(f"{self.remoteControlService.get_custom_mode_operation_hint()}")
        while True:
            if self.remoteControlService.start_custom_mode():
                break
            time.sleep(0.1)
        start_time = time.perf_counter()
        create_prepare_cmd(self.low_cmd, self.cfg)
        for i in range(B1JointCnt):
            self.dof_target[i] = self.low_cmd.motor_cmd[i].q
            self.filtered_dof_target[i] = self.low_cmd.motor_cmd[i].q
        self._send_cmd(self.low_cmd)
        send_time = time.perf_counter()
        self.logger.debug(f"Send cmd took {(send_time - start_time)*1000:.4f} ms")
        self.client.ChangeMode(RobotMode.kCustom)
        end_time = time.perf_counter()
        self.logger.debug(f"Change mode took {(end_time - send_time)*1000:.4f} ms")

    def start_rl_gait_conditionally(self):
        print(f"{self.remoteControlService.get_rl_gait_operation_hint()}")
        while True:
            if self.remoteControlService.start_rl_gait():
                break
            time.sleep(0.1)
        create_first_frame_rl_cmd(self.low_cmd, self.cfg)
        self._send_cmd(self.low_cmd)
        self.next_inference_time = self.timer.get_time()
        self.next_publish_time = self.timer.get_time()
        self.publish_runner = threading.Thread(target=self._publish_cmd)
        self.publish_runner.daemon = True
        self.publish_runner.start()
        print(f"{self.remoteControlService.get_operation_hint()}")

    def run(self):
        time_now = self.timer.get_time()
        if time_now < self.next_inference_time:
            time.sleep(0.001)
            return
        self.logger.debug("-----------------------------------------------------")
        self.next_inference_time += self.policy.get_policy_interval()
        self.logger.debug(f"Next start time: {self.next_inference_time}")
        start_time = time.perf_counter()
        if self.step != 0:
            self.data_buffers[0]['state'].append(self.policy.obs)
            self.data_buffers[0]['wm_state'].append(self.policy.wm_obs)
            self.data_buffers[0]['priv_state'].append(self.policy.priv_obs)
            self.data_buffers[0]['actions'].append(self.policy.actions)
            self.data_buffers[0]['torques'].append(self.torques[:11])
            self.data_buffers[0]['contacts'].append([0.0,0.0])
            self.data_buffers[0]['rewards'].append(0)
            self.data_buffers[0]['timestamps'].append(0)

        self.dof_target[:] = self.policy.inference(
            time_now=time_now,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            base_ang_vel=self.base_ang_vel,
            projected_gravity=self.projected_gravity,
            vx=0.1,#self.remoteControlService.get_vx_cmd(),
            vy=self.remoteControlService.get_vy_cmd(),
            vyaw=self.remoteControlService.get_vyaw_cmd(),
            quat_wxyz=self.quat_wxyz, 
            base_lin_vel=self.base_lin_vel, 
            body_height=self.body_height, 
            ang_vel_global=self.global_ang_vel
        )
        # gm: GetModeResponse = GetModeResponse()
        # res = self.client.GetMode(gm)
        # if gm.mode == RobotMode.kPrepare:
        #     self.logger.warning("robot mode: {}".format(gm.mode))
        #     self.running = False
        self.step += 1


        inference_time = time.perf_counter()
        self.logger.debug(f"Inference took {(inference_time - start_time)*1000:.4f} ms")
        time.sleep(0.001)

    def _publish_cmd(self):
        while self.running:
            time_now = self.timer.get_time()
            if time_now < self.next_publish_time:
                time.sleep(0.001)
                continue
            self.next_publish_time += self.cfg["common"]["dt"]
            self.logger.debug(f"Next publish time: {self.next_publish_time}")

            self.filtered_dof_target = self.dof_target

            for i in range(B1JointCnt):
                self.low_cmd.motor_cmd[i].q = self.filtered_dof_target[i]

            # Use series-parallel conversion for torque to avoid non-linearity
            for i in self.cfg["mech"]["parallel_mech_indexes"]:
                self.low_cmd.motor_cmd[i].q = self.dof_pos_latest[i]
                self.low_cmd.motor_cmd[i].tau = np.clip(
                    (self.filtered_dof_target[i] - self.dof_pos_latest[i]) * self.cfg["common"]["stiffness"][i],
                    -self.cfg["common"]["torque_limit"][i],
                    self.cfg["common"]["torque_limit"][i],
                )
                self.low_cmd.motor_cmd[i].kp = 0.0

            start_time = time.perf_counter()
            self._send_cmd(self.low_cmd)
            publish_time = time.perf_counter()
            self.logger.debug(f"Publish took {(publish_time - start_time)*1000:.4f} ms")
            time.sleep(0.001)

    def __enter__(self) -> "Controller":
        return self

    def __exit__(self, *args) -> None:
        self.cleanup()

def run_real(cfg_file, policy_path, max_episode_length):
    print(f"Starting custom controller, connecting to {args.net} ...")
    ChannelFactory.Instance().Init(0, args.net)

    with Controller(cfg_file, policy_path, max_episode_length) as controller:
        time.sleep(2)  # Wait for channels to initialize
        print("Initialization complete.")
        controller.start_custom_mode_conditionally()
        controller.start_rl_gait_conditionally()

        try:
            while controller.running:
                controller.run()
            if controller.step >= max_episode_length:
                controller.client.ChangeMode(RobotMode.kPrepare)
            else:
                controller.client.ChangeMode(RobotMode.kPrepare)
            return controller.step, controller.data_buffers

        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Cleaning up...")
            return controller.step, controller.data_buffers
def wait_for_yes():
    """
    在 /dev/tty 上阻塞等待用户输入 'y' 或 'Y' + Enter。
    如果一时拿不到控制台，就每 0.5s 重试一次，不会刷屏。
    """
    import sys, time
    prompt_printed = False
    while True:
        if not prompt_printed:
            print("\nPress Y then Enter to continue: ", end="", flush=True)
            prompt_printed = True

        line = None
        # 1) 优先用真正的 TTY
        try:
            with open('/dev/tty', 'r') as tty:
                line = tty.readline()
        except Exception:
            # 2) 退而求其次：如果当前 stdin 还是 TTY，用它
            try:
                if sys.stdin and sys.stdin.isatty():
                    line = sys.stdin.readline()
            except Exception:
                pass

        if line is None:
            time.sleep(0.5)   # 没拿到输入设备，稍等再试
            continue

        if line.strip().lower() == 'y':
            print()  # 换行美观
            return True

        # 输入不是 Y，则给个简短提示，但不重复整行 prompt
        print("\nType 'Y' and press Enter to continue: ", end="", flush=True)

def confirm_continue():
    try:
        s = input("press Y to continue").strip()
    except EOFError:
        return False
    return s.lower() == 'y'

if __name__ == "__main__":
    import argparse
    import signal
    import sys
    import os

    def signal_handler(sig, frame):
        print("\nShutting down...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str, help="Name of the configuration file.")
    parser.add_argument("--net", type=str, default="127.0.0.1", help="Network interface for SDK communication.")
    args = parser.parse_args()
    cfg_file = os.path.join("configs", args.config)


    base_data_dir = os.path.join('./lift_data/data_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(base_data_dir, exist_ok=True)

    real_data_dir = os.path.join(base_data_dir, 'real_data_dir')
    os.makedirs(real_data_dir, exist_ok=True)

    max_episode_length = 500
    training_server = '10.1.108.171'
    flat_port = 9002
    data_port = 9003

    flag_policy_train = os.path.join(base_data_dir, 'policy_train.flag')

    policy_set = set()
    policy_set.add(None)
    policy_dir = os.path.join(base_data_dir, 'policy_ckpt')
    os.makedirs(policy_dir, exist_ok=True)


    

    while True:
        env_id = 0
        total_step = 0
        episode_num = 0
        policy_path = None
        policy_server = BackgroundFileServer(host="0.0.0.0", port=9001, save_dir=policy_dir)
        policy_server.start() 
        while policy_path is None or policy_path in policy_set:
            policy_path = get_latest_policy_path(policy_dir)
            time.sleep(3)
        
        policy_set.add(policy_path)
        print("load the policy:", policy_path)
        policy_server.stop()
        while total_step < max_episode_length:
            wait_for_yes()

            episode_step, data_buffers = run_real(cfg_file, policy_path, max_episode_length)

            npz_filename = os.path.join(
                real_data_dir,
                f'env_{episode_num}_data_{episode_step}.npz',
            )
            state_array = np.array(data_buffers[env_id]['state'], dtype=np.float32)
            wm_state_array = np.array(data_buffers[env_id]['wm_state'], dtype=np.float32)
            priv_state_array = np.array(data_buffers[env_id]['priv_state'], dtype=np.float32)
            actions_array = np.array(data_buffers[env_id]['actions'], dtype=np.float32)
            torques_array = np.array(data_buffers[env_id]['torques'], dtype=np.float32)
            contact_array = np.array(data_buffers[env_id]['contacts'], dtype=np.float32)
            rewards_array = np.array(data_buffers[env_id]['rewards'], dtype=np.float32)
            timestamps_array = np.array(data_buffers[env_id]['timestamps'], dtype=np.float64)
            wait_for_yes()

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
            send_checkpoint_until_success(
            ip=training_server,
            port=data_port,
            file_path=npz_filename,
            )
            episode_num += 1
            total_step += episode_step                
            data_buffers[env_id]['state'].clear()
            data_buffers[env_id]['wm_state'].clear()
            data_buffers[env_id]['priv_state'].clear()
            data_buffers[env_id]['actions'].clear()
            data_buffers[env_id]['torques'].clear()
            data_buffers[env_id]['contacts'].clear()
            data_buffers[env_id]['rewards'].clear()
            data_buffers[env_id]['timestamps'].clear()

            
        open(flag_policy_train, 'w').close()
        send_checkpoint_until_success(
        ip=training_server,
        port=flat_port,
        file_path=flag_policy_train,
        )

