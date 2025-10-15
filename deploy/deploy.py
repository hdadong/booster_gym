import numpy as np
import time
import yaml
import logging
import threading
import csv
from pathlib import Path
from datetime import datetime

from booster_robotics_sdk_python import (
    ChannelFactory,
    B1LocoClient,
    B1LowCmdPublisher,
    B1LowStateSubscriber,
    LowCmd,
    LowState,
    B1JointCnt,
    RobotMode,
)

from utils.command import create_prepare_cmd, create_first_frame_rl_cmd
from utils.remote_control_service import RemoteControlService
from utils.rotate import rotate_vector_inverse_rpy, rotate_vector_rpy
from utils.timer import TimerConfig, Timer
from utils.policy import Policy
from utils.policy_simp import Policy as Policy_simp


class Controller:
    def __init__(self, cfg_file) -> None:
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Load config
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)

        # Initialize components
        self.remoteControlService = RemoteControlService()
        if self.cfg["common"]["use_simp"]:
            print("simp policy")
            self.policy = Policy_simp(cfg=self.cfg)
        else:
            self.policy = Policy(cfg=self.cfg)

        self._init_timer()
        self._init_low_state_values()
        self._init_communication()
        self.publish_runner = None
        self.running = True

        self.publish_lock = threading.Lock()
        self._init_csv_logger()
        self.last_logged_tick = float('-inf')   # guard to avoid duplicate rows per tick

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
        self.base_lin_vel = np.zeros(3, dtype=np.float32)
        self.acc = np.zeros(3, dtype=np.float32)
        self.acc_update_time = self.timer.get_time()
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.dof_pos = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_vel = np.zeros(B1JointCnt, dtype=np.float32)

        self.dof_target = np.zeros(B1JointCnt, dtype=np.float32)
        self.filtered_dof_target = np.zeros(B1JointCnt, dtype=np.float32)
        self.dof_pos_latest = np.zeros(B1JointCnt, dtype=np.float32)

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
        if abs(low_state_msg.imu_state.rpy[0]) > 1.0 or abs(low_state_msg.imu_state.rpy[1]) > 1.0:
            self.logger.warning("IMU base rpy values are too large: {}".format(low_state_msg.imu_state.rpy))
            self.running = False
        self.timer.tick_timer_if_sim()
        time_now = self.timer.get_time()
        for i, motor in enumerate(low_state_msg.motor_state_serial):
            self.dof_pos_latest[i] = motor.q
        self.acc = low_state_msg.imu_state.acc
        self.base_lin_vel = self.base_lin_vel + self.acc * (time_now - self.acc_update_time)
        self.acc_update_time = time_now

        if time_now >= self.next_inference_time:
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

        self.dof_target[:] = self.policy.inference(
            time_now=time_now,
            dof_pos=self.dof_pos,
            dof_vel=self.dof_vel,
            base_ang_vel=self.base_ang_vel,
            projected_gravity=self.projected_gravity,
            vx=self.remoteControlService.get_vx_cmd(),
            vy=self.remoteControlService.get_vy_cmd(),
            vyaw=self.remoteControlService.get_vyaw_cmd(),
        )

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

    print(f"Starting custom controller, connecting to {args.net} ...")
    ChannelFactory.Instance().Init(0, args.net)

    with Controller(cfg_file) as controller:
        time.sleep(2)  # Wait for channels to initialize
        print("Initialization complete.")
        controller.start_custom_mode_conditionally()
        controller.start_rl_gait_conditionally()

        try:
            while controller.running:
                controller.run()
            controller.client.ChangeMode(RobotMode.kDamping)
        except KeyboardInterrupt:
            print("\nKeyboard interrupt received. Cleaning up...")
            controller.cleanup()
