import os
import glob
import yaml
import argparse
import numpy as np
import random
import time
import signal
import imageio
import torch
import torch.nn.functional as F
import shutil
import gc
from utils.model_mbrl import *
from utils.buffer import ExperienceBuffer
from utils.utils import compute_returns_advantages, surrogate_loss
from utils.recorder import Recorder
from envs import *
from datetime import datetime
from torch.distributions import Normal


def get_latest_data_dir(base_data_dir, processed_dirs):
    data_dirs = [d for d in os.listdir(base_data_dir) if d.startswith('data_') and d not in processed_dirs]
    if not data_dirs:
        return None
    data_dirs.sort()
    latest_data_dir = data_dirs[-1]
    return os.path.join(base_data_dir, latest_data_dir)

class Runner:

    def __init__(self, test=False):
        self.test = test
        self.policy_path = None
        # prepare the environment
        self._get_args()
        self._update_cfg_from_args()
        self._set_seed()
        self.task_class = eval(self.cfg["basic"]["task"])

        self.device = self.cfg["basic"]["rl_device"]
        self.learning_rate = self.cfg["algorithm"]["learning_rate"]
        self.model = ActorCritic(self.cfg["env"]["num_actions"], self.cfg["env"]["num_observations"] * self.cfg["env"]["obs_frame_stack"], self.cfg["env"]["num_privileged_obs"]* self.cfg["env"]["pri_obs_frame_stack"]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self._load()

        self.buffer = ExperienceBuffer(self.cfg["runner"]["horizon_length"], self.cfg["env"]["num_envs"], self.device)
        self.buffer.add_buffer("actions", (self.cfg["env"]["num_actions"],))
        self.buffer.add_buffer("obses", (self.cfg["env"]["num_observations"],))
        self.buffer.add_buffer("privileged_obses", (self.cfg["env"]["num_privileged_obs"],))
        self.buffer.add_buffer("rewards", ())
        self.buffer.add_buffer("dones", (), dtype=bool)
        self.buffer.add_buffer("time_outs", (), dtype=bool)




    def _get_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--task", required=True, type=str, help="Name of the task to run.")
        parser.add_argument("--checkpoint", type=str, help="Path of the model checkpoint to load. Overrides config file if provided.")
        parser.add_argument("--num_envs", type=int, help="Number of environments to create. Overrides config file if provided.")
        parser.add_argument("--headless", type=bool, help="Run headless without creating a viewer window. Overrides config file if provided.")
        parser.add_argument("--sim_device", type=str, help="Device for physics simulation. Overrides config file if provided.")
        parser.add_argument("--rl_device", type=str, help="Device for the RL algorithm. Overrides config file if provided.")
        parser.add_argument("--seed", type=int, help="Random seed. Overrides config file if provided.")
        parser.add_argument("--max_iterations", type=int, help="Maximum number of training iterations. Overrides config file if provided.")
        self.args = parser.parse_args()

    # Override config file with args if needed
    def _update_cfg_from_args(self):
        cfg_file = os.path.join("envs", "{}.yaml".format(self.args.task))
        with open(cfg_file, "r", encoding="utf-8") as f:
            self.cfg = yaml.load(f.read(), Loader=yaml.FullLoader)
        for arg in vars(self.args):
            if getattr(self.args, arg) is not None:
                if arg == "num_envs":
                    self.cfg["env"][arg] = getattr(self.args, arg)
                else:
                    self.cfg["basic"][arg] = getattr(self.args, arg)
        if not self.test:
            self.cfg["viewer"]["record_video"] = False

    def _set_seed(self):
        if self.cfg["basic"]["seed"] == -1:
            self.cfg["basic"]["seed"] = np.random.randint(0, 10000)
        print("Setting seed: {}".format(self.cfg["basic"]["seed"]))

        random.seed(self.cfg["basic"]["seed"])
        np.random.seed(self.cfg["basic"]["seed"])
        torch.manual_seed(self.cfg["basic"]["seed"])
        os.environ["PYTHONHASHSEED"] = str(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed(self.cfg["basic"]["seed"])
        torch.cuda.manual_seed_all(self.cfg["basic"]["seed"])

    def _load(self):
        if not self.cfg["basic"]["checkpoint"]:
            return
        if self.policy_path == None:
            if (self.cfg["basic"]["checkpoint"] == "-1") or (self.cfg["basic"]["checkpoint"] == -1):
                newest_dir = sorted(glob.glob(os.path.join("logs", "*")), key=lambda x: x.split('/')[-1], reverse=True)[0]
                self.cfg["basic"]["checkpoint"] = sorted(glob.glob(os.path.join(newest_dir, "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
            self.policy_path = newest_dir.split('/')[1]
        else:
            if (self.cfg["basic"]["checkpoint"] == "-1") or (self.cfg["basic"]["checkpoint"] == -1):
                self.cfg["basic"]["checkpoint"] = sorted(glob.glob(os.path.join("logs/"+self.policy_path+"/", "**/*.pth"), recursive=True), key=os.path.getmtime)[-1]
        print("self.policy_path", self.policy_path)
        print("Loading model from {}".format(self.cfg["basic"]["checkpoint"]))
        model_dict = torch.load(self.cfg["basic"]["checkpoint"], map_location=self.device, weights_only=True)
        self.model.load_state_dict(model_dict["model"], strict=False)
        try:
            self.env.curriculum_prob = model_dict["curriculum"]
        except Exception as e:
            print(f"Failed to load curriculum: {e}")
        try:
            self.optimizer.load_state_dict(model_dict["optimizer"])
            self.learning_rate = model_dict["learning_rate"]
        except Exception as e:
            print(f"Failed to load optimizer: {e}")
        del model_dict
    def train(self):
        self.recorder = Recorder(self.cfg, name="policy_train")


        self.recorder.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "learning_rate": self.learning_rate,
            },
            0,
        )
        self.cfg["basic"]["checkpoint"] = -1
        self._load()
        generate_data_dir = os.path.join("logs/", os.path.join(self.policy_path, 'generate_data'))
        base_data_dir = os.path.join("logs/", os.path.join(self.policy_path, 'base_data_dir'))

        os.makedirs(base_data_dir, exist_ok=True)
        os.makedirs(generate_data_dir, exist_ok=True)


        flag_policy_train = os.path.join(os.path.join("logs/", self.policy_path), 'flag_policy_train.flag')
        processed_dirs = set()

        for it in range(self.cfg["basic"]["max_iterations"]):
            # within horizon_length, env.step() is called with same act
            print(f"[{datetime.now()}] waitting for world model generate data")

            while not os.path.exists(flag_policy_train):
                time.sleep(0.1)
            data_dir = get_latest_data_dir(generate_data_dir, processed_dirs)
            processed_dirs.add(os.path.basename(data_dir))
            print(f"[{datetime.now()}] Processing data from {data_dir}")

            self.buffer.load_data(npz_file=data_dir)

            # Compute returns and advantages using old value
            with torch.no_grad():
                values = self.model.est_value_all(self.buffer["privileged_obses"])
                last_values = self.model.est_value_all(self.buffer["last_obses_all"])
                returns, advantages = compute_returns_advantages(
                    self.buffer["rewards"],
                    self.buffer["dones"],
                    values,
                    last_values,
                    self.cfg["algorithm"]["gamma"],
                    self.cfg["algorithm"]["lam"],
                )
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            old_dist = self.model.act(self.buffer["obses"])
            old_mu = old_dist.loc
            old_sigma = old_dist.scale
            old_log_prob = old_dist.log_prob(self.buffer["actions"]).sum(dim=-1)

            obs = self.buffer["obses"]
            pri_obs = self.buffer["privileged_obses"]
            actions = self.buffer["actions"]
            dones = self.buffer["dones"]
            mean_kl = 0.0
            mean_value_loss = 0
            mean_actor_loss = 0
            mean_bound_loss = 0
            mean_entropy = 0
            for epoch in range(self.cfg["runner"]["mini_epochs"]):
                indices = torch.randperm(obs.shape[0] * obs.shape[1], device=self.device)
                batch_size = obs.shape[0] * obs.shape[1]
                mini_batch_size = batch_size // self.cfg["runner"]["mini_batches"]

                for i in range(self.cfg["runner"]["mini_batches"]):
                    start = i * mini_batch_size
                    end = (i + 1) * mini_batch_size
                    batch_idx = indices[start:end]


                    obs_batch = obs.reshape(-1, obs.shape[-1])[batch_idx]
                    pri_obs_batch = pri_obs.reshape(-1, pri_obs.shape[-1])[batch_idx]
                    actions_batch = actions.reshape(-1, actions.shape[-1])[batch_idx]
                    returns_batch = returns.reshape(-1)[batch_idx]
                    advantages_batch = advantages.reshape(-1)[batch_idx]
                    old_log_prob_batch = old_log_prob.reshape(-1)[batch_idx]
                    old_mu_batch = old_mu.reshape(-1, old_mu.shape[-1])[batch_idx]
                    old_sigma_batch = old_sigma.reshape(-1, old_sigma.shape[-1])[batch_idx]
                    # current policy distribution
                    dist = self.model.act(obs_batch)
                    value = self.model.est_value_all(pri_obs_batch)

                    # loss
                    new_log_prob = dist.log_prob(actions_batch).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1)
                    policy_loss = surrogate_loss(old_log_prob_batch, new_log_prob, advantages_batch)

                    value_loss = F.mse_loss(value, returns_batch)

                    bound_loss = torch.clip(dist.loc - 1.0, min=0.0).square().mean() + torch.clip(dist.loc + 1.0, max=0.0).square().mean()

                    loss = (
                        policy_loss
                        + value_loss
                        + self.cfg["algorithm"]["entropy_coef"] * entropy.mean()
                        + self.cfg["algorithm"]["bound_coef"] * bound_loss
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                    with torch.no_grad():
                        dist = self.model.act(obs_batch)

                        eps = 1e-6
                        kl = torch.sum(
                            torch.log((dist.scale + eps) / (old_sigma_batch + eps))
                            + 0.5 * (torch.square(old_sigma_batch) + torch.square(dist.loc - old_mu_batch)) / (torch.square(dist.scale) + eps)
                            - 0.5,
                            axis=-1,
                        )
                        kl_mean = torch.mean(kl)
                        if kl_mean > self.cfg["algorithm"]["desired_kl"] * 2:
                            self.original_learning_rate = self.learning_rate
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                            print(f"Reducing learning rate {self.original_learning_rate} to {self.learning_rate} due to high KL divergence: {kl_mean}")
                        elif kl_mean < self.cfg["algorithm"]["desired_kl"] / 2:
                            self.original_learning_rate = self.learning_rate
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                            print(f"Increasing learning rate {self.original_learning_rate} to {self.learning_rate} due to low KL divergence: {kl_mean}")
                        for param_group in self.optimizer.param_groups:
                            param_group["lr"] = self.learning_rate
                        mean_value_loss += value_loss.item()
                        mean_actor_loss += policy_loss.item()
                        mean_bound_loss += bound_loss.item()
                        mean_entropy += entropy.mean()
                        mean_kl += kl_mean.item()

            mean_value_loss /= self.cfg["runner"]["mini_batches"] * self.cfg["runner"]["mini_epochs"]
            mean_actor_loss /= self.cfg["runner"]["mini_batches"] * self.cfg["runner"]["mini_epochs"]
            mean_bound_loss /= self.cfg["runner"]["mini_batches"] * self.cfg["runner"]["mini_epochs"]
            mean_entropy /= self.cfg["runner"]["mini_batches"] * self.cfg["runner"]["mini_epochs"]
            mean_kl /= self.cfg["runner"]["mini_batches"] * self.cfg["runner"]["mini_epochs"]
            self.recorder.record_statistics({
                "value_loss": mean_value_loss,
                "actor_loss": mean_actor_loss,
                "bound_loss": mean_bound_loss,
                "entropy": mean_entropy,
                "kl_mean": mean_kl,
                "lr": self.learning_rate,

            }, it)

            self.recorder.save({
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "learning_rate": self.learning_rate,
            }, it + 1)
            print("epoch: {}/{}".format(it + 1, self.cfg["basic"]["max_iterations"]))

            os.remove(data_dir)
            os.remove(flag_policy_train)
            print("Train done, data/flag cleared.")

    def play(self):
        self.cfg["basic"]["checkpoint"] = -1
        self._load()
        base_data_dir = os.path.join("logs/", os.path.join(self.policy_path, 'base_data_dir'))
        flag_model_train = os.path.join(os.path.join("logs/", self.policy_path), 'flag_model_train.flag')
        while os.path.exists(flag_model_train):
            time.sleep(3)  # 每隔10秒检查一次


        self.cfg["env"]["num_envs"] = 1

        self.env = self.task_class(self.cfg)
        plot_reward = False
        max_total_steps = self.cfg["basic"]["max_total_steps"]
        num_envs = self.env.num_envs
        step_count = 0
        num_collect = 0
        data_buffers = [self.create_data_dict(plot_reward=plot_reward, reward_names=self.env.reward_names) for _ in range(num_envs)]
        data_counts = [0] * num_envs
        data_dir = os.path.join(base_data_dir, 'data_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(data_dir, exist_ok=True)
        self.recorder = Recorder(self.cfg, name="collect_data", log_path="collect_data")



        obs, infos = self.env.reset()
        obs = obs.to(self.device)
        obs_cpu = obs.cpu().numpy()
        pri_obs_cpu = infos["privileged_obs"].cpu().numpy()

        while True:

            with torch.no_grad():
                episode_length_buf = self.env.episode_length_buf.cpu().numpy().copy()
                dist = self.model.act(obs)
                act = dist.loc
                obs, rew, done, infos = self.env.step(act.detach())
                obs, rew, done = obs.to(self.device), rew.to(self.device), done.to(self.device)
                time_outs = infos["time_outs"].to(self.device)
                ep_info = {"reward": rew}
                ep_info.update(infos["rew_terms"])

            actions_cpu = act.detach().cpu().numpy()
            torques_cpu = self.env.torques.cpu().numpy()
            rews_cpu = rew.cpu().numpy()
            current_time = time.time()
            contact_cpu = self.env.feet_contact.cpu().numpy()
            sub_reward_dict = {}
            for env_idx in range(obs.shape[0]):
                data_buffers[env_idx]['observations'].append(obs_cpu[env_idx])
                data_buffers[env_idx]['pri_observations'].append(pri_obs_cpu[env_idx])
                data_buffers[env_idx]['actions'].append(actions_cpu[env_idx])
                data_buffers[env_idx]['torques'].append(torques_cpu[env_idx])
                data_buffers[env_idx]['contacts'].append(contact_cpu[env_idx])
                data_buffers[env_idx]['rewards'].append(rews_cpu[env_idx])
                data_buffers[env_idx]['timestamps'].append(current_time)
                if plot_reward:
                    for i in range(len(self.env.reward_functions)):
                        name = self.env.reward_names[i]
                        sub_reward_dict['reward'+name] = infos['rew_terms'][name]
                        name = self.env.reward_names[i]
                        data_buffers[env_idx]['reward_' +name].append(sub_reward_dict['reward'+name][env_idx].item())
                        if "height" in name:
                            print("height", infos['rew_terms'][name])
            del sub_reward_dict
            env_ids = done.nonzero(as_tuple=False).flatten()
            obs_cpu = obs.cpu().numpy()
            pri_obs_cpu = infos["privileged_obs"].cpu().numpy()

            for env_id in env_ids:
                # save npz
                npz_filename = os.path.join(
                    data_dir,
                    f'env_{env_id}_data_{data_counts[env_id]}.npz',
                )
                print("npz_filename", npz_filename)
                obs_array = np.array(data_buffers[env_id]['observations'], dtype=np.float32)
                pri_obs_array = np.array(data_buffers[env_id]['pri_observations'], dtype=np.float32)
                actions_array = np.array(data_buffers[env_id]['actions'], dtype=np.float32)
                torques_array = np.array(data_buffers[env_id]['torques'], dtype=np.float32)
                contacts_array = np.array(data_buffers[env_id]['contacts'], dtype=np.float32)
                rewards_array = np.array(data_buffers[env_id]['rewards'], dtype=np.float32)
                timestamps_array = np.array(data_buffers[env_id]['timestamps'], dtype=np.float64)

                if plot_reward:
                    reward_comp = {}
                    for name in self.env.reward_names:
                        rew_array = np.array(data_buffers[env_id]['reward_' + name], dtype=np.float32)
                        reward_comp['reward_'+name] = rew_array
                    np.savez_compressed(
                        npz_filename,
                        observations=obs_array,
                        pri_observations=pri_obs_array,
                        actions=actions_array,
                        torques=torques_array,
                        contacts=contacts_array,
                        rewards=rewards_array,
                        timestamps=timestamps_array,
                        **reward_comp,
                    )
                else:
                    np.savez_compressed(
                        npz_filename,
                        observations=obs_array,
                        pri_observations=pri_obs_array,
                        actions=actions_array,
                        torques=torques_array,
                        contacts=contacts_array,
                        rewards=rewards_array,
                        timestamps=timestamps_array,
                    )
                data_counts[env_id] += 1
                step_count += episode_length_buf[env_id]
                # 清空该环境的数据缓冲区
                del obs_array, pri_obs_array, actions_array, torques_array, contacts_array, rewards_array, timestamps_array
                data_buffers[env_id]['observations'].clear()
                data_buffers[env_id]['pri_observations'].clear()
                data_buffers[env_id]['actions'].clear()
                data_buffers[env_id]['torques'].clear()
                data_buffers[env_id]['contacts'].clear()
                data_buffers[env_id]['rewards'].clear()
                data_buffers[env_id]['timestamps'].clear()
                if plot_reward:
                    for i in range(len(self.env.reward_functions)):
                        name = self.env.reward_names[i]
                        data_buffers[env_idx]['reward_' +name].clear()
                data_buffers[env_id] = self.create_data_dict(plot_reward=plot_reward, reward_names=self.env.reward_names)

                # load npz and check the size of the data
                npz_filename = os.path.join(
                    data_dir,
                    f'env_{env_id}_data_{data_counts[env_id]-1}.npz',
                )
                
                print("episode_length_buf[env_id]", episode_length_buf[env_id])
            
            self.recorder.record_episode_statistics(done, ep_info, num_collect, step_count > max_total_steps)

            if step_count > max_total_steps:# or episode_length_buf[env_id]!= 1000:
                num_collect +=1

                # TODO: reset the env for the new policy
                # data_buffers = [self.create_data_dict(plot_reward=plot_reward, reward_names=self.env.reward_names) for _ in range(num_envs)]
                # obs, infos = self.env.reset()
                # obs = obs.to(self.device)
                # obs_cpu = obs.cpu().numpy()
                # pri_obs_cpu = infos["privileged_obs"].cpu().numpy()


                step_count = 0
                torch.cuda.empty_cache()
                gc.collect()
                print("Collect Done, waiting for world model and policy training。")
                open(flag_model_train, 'w').close()
                while os.path.exists(flag_model_train):
                    time.sleep(3)  # 每隔10秒检查一次
                print("Train Done. Collecting data.")
                self.cfg["basic"]["checkpoint"] = -1
                self._load() # load the newest model
                data_dir = os.path.join(base_data_dir, 'data_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
                os.makedirs(data_dir, exist_ok=True)
                data_counts = [0] * num_envs


    def create_data_dict(self, plot_reward=False, reward_names=[]):
        d = {
            'observations': [],
            'pri_observations': [],
            'actions': [],
            'torques': [],
            'contacts': [],
            'rewards': [],
            'timestamps': []
        }
        if plot_reward:
            d.update({f'reward_{name}': [] for name in reward_names})
        return d



    def interrupt_handler(self, signal, frame):
        print("\nInterrupt received, waiting for video to finish...")
        self.interrupt = True
