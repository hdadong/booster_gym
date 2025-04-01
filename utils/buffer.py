import torch
import numpy as np

class ExperienceBuffer:

    def __init__(self, horizon_length, num_envs, device):
        self.tensor_dict = {}
        self.horizon_length = horizon_length
        self.num_envs = num_envs
        self.device = device

    def add_buffer(self, name, shape, dtype=None):
        self.tensor_dict[name] = torch.zeros(self.horizon_length, self.num_envs, *shape, dtype=dtype, device=self.device)

    def update_data(self, name, idx, data):
        self.tensor_dict[name][idx, :] = data

    def load_data(self, npz_file):
        npz_data = np.load(npz_file)

        self.tensor_dict["obses"] = torch.tensor(npz_data['observations'], device=self.device)
        self.tensor_dict["privileged_obses"] = torch.tensor(npz_data['pri_observations'], device=self.device)
        self.tensor_dict["actions"] = torch.tensor(npz_data['actions'], device=self.device)
        self.tensor_dict["rewards"] = torch.tensor(npz_data['rewards'], device=self.device)


        if "last_obs_all" in npz_data.keys():
            self.tensor_dict["last_obses_all"] = torch.tensor(npz_data['last_obs_all'], device=self.device)
        else:
            self.tensor_dict["last_obses"] = torch.tensor(npz_data['last_observations'], device=self.device)
            self.tensor_dict["last_pri_obses"] = torch.tensor(npz_data['last_pri_observations'], device=self.device)

        self.tensor_dict["dones"] = torch.tensor(npz_data['dones'], device=self.device)
        self.tensor_dict["time_outs"] = torch.tensor(npz_data['timeouts'], device=self.device)
        

    def __len__(self):
        return len(self.tensor_dict)

    def __getitem__(self, buf_name):
        return self.tensor_dict[buf_name]

    def keys(self):
        return self.tensor_dict.keys()
