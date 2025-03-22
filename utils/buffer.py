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

        self.tensor_dict["obses"] = torch.tensor(npz_data['observation'], device=self.device)
        self.tensor_dict["privileged_obses"] = torch.tensor(npz_data['critic_observation'], device=self.device)
        self.tensor_dict["actions"] = torch.tensor(npz_data['action'], device=self.device)
        self.tensor_dict["rewards"] = torch.tensor(npz_data['reward'], device=self.device)

        self.tensor_dict["values"] = torch.tensor(npz_data['value'], device=self.device)
        self.tensor_dict["mu"] = torch.tensor(npz_data['mu'], device=self.device)
        if npz_data['mu'].shape == npz_data['std'].shape:
            self.tensor_dict["std"] = torch.tensor(npz_data['std'], device=self.device)
        else:
            self.tensor_dict["std"] = torch.tensor(npz_data['std'], device=self.device).repeat(1, self.num_envs, 1)
        self.tensor_dict["last_values"] = torch.tensor(npz_data['last_value'], device=self.device)
        self.tensor_dict["dones"] = torch.tensor(npz_data['done'], device=self.device)

    def __len__(self):
        return len(self.tensor_dict)

    def __getitem__(self, buf_name):
        return self.tensor_dict[buf_name]

    def keys(self):
        return self.tensor_dict.keys()
