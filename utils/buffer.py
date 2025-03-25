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

        # self.tensor_dict["values"] = torch.tensor(npz_data['value'], device=self.device)
        # self.tensor_dict["mu"] = torch.tensor(npz_data['mu'], device=self.device)
        # if npz_data['mu'].shape == npz_data['std'].shape:
        #     self.tensor_dict["std"] = torch.tensor(npz_data['std'], device=self.device)
        # else:
        #     self.tensor_dict["std"] = torch.tensor(npz_data['std'], device=self.device).repeat(1, self.num_envs, 1)
        self.tensor_dict["last_values"] = torch.tensor(npz_data['last_value'], device=self.device)
        self.tensor_dict["dones"] = torch.tensor(npz_data['dones'], device=self.device)

        print("1",self.tensor_dict["obses"].shape)
        print("2",self.tensor_dict["privileged_obses"].shape)
        print("3",self.tensor_dict["actions"].shape)
        print("4",self.tensor_dict["rewards"].shape)
        print("5",self.tensor_dict["last_values"].shape)
        print("6",self.tensor_dict["dones"].shape)
        # 1 torch.Size([40, 400, 47])
        # 2 torch.Size([40, 400, 13])
        # 3 torch.Size([40, 400, 12])
        # 4 torch.Size([40, 400])
        # 5 torch.Size([400])
        # 6 torch.Size([40, 400])
    def __len__(self):
        return len(self.tensor_dict)

    def __getitem__(self, buf_name):
        return self.tensor_dict[buf_name]

    def keys(self):
        return self.tensor_dict.keys()
