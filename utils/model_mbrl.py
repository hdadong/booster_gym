import torch
import torch.nn.functional as F


class ActorCritic(torch.nn.Module):

    def __init__(self, num_act, num_obs, num_privileged_obs):
        super().__init__()
        self.critic = torch.nn.Sequential(
            torch.nn.Linear(num_privileged_obs, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 1),
        )
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(num_obs, 256),
            torch.nn.SiLU(),
            torch.nn.Linear(256, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, 128),
            torch.nn.SiLU(),
            torch.nn.Linear(128, num_act),
        )
        self.std = torch.nn.parameter.Parameter(torch.full((1, num_act), fill_value=0.02), requires_grad=True)

    def act(self, obs):
        action_mean = self.actor(obs)
        action_std = self.std.expand_as(action_mean)
        return torch.distributions.Normal(action_mean, action_std)

    def est_value_all(self, obs_privileged_obs):
        return self.critic(obs_privileged_obs).squeeze(-1)
    
    def est_value(self, obs, privileged_obs):
        critic_input = torch.cat((obs, privileged_obs), dim=-1)
        return self.critic(critic_input).squeeze(-1)

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

