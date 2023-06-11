import parl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# clamp bounds for Std of action_log
LOG_SIG_MAX = 2.0
LOG_SIG_MIN = -20.0

__all__ = ['TorchCNNModel']

def get_pretrained_model(model='resnet18', weight='IMAGENET1K_V1', freeze=True):
    if model == 'resnet18':
        model = torchvision.models.resnet18(weights=weight)
    num_ftrs = model.fc.in_features
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
    return model, num_ftrs

class TorchCNNModel(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(TorchCNNModel, self).__init__()
        self.actor_model = Actor(3, action_dim)
        self.critic_model = Critic(3, action_dim)

    def policy(self, obs):
        return self.actor_model(obs)

    def value(self, obs, action):
        return self.critic_model(obs, action)


class Actor(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(Actor, self).__init__()

        self.pretrained, self.num_ftrs = get_pretrained_model()

        self.Linear_layer = nn.Sequential(
            nn.Linear(self.num_ftrs, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.pretrained.fc = self.Linear_layer

        self.mean_linear1 = nn.Linear(256, 256)
        self.mean_linear2 = nn.Linear(256, 256)
        self.mean_linear = nn.Linear(256, action_dim)

        self.std_linear1 = nn.Linear(256, 256)
        self.std_linear2 = nn.Linear(256, 256)
        self.std_linear = nn.Linear(256, action_dim)

    def forward(self, obs):
        x = obs[0].permute(0, 3, 1, 2)
        x = self.pretrained(x)

        act_mean = F.relu(self.mean_linear1(x))
        act_mean = F.relu(self.mean_linear2(act_mean))
        act_mean = self.mean_linear(act_mean)

        act_std = F.relu(self.std_linear1(x))
        act_std = F.relu(self.std_linear2(act_std))
        act_std = self.std_linear(act_std)
        act_log_std = torch.clamp(act_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return act_mean, act_log_std


class Critic(parl.Model):
    def __init__(self, obs_dim, action_dim):
        super(Critic, self).__init__()

        self.pretrained, self.num_ftrs = get_pretrained_model()

        self.pretrained.fc = nn.Linear(self.num_ftrs, 256)

        self.Linear_layer = nn.Sequential(
            nn.Linear(256 + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )


    def forward(self, obs, action):
        x = obs[0].permute(0, 3, 1, 2)
        x = self.pretrained(x)
        x = torch.cat([x, action], 1)
        q1 = self.Linear_layer(x)
        q2 = self.Linear_layer(x)

        return q1, q2
