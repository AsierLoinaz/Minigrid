# initial comit
# prueba
import torch 
import gymnasium as gym
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from minigrid.wrappers import ImgObsWrapper

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)


# Environment name and good completions to advance to the next env
envList = {"MiniGrid-ObstructedMaze-1Dl-v0" : 10,
           "MiniGrid-ObstructedMaze-1Dlh-v0": 15,
           "MiniGrid-ObstructedMaze-1Dlhb-v0": 25,
           "MiniGrid-ObstructedMaze-2Dlhb-v1": 50,
           "MiniGrid-ObstructedMaze-Full-v1": 100}

env = gym.make("MiniGrid-ObstructedMaze-1Dl-v0", render_mode="rgb_array")
env = ImgObsWrapper(env)



model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate=0.001)
model.learn(int(1e5), progress_bar=True)