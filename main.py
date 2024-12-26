import os
import torch 
import gymnasium as gym
import torch.nn as nn
from gymnasium import spaces
import pandas as pd

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
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

# Directorio para los logs de TensorBoard
log_dir = "logs/curriculum_learning/"
model_dir = "models/curriculum_learning/"
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
# Configurar el entorno con Monitor
env_list = [
    {"environment": "MiniGrid-ObstructedMaze-1Dl-v0", "n_steps": 5e5, "completions": 10, "threshold": 0.9},
    {"environment": "MiniGrid-ObstructedMaze-1Dlh-v0", "n_steps": 7e5, "completions": 15, "threshold": 0.8},
    {"environment": "MiniGrid-ObstructedMaze-1Dlhb-v0", "n_steps": 1e6, "completions": 25, "threshold": 0.85},
    {"environment": "MiniGrid-ObstructedMaze-2Dlhb-v1", "n_steps": 2e6, "completions": 50, "threshold": 0.75},
    {"environment": "MiniGrid-ObstructedMaze-Full-v1", "n_steps": 3e6, "completions": 100, "threshold": 0.95},
]

# Crear un DataFrame para visualizar los entornos
env_df = pd.DataFrame(env_list)


env = gym.make("MiniGrid-ObstructedMaze-1Dl-v0", render_mode="rgb_array")

env = ImgObsWrapper(env)
# Crear modelo con logging habilitado
model = PPO(
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    learning_rate=0.01,
    tensorboard_log=log_dir  # Habilitar logging para TensorBoard
)

# PPO, A2C, DQN

# Entrenar el modelo
model.learn(total_timesteps=int(5e5), progress_bar=True)

# Guardar el modelo entrenado
model.save("ppo_minigrid")
