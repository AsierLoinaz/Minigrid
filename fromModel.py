import os
import pandas as pd
import torch
import gymnasium as gym
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from minigrid.wrappers import ImgObsWrapper
import numpy as np
from minigrid.manual_control import ManualControl


from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback

# Custom class por Early Stopping if the thershold is met
class EarlyStoppingCallback(BaseCallback):
    def __init__(self, log_dir, completions, threshold, verbose=0):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.completions = completions
        self.threshold = threshold

    def _on_step(self) -> bool:
        average_reward = calculate_average_reward(self.log_dir, self.completions)
        if average_reward >= self.threshold:
            print(f"Threshold alcanzado: {average_reward}")
            return False  # Detener entrenamiento
        return True

def calculate_average_reward(directory, num_episodes):
    rewards_file = os.path.join(directory, "monitor.csv")
    if not os.path.exists(rewards_file):
        return -float("inf")
    data = pd.read_csv(rewards_file, skiprows=1)
    if len(data) < num_episodes:
        return -float("inf")
    return data["r"][-num_episodes:].mean()



# Clase personalizada para el extractor de características
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

        # Calcular la forma con una pasada de prueba
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))


# Configuración de políticas
policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

# Directorio para los logs y modelos
log_dir = "logs/curriculum_learning/"
model_dir = "models/curriculum_learning/"

env_list = [
    {"environment": "MiniGrid-ObstructedMaze-1Dlh-v0", "n_steps": 2e6, "completions": 200, "threshold": 0.8},
    {"environment": "MiniGrid-ObstructedMaze-1Dlhb-v0", "n_steps": 10e6, "completions": 150, "threshold": 0.7},
    {"environment": "MiniGrid-ObstructedMaze-2Dl-v0", "n_steps": 10e6, "completions": 150, "threshold": 0.7},
    {"environment": "MiniGrid-ObstructedMaze-2Dlh-v0", "n_steps": 10e6, "completions": 150, "threshold": 0.7},
    {"environment": "MiniGrid-ObstructedMaze-1Dlhb-v0", "n_steps": 10e6, "completions": 150, "threshold": 0.7},
    {"environment": "MiniGrid-ObstructedMaze-2Dlhb-v0", "n_steps": 10e6, "completions": 100, "threshold": 0.7},
    {"environment": "MiniGrid-ObstructedMaze-Full-v1", "n_steps": 15e6, "completions": 100, "threshold": 0.7}
    
]
# Crear un DataFrame para visualizar los entornos
env_df = pd.DataFrame(env_list)
env_df = env_df.reset_index(drop=True)

# Proceso de Curriculum Learning
model_type = "ppo"  # Cambia esto a "ppo", "dqn", etc., según el modelo que uses.
MODEL = PPO

# Ruta del modelo inicial preentrenado
initial_model_path = "models/curriculum_learning/MiniGrid_ObstructedMaze_1Dlh_v0_ppo/MiniGrid_ObstructedMaze_1Dlh_v0_ppo_final.zip"

for idx, row in env_df.iterrows():
    env_name = row["environment"]
    completions = row["completions"]
    threshold = row["threshold"]
    n_steps = row["n_steps"]

    print(f"Entrenando en el entorno: {env_name} con {n_steps} steps usando {model_type.upper()}")

    # Crear carpeta específica para el entorno y el modelo
    env_model_dir = os.path.join(model_dir, f"{env_name.replace('-', '_')}_{model_type}")
    os.makedirs(env_model_dir, exist_ok=True)

    # Configurar el entorno con Monitor
    env = gym.make(env_name, render_mode="rgb_array")
    # env = gym.make(env_name, render_mode="human")
    env = ImgObsWrapper(env)
    env_dir = os.path.join(log_dir, env_name)
    os.makedirs(env_dir, exist_ok=True)

    env = Monitor(env, env_dir)

    # enable manual control for testing
    # env = gym.make(env_name, render_mode="human")
    # manual_control = ManualControl(env, seed=42)
    # manual_control.start()
    
    # Configuración del tensorboard
    tensorboard_log = f"{log_dir}/tensorboard/{model_type}_{env_name}"

    if idx == 0:
        # En la primera iteración, cargar el modelo preentrenado
        print(f"Cargando modelo inicial preentrenado desde: {initial_model_path}")
        model = MODEL.load(initial_model_path, env=env)
        model.tensorboard_log = tensorboard_log
    else:
        # En las siguientes iteraciones, cargar el modelo entrenado previamente en el paso anterior
        previous_env_name = env_df.loc[idx - 1, "environment"]
        previous_env_model_dir = os.path.join(model_dir, f"{previous_env_name.replace('-', '_')}_{model_type}")
        previous_model_path = os.path.join(previous_env_model_dir, f"{previous_env_name.replace('-', '_')}_{model_type}_final.zip")
        print(f"Cargando modelo previo: {previous_model_path}")
        model = MODEL.load(previous_model_path, env=env)
        model.tensorboard_log = tensorboard_log

    # Callback para checkpoints cada 20k pasos
    checkpoint_callback = CheckpointCallback(
        save_freq=20000,  # Guardar cada 20k pasos
        save_path=env_model_dir,  # Directorio específico para este entorno y modelo
        name_prefix=f"{env_name.replace('-', '_')}_{model_type}",  # Prefijo con el tipo de modelo
    )

    # Callback para early stopping
    early_stopping_callback = EarlyStoppingCallback(env_dir, completions, threshold)

    # Entrenar hasta superar el threshold
    average_reward = -float("inf")
    total_steps = 0
    while average_reward < threshold:
        model.learn(
            total_timesteps=int(n_steps - total_steps),  # Asegurarse de no exceder n_steps
            progress_bar=True,
            callback=[checkpoint_callback, early_stopping_callback],  # Usar ambos callbacks
        )
        total_steps += int(n_steps - total_steps)
        print(f"Entrenamiento acumulado: {total_steps} steps")

        # Calcular la recompensa promedio
        average_reward = calculate_average_reward(env_dir, completions)
        print(f"Recompensa promedio de las últimas {completions} iteraciones: {average_reward}")

    # Guardar el modelo final
    final_model_path = os.path.join(env_model_dir, f"{env_name.replace('-', '_')}_{model_type}_final.zip")
    model.save(final_model_path)
    print(f"Modelo final guardado como: {final_model_path}")
