import gymnasium as gym
from gymnasium import spaces
from minigrid.wrappers import ImgObsWrapper
import torch
import torch.nn as nn
import optuna
from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Parámetros generales
env_name = "MiniGrid-ObstructedMaze-1Dl-v0"
checkpoint_dir = './checkpoints/'  # Carpeta para guardar checkpoints
n_trials = 50  # Número de iteraciones de Optuna

# Definición del entorno con los wrappers necesarios
def make_env(env_name):
    env = gym.make(env_name, render_mode="rgb_array")
    env = ImgObsWrapper(env)  # Convierte la observación en imágenes
    env = Monitor(env)  # Permite el registro de métricas
    return env

# Extractor de características personalizado
class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
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

        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        if observations.ndim == 3:
            observations = observations.unsqueeze(0)  # Agregar dimensión batch
        return self.linear(self.cnn(observations))

# Evaluación de políticas
def evaluate_policy(model, env, n_episodes=10):
    total_reward = 0
    total_steps = 0
    for _ in range(n_episodes):
        obs = env.reset()
        done = [False]
        episode_reward = 0
        episode_steps = 0
        while not all(done):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, done, _ = env.step(action)
            episode_reward += rewards[0]
            episode_steps += 1
        total_reward += episode_reward
        total_steps += episode_steps
    avg_reward = total_reward / n_episodes
    avg_steps = total_steps / n_episodes
    return avg_reward, avg_steps

# Objetivo para Optuna
def objective(trial):
    # Hiperparámetros
    algo_name = trial.suggest_categorical("algo_name", ["PPO", "A2C", "DQN"])
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    n_epochs = trial.suggest_categorical("n_epochs", [10, 50, 100])
    gamma = trial.suggest_float("gamma", 0.8, 0.99, step=0.05)
    clip_range = trial.suggest_float("clip_range", 0.0, 0.5, step=0.05)

    # Crear entorno con wrappers
    env = DummyVecEnv([lambda: make_env(env_name)])

    # Elegir algoritmo
    if algo_name == "PPO":
        algo = PPO("MlpPolicy", env, learning_rate=lr, batch_size=batch_size, n_epochs=n_epochs, gamma=gamma, clip_range=clip_range, verbose=0)
    elif algo_name == "A2C":
        algo = A2C("MlpPolicy", env, learning_rate=lr, gamma=gamma, verbose=0)
    elif algo_name == "DQN":
        algo = DQN("MlpPolicy", env, learning_rate=lr, gamma=gamma, batch_size=batch_size, verbose=0)

    try:
        # Entrenar el modelo
        algo.learn(total_timesteps=1e5)
        # Evaluar el modelo
        avg_reward, _ = evaluate_policy(algo, env, n_episodes=10)
        return avg_reward
    except Exception as e:
        return -float('inf')  # Penalización si algo falla

# Crear estudio de Optuna
sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=sampler)

# Optimizar hiperparámetros
study.optimize(objective, n_trials=n_trials)

# Entrenar el mejor modelo
best_params = study.best_params
env = DummyVecEnv([lambda: make_env(env_name)])
policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

if best_params["algo_name"] == "PPO":
    model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, learning_rate=best_params["learning_rate"],
                batch_size=best_params["batch_size"], n_epochs=best_params["n_epochs"],
                gamma=best_params["gamma"], clip_range=best_params["clip_range"], verbose=1)
elif best_params["algo_name"] == "A2C":
    model = A2C("CnnPolicy", env, policy_kwargs=policy_kwargs, learning_rate=best_params["learning_rate"],
                gamma=best_params["gamma"], verbose=1)
elif best_params["algo_name"] == "DQN":
    model = DQN("CnnPolicy", env, policy_kwargs=policy_kwargs, learning_rate=best_params["learning_rate"],
                batch_size=best_params["batch_size"], gamma=best_params["gamma"], verbose=1)

# Guardar modelo entrenado
checkpoint_callback = CheckpointCallback(save_freq=10_000, save_path=checkpoint_dir, name_prefix=best_params["algo_name"].lower())
model.learn(total_timesteps=1e6, callback=checkpoint_callback)
model.save(f"{best_params['algo_name'].lower()}_minigrid_model")
env.close()

print("Entrenamiento completo.")
