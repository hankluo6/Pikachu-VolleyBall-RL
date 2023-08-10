from stable_baselines3.common import base_class
from gym_pikachu_volleyball.envs import PikachuVolleyballEnv, PikachuVolleyballRandomEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from typing import Optional
from collections import deque
import os
import numpy as np

SEED = 426
N_STEPS = 4096
ENT_COEF = 0.001
VERBOSE = 1
NUM_TIMESTEPS = int(2e7)
EVAL_FREQ = 10000
EVAL_EPISODES = 100
WINDOW_SIZE = 10
LATEST_MODEL_RATIO = 0.5
SWAP_STEPS = 10000
SAVE_STEPS = 20000
INITIAL_ELO = 1200
TENSORBOARD_LOG = "./tensorboard/pikachu-volleyball-selfplay/"
LOGDIR = "models"
set_random_seed(SEED)

class PikachuVolleyballSelfPlayEnv(PikachuVolleyballRandomEnv):
    def __init__(self, is_player1_computer: bool, is_player2_computer: bool, render_mode: str, limited_timestep: int):
        super().__init__(is_player1_computer, is_player2_computer, render_mode, limited_timestep)
        self.model = None
        self.obs = None
        self.current_model = None
        self.next_model = None

    def step(self, action):
        if self.model != None and isinstance(self.obs, np.ndarray):
            other_action, _state = self.model.predict(self.obs, deterministic=True)
            obs, reward, terminated, truncated, info = super().step(action, other_action)
        else:
            obs, reward, terminated, truncated, info = super().step(action)
        self.obs = info['other_obs']
        return obs, reward, terminated, truncated, info
    
    def reset(self, options=None, seed: Optional[int]=None, return_info: bool=False):
        obs, info = super().reset(options, seed, return_info)
        self.obs = info['other_obs']
        return obs, info

class SelfPlayCallback(BaseCallback):
    def __init__(self, swap_steps, save_steps, selfplayAgent):
        super(SelfPlayCallback, self).__init__()
        self.swap_steps = swap_steps
        self.save_steps = save_steps
        self.selfplay_agent = selfplayAgent
        self.best_mean_reward = 0.5
    
    def init_callback(self, model: base_class.BaseAlgorithm) -> None:
        super().init_callback(model)
        self.selfplay_agent.push_snapshot(self.model, 0)
        self.selfplay_agent.swap_snapshots()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_steps == 0:
            self.selfplay_agent.push_snapshot(self.model, self.n_calls)
        if self.num_timesteps % self.swap_steps == 0:
            self.selfplay_agent.swap_snapshots()

        if self.locals['dones'][0]:
            self.logger.record('elo', self.selfplay_agent.update_elo(self.locals['rewards'][0]))

        return True

class SelfplayAgent:
    def __init__(self, play_against_latest_model_ratio, window_size, env, log_path) -> None:
        self.play_against_latest_model_ratio = play_against_latest_model_ratio
        self.env = env
        self.window_size = window_size
        self.agent_elo = INITIAL_ELO
        self.others_elos = deque(maxlen=window_size)
        self.snapshots = deque(maxlen=window_size)
        self.current_opponent = None
        self.log_path = os.path.join(log_path, "history")
    
    def swap_snapshots(self):
        if len(self.snapshots) == 0:
            return
        if np.random.uniform() < (1 - self.play_against_latest_model_ratio):
            x = np.random.randint(len(self.snapshots))
        else:
            x = -1
        snapshot = self.snapshots[x]
        self.env.next_model = snapshot
        self.current_opponent = x
        # load model if it's there
        if self.env.next_model:
            self.env.model = PPO.load(self.env.next_model, env=self.env)
            self.env.current_model = self.env.next_model
            print(f'change model to {self.env.current_model}')

    def push_snapshot(self, model, n_calls):
        name = os.path.join(self.log_path, f"model_history_{n_calls}")
        model.save(name)
        self.snapshots.append(name)
        self.others_elos.append(self.agent_elo)

    def update_elo(self, reward):
        change = self._calculate_elo(self.agent_elo, self.others_elos[self.current_opponent], reward)
        self.agent_elo += change
        self.others_elos[self.current_opponent] -= change
        return self.agent_elo

    def _calculate_elo(self, current, opponent, reward):
        # adjust the reward for elo
        if reward != 1:
            reward = 0
        k = 16
        r1 = pow(10, current / 400)
        r2 = pow(10, opponent / 400)
        summed = r1 + r2
        e1 = r1 / summed
        return k * (reward - e1)
    
if __name__ == '__main__':
    env = PikachuVolleyballSelfPlayEnv(render_mode="rgb_array", is_player1_computer=False, is_player2_computer=True, limited_timestep=1000)
    eval_env = PikachuVolleyballRandomEnv(render_mode="rgb_array", is_player1_computer=False, is_player2_computer=True, limited_timestep=1000)
    selfplayAgent = SelfplayAgent(play_against_latest_model_ratio=LATEST_MODEL_RATIO, window_size=WINDOW_SIZE, log_path=LOGDIR, env=env)
    env = Monitor(env)

    selfplay_callback = SelfPlayCallback(selfplayAgent=selfplayAgent, swap_steps=SWAP_STEPS, save_steps=SAVE_STEPS)
    eval_callback = EvalCallback(eval_env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

    model = PPO("MlpPolicy", env, seed=SEED, verbose=VERBOSE,  n_steps=N_STEPS, ent_coef=ENT_COEF, tensorboard_log=TENSORBOARD_LOG)

    try:
        model.learn(total_timesteps=NUM_TIMESTEPS, callback=[selfplay_callback, eval_callback])    
    except KeyboardInterrupt:
        print('stop by keyboard')

    model.save(os.path.join(LOGDIR, "final_model"))
    env.close()