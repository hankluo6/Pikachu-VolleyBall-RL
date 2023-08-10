import gymnasium as gym
import gym_pikachu_volleyball
from gym_pikachu_volleyball.envs import PikachuVolleyballRandomEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Optional
import numpy as np
import argparse
import time

class PikachuVolleyballSelfPlayEnv(PikachuVolleyballRandomEnv):
    def __init__(self, is_player1_computer: bool, is_player2_computer: bool, render_mode: str, limited_timestep: int):
        super().__init__(is_player1_computer, is_player2_computer, render_mode, limited_timestep)
        self.model = None
        self.obs = None
        self.current_model = None
        self.next_model = None

    def step(self, action):
        other_action, _state = self.model.predict(self.obs, deterministic=True)
        obs, reward, terminated, truncated, info = super().step(action, other_action)
        self.obs = info['other_obs']
        return obs, reward, terminated, truncated, info
    
    def reset(self, options=None, seed: Optional[int]=None, return_info: bool=False):      
        obs, info = super().reset(options, seed, return_info)
        self.obs = info['other_obs']
        return obs, info
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the trained PPO agent.')
    parser.add_argument('--first_model', metavar='PATH', help='The path that store PPO model.',
                            type=str, default="models/PPO_selfplay_for_randomEnv")
    parser.add_argument('--second_model', metavar='PATH', help='The path that store PPO model.',
                            type=str, default=None)
    parser.add_argument('--eval', help='Whether to evaluate the model or not', action='store_true')
    parser.add_argument('--n_eval', metavar='N', help='Number of episode of the evaluation', type=int, default=10000)

    args = parser.parse_args()

    first_model = PPO.load(args.first_model)
    if args.second_model:
        second_model = PPO.load(args.second_model)
        env = PikachuVolleyballSelfPlayEnv
    else:
        second_model = None
        env = PikachuVolleyballRandomEnv

    if args.eval:
        print('evaluate the model')
        eval_env = env(render_mode="rgb_array", is_player1_computer=False, is_player2_computer=True, limited_timestep=1000)
        eval_env.model = second_model
        eval_env = Monitor(eval_env)
        episode_rewards, episode_lengths = evaluate_policy(
                    first_model,
                    eval_env,
                    n_eval_episodes=args.n_eval,
                    render=False,
                    deterministic=True,
                    return_episode_rewards=True,
                    warn=True,
                )
        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        print(f"Eval num_timesteps={args.n_eval}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

    else:
        env = env(render_mode="human", is_player1_computer=False, is_player2_computer=True, limited_timestep=1000)
        env.model = second_model
        obs, _ = env.reset()
        try:
            while True:
                action, _state = first_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                env.render()
                time.sleep(0.05)
                if terminated:
                    obs, _ = env.reset()
        except KeyboardInterrupt:
            env.close()