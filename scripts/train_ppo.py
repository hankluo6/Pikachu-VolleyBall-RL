import gymnasium as gym
import gym_pikachu_volleyball
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
import os

SEED = 1119
N_STEPS = 4096
ENT_COEF = 0.001
VERBOSE = 1
NUM_TIMESTEPS = int(2e7)
EVAL_FREQ = 10000
EVAL_EPISODES = 100
TENSORBOARD_LOG = "./tensorboard/pikachu-volleyball/"
LOGDIR = "models"
set_random_seed(SEED)

if __name__ == "__main__":
    env = gym.make("PikachuVolleyballRandom-v0", render_mode="rgb_array", is_player1_computer=False, is_player2_computer=True, limited_timestep=1000)
    env = Monitor(env)

    model = PPO("MlpPolicy", env, seed=SEED, n_steps=N_STEPS, ent_coef=ENT_COEF, verbose=VERBOSE, tensorboard_log=TENSORBOARD_LOG)
    eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

    try:
        model.learn(total_timesteps=NUM_TIMESTEPS, callback=eval_callback)
    except KeyboardInterrupt:
        print('stop by keyboard')

    model.save(os.path.join(LOGDIR, "final_model"))
    env.close()
