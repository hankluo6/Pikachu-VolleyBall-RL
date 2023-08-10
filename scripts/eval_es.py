"""
This code is edited from mofan's ES tutorial
The original code can be found in https://github.com/MorvanZhou/Evolutionary-Algorithm/
"""
import numpy as np
import gymnasium as gym
import argparse
import gym_pikachu_volleyball

def params_reshape(shapes, params):     # reshape to be a matrix
    p, start = [], 0
    for i, shape in enumerate(shapes):  # flat params to matrix
        n_w, n_b = shape[0] * shape[1], shape[1]
        p = p + [params[start: start + n_w].reshape(shape),
                 params[start + n_w: start + n_w + n_b].reshape((1, shape[1]))]
        start += n_w + n_b
    return p

def build_net():
    def linear(n_in, n_out):  # network linear layer
        w = np.random.randn(n_in * n_out).astype(np.float32) * .1
        b = np.random.randn(n_out).astype(np.float32) * .1
        return (n_in, n_out), np.concatenate((w, b))
    s0, p0 = linear(10, 30)
    s1, p1 = linear(30, 20)
    s2, p2 = linear(20, 18)
    return [s0, s1, s2], np.concatenate((p0, p1, p2))

def get_action(params, x):
    x = x[np.newaxis, :]
    x = np.tanh(x.dot(params[0]) + params[1])
    x = np.tanh(x.dot(params[2]) + params[3])
    x = x.dot(params[4]) + params[5]
    return np.argmax(x, axis=1)[0]      # for discrete action

def rollout(env, net_shapes, player1, player2, n_eval, second_player):
    p1 = params_reshape(net_shapes, player1)
    if second_player:
        p2 = params_reshape(net_shapes, player2)
    total_r = []
    total_l = []
    while True:
        if n_eval != None and len(total_r) == n_eval:
            break
        s, info = env.reset()
        ep_r = 0
        ep_l = 0
        while True:
            env.render()
            a = get_action(p1, s)
            if second_player:
                a2 = get_action(p2, info['other_obs'])
                s, r, done, _, _ = env.step(a, a2)
            else:
                s, r, done, _, _ = env.step(a)
            ep_r += r
            ep_l += 1
            if done: 
                break
        total_r.append(ep_r)
        total_l.append(ep_l)
    return total_r, total_l

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate the trained PPO agent.')
    parser.add_argument('--first_model', metavar='PATH', help='The path that store PPO model.',
                            type=str, default="models/ES_selfplay_for_randomEnv.npy")
    parser.add_argument('--second_model', metavar='PATH', help='The path that store PPO model.',
                            type=str, default=None)
    parser.add_argument('--eval', help='Whether to evaluate the model or not', action='store_true')
    parser.add_argument('--n_eval', metavar='N', help='Number of episode of the evaluation', type=int, default=10000)

    args = parser.parse_args()

    net_shapes, net_params_first = build_net()
    net_params_first = np.load(args.first_model)
    if args.second_model:
        net_params_second = np.load(args.second_model)
        second_player = True
    else:
        net_params_second = None
        second_player = False

    if args.eval:
        print('evaluate the model')
        env = gym.make("PikachuVolleyballRandom-v0", render_mode="rgb_array", is_player1_computer=False, is_player2_computer=True, limited_timestep=1000).unwrapped
        episode_rewards, episode_lengths = rollout(env=env, net_shapes=net_shapes, player1=net_params_first, player2=net_params_second, n_eval=args.n_eval, second_player=second_player)
        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        print(f"Eval num_timesteps={args.n_eval}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

    else:
        env = gym.make("PikachuVolleyballRandom-v0", render_mode="human", is_player1_computer=False, is_player2_computer=True, limited_timestep=1000).unwrapped
        episode_rewards, episode_lengths = rollout(env=env, net_shapes=net_shapes, player1=net_params_first, player2=net_params_second, n_eval=None, second_player=second_player)
