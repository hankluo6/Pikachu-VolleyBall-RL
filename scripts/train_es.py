"""
This code is edited from mofan's ES tutorial
The original code can be found in https://github.com/MorvanZhou/Evolutionary-Algorithm/
"""
import numpy as np
import gymnasium as gym
import multiprocessing as mp
import time
import gym_pikachu_volleyball

N_KID = 10                  # half of the training population
N_GENERATION = 5000         # training step
LR = .005                    # learning rate
SIGMA = .05                 # mutation strength or step size
N_CORE = mp.cpu_count()-1
SEED = 1119

np.random.seed(SEED)

def sign(k_id): return -1. if k_id % 2 != 0 else 1.  # mirrored sampling


class SGD(object):                      # optimizer with momentum
    def __init__(self, params, learning_rate, momentum=0.9):
        self.v = np.zeros_like(params).astype(np.float32)
        self.lr, self.momentum = learning_rate, momentum

    def get_gradients(self, gradients):
        self.v = self.momentum * self.v + (1. - self.momentum) * gradients
        return self.lr * self.v


def params_reshape(shapes, params):     # reshape to be a matrix
    p, start = [], 0
    for i, shape in enumerate(shapes):  # flat params to matrix
        n_w, n_b = shape[0] * shape[1], shape[1]
        p = p + [params[start: start + n_w].reshape(shape),
                 params[start + n_w: start + n_w + n_b].reshape((1, shape[1]))]
        start += n_w + n_b
    return p

def get_reward(shapes, params, seed_and_id=None,):
    env = gym.make("PikachuVolleyballRandom-v0", render_mode="rgb_array", is_player1_computer=False, is_player2_computer=True, limited_timestep=1000).unwrapped
    # perturb parameters using seed
    if seed_and_id is not None:
        seed, k_id = seed_and_id
        np.random.seed(seed)
        params += sign(k_id) * SIGMA * np.random.randn(params.size)
    p = params_reshape(shapes, params)
    # run episode
    ep_r = 0.
    ep_max = 100
    for _ in range(ep_max):
        s, _ = env.reset()
        while True:
            a = get_action(p, s)
            s, r, done, _, _= env.step(a)
            ep_r += r
            if done: 
                break
    env.close()
    return ep_r / ep_max

def get_action(params, x):
    x = x[np.newaxis, :]
    x = np.tanh(x.dot(params[0]) + params[1])
    x = np.tanh(x.dot(params[2]) + params[3])
    x = x.dot(params[4]) + params[5]
    return np.argmax(x, axis=1)[0]      # for discrete action

class Trainer:
    def __init__(self, generation, n_kid, lr, sigma):
        self.generation = generation
        self.n_kid = n_kid
        self.lr = lr
        self.sigma = sigma
        self.eval_freq = 1
        self.net_shapes, self.net_params = self.__build_net()
        self.utility = self.__cal_util()
        self.optimizer = SGD(self.net_params, self.lr)

    def __build_net(self):
        def linear(n_in, n_out):  # network linear layer
            w = np.random.randn(n_in * n_out).astype(np.float32) * .1
            b = np.random.randn(n_out).astype(np.float32) * .1
            return (n_in, n_out), np.concatenate((w, b))
        s0, p0 = linear(10, 30)
        s1, p1 = linear(30, 20)
        s2, p2 = linear(20, 18)
        return [s0, s1, s2], np.concatenate((p0, p1, p2))
    
    def __cal_util(self):
        # utility instead reward for update parameters (rank transformation)
        base = self.n_kid * 2    # *2 for mirrored sampling
        rank = np.arange(1, base + 1)
        util_ = np.maximum(0, np.log(base / 2 + 1) - np.log(rank))
        utility = util_ / util_.sum() - 1 / base
        return utility

    def train(self):
        mar = None      # moving average reward
        best_r = -2
        pool = mp.Pool(processes=N_CORE)
    
        for g in range(self.generation):
            t0 = time.time()
            # pass seed instead whole noise matrix to parallel will save your time
            noise_seed = np.random.randint(0, 2 ** 32 - 1, size=self.n_kid, dtype=np.uint32).repeat(2)    # mirrored sampling

            # distribute training in parallel
            jobs = [pool.apply_async(get_reward, (self.net_shapes, self.net_params,
                                                [noise_seed[k_id], k_id], )) for k_id in range(self.n_kid*2)]
            rewards = np.array([j.get() for j in jobs])
            
            kids_rank = np.argsort(rewards)[::-1]               # rank kid id by reward

            cumulative_update = np.zeros_like(self.net_params)       # initialize update values
            for ui, k_id in enumerate(kids_rank):
                np.random.seed(noise_seed[k_id])                # reconstruct noise using seed
                cumulative_update += self.utility[ui] * sign(k_id) * np.random.randn(self.net_params.size)

            gradients = self.optimizer.get_gradients(cumulative_update/(2*self.n_kid*self.sigma))
            self.net_params += gradients

            if g % self.eval_freq == 0:
                net_r = get_reward(self.net_shapes, self.net_params, None,)
                mar = net_r if mar is None else 0.9 * mar + 0.1 * net_r       # moving average reward
                print(
                    'Gen: ', g,
                    '| Net_R: %.1f' % net_r, 
                    '| Avg_R: %.1f' % mar,
                    '| Kid_avg_R: %.1f' % rewards.mean(),
                    '| Gen_T: %.2f' % (time.time() - t0),)
                
                with open('record.txt', 'a+') as f:
                    f.write(f'Gen: {g} | Net_R: {net_r} | Avg_R: {mar} | Kid_avg_R: {rewards.mean()} | Gen_T: {(time.time() - t0)}\n')
                            
                if best_r <= net_r:
                    best_r = net_r
                    np.save(f'params_{g}_{net_r}', self.net_params)

if __name__ == "__main__":
    trainer = Trainer(N_GENERATION, n_kid=N_KID, lr=LR, sigma=SIGMA)
    trainer.train()