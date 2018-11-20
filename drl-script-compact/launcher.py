import parameters
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt

import environment

def get_traj(env, episode_max_length):

    env.reset()
    obs = []
    acts = []
    rews = []
    info = []

    ob = env.observe()

    for _ in xrange(episode_max_length):
        act = 
        obs.append(ob)
        acts.append(act)

        ob,rew,done,info = env.step(act,repeat = True)

        rews.append(rew)
        
        if done:break
    
    return {'reward': np.array(rews),
            'ob': np.array(obs),
            'action': np.array(acts),
            'info': info
            }
        
def concatenate_all_ob(trajs, pa):

    timesteps_total = 0
    for i in xrange(len(trajs)):
        timesteps_total += len(trajs[i]['reward'])
    
    all_ob = np.zeros(
        (timesteps_total, 1, pa.network_input_height, pa.network_input_width))

    timesteps = 0
    ffor i in xrange(len(trajs)):
        for j in xrange(len(trajs[i]['reward'])):
            all_ob[timesteps, 0, :, :] = trajs[i]['ob'][j]
            timesteps += 1
    
    return all_ob

def main():

    pa = parameters.Parameters()

    env = environment.Env(pa, end = end)

    timer_start = time.time()

    for iteration in xrange(pa.num_epochs):

        all_oball_ob = []
        all_action = []
        all_adv = []
        all_eprews = []
        all_eplens = []

        for ex in xrange(pa.num_ex):

            trajs = []

            for i in xrange(pa.num_seq_per_batch):
                trajs = get_traj(env, pa.episode_max_length)
                trajs.append(traj)
            
            env.seq_no = (env.seq_no + 1) % env.pa.num_ex

            all_ob.append(concatenate_all_ob(trajs, pa))

            rets = []


if __name__ == '__main__':
    main()