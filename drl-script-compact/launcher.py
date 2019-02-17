import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import parameters
import environment
import tf_network
import slow_down_cdf

def discount(x, gamma):
    """
    Given vector x, computes a vector y such that
    y[i] = x[i] + gamma * x[i+1] + gamma^2 x[i+2] + ...
    """
    out = np.zeros(len(x))
    out[-1] = x[-1]
    for i in reversed(range(len(x)-1)):
        out[i] = x[i] + gamma*out[i+1]
    assert x.ndim >= 1
    # More efficient version:
    # scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]
    return out

def get_traj(agent, env, episode_max_length):

    env.reset()
    obs = []
    acts = []
    rews = []
    info = []

    ob = env.observe()

    for _ in range(episode_max_length):

        act = agent.choose_action(ob)

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
    for i in range(len(trajs)):
        timesteps_total += len(trajs[i]['reward'])
    
    all_ob = np.zeros(
        (timesteps_total, pa.network_input_height*pa.network_input_width))
        #(timesteps_total, 1, pa.network_input_height*pa.network_input_width))

    timesteps = 0
    
    for i in range(len(trajs)):
        for j in range(len(trajs[i]['reward'])):
            all_ob[timesteps, :] = trajs[i]['ob'][j]
            timesteps += 1
    
    return all_ob

def concatenate_all_ob_across_examples(all_ob, pa):
    num_ex = len(all_ob)
    total_samp = 0
    for i in range(num_ex):
        total_samp += all_ob[i].shape[0]

    all_ob_contact = np.zeros(
        (total_samp, pa.network_input_height*pa.network_input_width))

    total_samp = 0

    for i in range(num_ex):
        prev_samp = total_samp
        total_samp += all_ob[i].shape[0]
        all_ob_contact[prev_samp : total_samp, :] = all_ob[i]

    return all_ob_contact

def plot_lr_curve(output_file_prefix, max_rew_lr_curve, mean_rew_lr_curve, ref_discount_rews):
    num_colors = 10
    cm = plt.get_cmap('gist_rainbow')

    fig = plt.figure(figsize=(12, 5))

    ax = fig.add_subplot(111)
    ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

    for k in ref_discount_rews:
        ax.plot(np.tile(np.average(ref_discount_rews[k]), len(mean_rew_lr_curve)), linewidth=2, label=k)

    ax.plot(mean_rew_lr_curve, linewidth=2, label='PG mean')
    ax.plot(max_rew_lr_curve, linewidth=2, label='PG max')

    plt.legend(loc=4)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Discounted Total Reward", fontsize=20)

    plt.savefig(output_file_prefix + "_lr_curve" + ".pdf")

def main():

    pa = parameters.Parameters()

    env = environment.Env(pa, end = 'all_done')

    tf_learner = tf_network.TFLearner(pa, pa.network_input_height, pa.network_input_width, 33)

    ref_discount_rews = slow_down_cdf.launch(pa, pg_resume=None, render=False, end='all_done')

    timer_start = time.time()

    max_rew_lr_curve = []
    mean_rew_lr_curve = []

    for iteration in range(pa.num_epochs):

        all_ob = []
        all_action = []
        all_adv = []
        all_eprews = []
        all_eplens = []

        for ex in range(pa.num_ex):

            trajs = []

            for i in range(pa.num_seq_per_batch):
                traj = get_traj(tf_learner, env, pa.episode_max_length)
                trajs.append(traj)
            
            env.seq_no = (env.seq_no + 1) % env.pa.num_ex

            all_ob.append(concatenate_all_ob(trajs, pa))

            rets = [discount(traj["reward"], pa.discount) for traj in trajs]
            maxlen = max(len(ret) for ret in rets)
            padded_rets = [np.concatenate([ret, np.zeros(maxlen - len(ret))]) for ret in rets]

            baseline = np.mean(padded_rets, axis=0)

            advs = [ret - baseline[:len(ret)] for ret in rets]
            all_action.append(np.concatenate([traj["action"] for traj in trajs]))
            all_adv.append(np.concatenate(advs))

            all_eprews.append(np.array([discount(traj["reward"], pa.discount)[0] for traj in trajs]))  # episode total rewards
            all_eplens.append(np.array([len(traj["reward"]) for traj in trajs]))  # episode lengths
        
        all_ob = concatenate_all_ob_across_examples(all_ob, pa)
        all_action = np.concatenate(all_action)
        all_adv = np.concatenate(all_adv)

        # Do policy gradient update step
        loss = tf_learner.learn(all_ob,all_action,all_adv)
        eprews = np.concatenate(all_eprews)  # episode total rewards
        eplens = np.concatenate(all_eplens)  # episode lengths

        timer_end = time.time()

        print ("-----------------")
        print ("Iteration: \t %i" % iteration)
        print ("NumTrajs: \t %i" % len(eprews))
        print ("NumTimesteps: \t %i" % np.sum(eplens))
        print ("Loss:     \t %s" % loss)
        print ("MaxRew: \t %s" % np.average([np.max(rew) for rew in all_eprews]))
        print ("MeanRew: \t %s +- %s" % (eprews.mean(), eprews.std()))
        print ("MeanLen: \t %s +- %s" % (eplens.mean(), eplens.std()))
        print ("Elapsed time\t %s" % (timer_end - timer_start), "seconds")
        print ("-----------------")

        timer_start = time.time()

        max_rew_lr_curve.append(np.average([np.max(rew) for rew in all_eprews]))
        mean_rew_lr_curve.append(eprews.mean())

        if iteration % pa.output_freq == 0:
            param_file = open(pa.output_filename + '_' + str(iteration) + '.pkl', 'wb')
            
            param_file.close()

            plot_lr_curve(pa.output_filename,max_rew_lr_curve, mean_rew_lr_curve, ref_discount_rews)


if __name__ == '__main__':
    main()