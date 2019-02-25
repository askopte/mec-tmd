import os
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import parameters
import environment
import job_distribution
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

def init_accums(tf_learner):  # in rmsprop
    accums = []
    params = tf_learner.get_num_params()
    for param in params:
        accum = np.zeros(param.shape, dtype=param.dtype)
        accums.append(accum)
    return accums

def rmsprop_updates_outside(grads, params, accums, stepsize, rho=0.9, epsilon=1e-9):

    assert len(grads) == len(params)
    assert len(grads) == len(accums)
    for dim in range(len(grads)):
        accums[dim] = rho * accums[dim] + (1 - rho) * grads[dim] ** 2
        params[dim] += (stepsize * grads[dim] / np.sqrt(accums[dim] + epsilon))
    
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
    num_colors = 2
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

def get_traj_worker(tf_learner, env, pa):

    trajs = []

    for i in range(pa.num_seq_per_batch):
        traj = get_traj(tf_learner, env, pa.episode_max_length)
        trajs.append(traj)
    
    all_ob = concatenate_all_ob(trajs, pa)

    rets = [discount(traj["reward"], pa.discount) for traj in trajs]
    maxlen = max(len(ret) for ret in rets)
    padded_rets = [np.concatenate([ret, np.zeros(maxlen - len(ret))]) for ret in rets]

    baseline = np.mean(padded_rets, axis=0)
    
    advs = [ret - baseline[:len(ret)] for ret in rets]
    all_action = np.concatenate([traj["action"] for traj in trajs])
    all_adv = np.concatenate(advs)

    all_eprews = np.array([discount(traj["reward"], pa.discount)[0] for traj in trajs])
    all_eplens = np.array([len(traj["reward"]) for traj in trajs])

    return all_ob, all_action, all_adv, all_eprews, all_eplens

def main():

    pa = parameters.Parameters()

    env = environment.Env(pa, end = 'all_done')
    
    ref_discount_rews = slow_down_cdf.launch(pa, pg_resume=None, render=False, end='all_done')

    # ----------------------------
    print("Preparing for workers...")
    # ----------------------------

    envs = []

    nw_len_seqs = job_distribution.generate_sequence_work(pa)
    nw_ambr_seqs = job_distribution.generate_sequence_ue_ambr(pa)

    for ex in range(pa.num_ex):

        print ("-prepare for env-", ex)
        env = environment.Env(pa, nw_len_seqs=nw_len_seqs, nw_ambr_seqs=nw_ambr_seqs, end = 'all_done')
        env.seq_no = ex
        envs.append(env)
    
    tf_learner = tf_network.TFLearner(pa, pa.network_input_height, pa.network_input_width, 33)

    # --------------------------------------
    print("Preparing for reference data...")
    # --------------------------------------

    ref_discount_rews = slow_down_cdf.launch(pa, pg_resume=None, render=False, end='all_done')
    
    max_rew_lr_curve = []
    mean_rew_lr_curve = []

    # --------------------------------------
    print("Start training...")
    # --------------------------------------

    timer_start = time.time()

    for iteration in range(1, pa.num_epochs):

        ex_indices = range(pa.num_ex)
        np.random.shuffle(ex_indices)

        all_eprews = []
        all_eplens = []
        eprews = []
        eplens = []
        all_loss = []

        ex_counter = 0

        for ex in range(pa.num_ex):

            ex_idx = ex_indices[ex]

            all_ob, all_action, all_adv, eprews, eplens = get_traj_worker(tf_learner, env, pa)
            all_eprews.append(eprews)
            all_eplens.append(eplens)

            loss = tf_learner.learn(all_ob,all_action, all_adv)
            all_loss.append(loss)

            ex_counter += 1

            if ex_counter >= pa.batch_size or ex == pa.num_ex - 1:

                print (ex,"out of", pa.num_ex)
        
        timer_end = time.time()

        print ("-----------------")
        print ("Iteration: \t %i" % iteration)
        print ("NumTrajs: \t %i" % len(eprews))
        print ("NumTimesteps: \t %i" % np.sum(eplens))
        print ("MaxRew: \t %s" % np.average([np.max(rew) for rew in all_eprews]))
        print ("MeanRew: \t %s +- %s" % (np.mean(eprews), np.std(eprews)))
        print ("MeanLen: \t %s +- %s" % (np.mean(eplens), np.std(eplens)))
        print ("Elapsed time\t %s" % (timer_end - timer_start), "seconds")
        print ("-----------------")

        timer_start = time.time()

        max_rew_lr_curve.append(np.average([np.max(rew) for rew in all_eprews]))
        mean_rew_lr_curve.append(np.mean(eprews))

        if iteration % pa.output_freq == 0:

            plot_lr_curve(pa.output_filename,max_rew_lr_curve, mean_rew_lr_curve, ref_discount_rews)

if __name__ == '__main__':
    main()