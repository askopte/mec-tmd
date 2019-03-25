import os
import time
import threading
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle

import parameters
import environment
import job_distribution
import tf_network
import slow_down_cdf

tf_lock = threading.Lock()

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
    infos = []

    ob = env.observe()

    for i in range(episode_max_length):

        tf_lock.acquire()
        try:
            act = agent.choose_action(ob)
        finally:
            tf_lock.release()

        obs.append(ob)
        acts.append(act)

        ob,rew,done,info,info2 = env.step(act,repeat = True)

        rews.append(rew)
        infos.append(info)
        
        if done:break
    
    max = 0

    for info in infos:
        max = info[0]
    
    worked_info = np.zeros((max))

    for info in infos:
        worked_info[info[0]-1] = info[1]

    return {'reward': np.array(rews),
            'ob': np.array(obs),
            'action': np.array(acts),
            'info': worked_info,
            'info2':info2
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

def plot_lr_curve(output_file_prefix, max_rew_lr_curve, mean_rew_lr_curve, ref_discount_rews, rate_lr_curve, ref_idle_rate, qos_lr_curve, ref_qos, latency_lr_curve, ref_latency, resume, pa_change, ref_ambr):
    num_colors = 20
    cm = plt.get_cmap('gist_rainbow')

    fig = plt.figure(figsize=(18, 12))

    ax = fig.add_subplot(221)
    ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

    for k in ref_discount_rews:
        ax.plot(np.tile(np.average(ref_discount_rews[k]), len(mean_rew_lr_curve)), linewidth=2, label=k)

    ax.plot(mean_rew_lr_curve, linewidth=2, label='DRL mean')
    ax.plot(max_rew_lr_curve, linewidth=2, label='DRL max')

    for change in pa_change:
        ax.axvline(x = change[0],linewidth = 2, label = 'Arriving rate from '+ str(change[1]) + ' to ' + str(change[2]))

    plt.legend(loc=4)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Discounted Total Reward", fontsize=20)

    ax = fig.add_subplot(222)
    ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

    for k in ref_idle_rate:
        ax.plot(np.tile(np.average(ref_idle_rate[k]), len(mean_rew_lr_curve)), linewidth=2, label=k)

    ax.plot(rate_lr_curve, linewidth=2, label='DRL-TO')

    for change in pa_change:
        ax.axvline(x = change[0],linewidth = 2, label = 'Arriving rate from '+ str(change[1]) + ' to ' + str(change[2]))

    plt.legend(loc=4)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Average idle Rate", fontsize=20)

    ax = fig.add_subplot(223)
    ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

    for k in ref_qos:
        ax.plot(np.tile(np.average(ref_qos[k]), len(mean_rew_lr_curve)), linewidth=2, label=k)
    
    ax.plot(np.tile(ref_ambr, len(mean_rew_lr_curve)), linewidth=2, label='Average UE-AMBR')

    ax.plot(qos_lr_curve, linewidth=2, label='DRL-TO')

    for change in pa_change:
        ax.axvline(x = change[0],linewidth = 2, label = 'Arriving rate from '+ str(change[1]) + ' to ' + str(change[2]))

    plt.legend(loc=4)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Average QoS", fontsize=20)

    ax = fig.add_subplot(224)
    ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

    for k in ref_latency:
        ax.plot(np.tile(np.average(ref_latency[k]), len(mean_rew_lr_curve)), linewidth=2, label=k)

    ax.plot(latency_lr_curve, linewidth=2, label='DRL-TO')

    for change in pa_change:
        ax.axvline(x = change[0],linewidth = 2, label = 'Arriving rate from '+ str(change[1]) + ' to ' + str(change[2]))

    plt.legend(loc=4)
    plt.xlabel("Iteration", fontsize=20)
    plt.ylabel("Average Latency", fontsize=20)

    plt.savefig(output_file_prefix + '_' + str(resume) + "_lr_curve" + ".pdf")

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

    all_rate = np.array([np.average(traj["info"]) for traj in trajs])
    all_info2 = np.array([traj["info2"] for traj in trajs])

    all_qos = all_info2[:,0]
    all_latency = all_info2[:,1]

    return all_ob, all_action, all_adv, all_eprews, all_eplens, all_rate, all_qos, all_latency

def mt_worker(tf_learner, env, pa, all_loss, all_eprews, all_eplens, all_rate, all_qos, all_latency, ex):

    all_ob, all_action, all_adv, eprews, eplens , rate, qos, latency = get_traj_worker(tf_learner, env, pa)

    tf_lock.acquire()
    try:
        loss = tf_learner.learn(all_ob,all_action, all_adv)
    finally:
        tf_lock.release()
    
    all_loss.append(loss)
    all_eprews.append(eprews)
    all_eplens.append(eplens)
    all_rate.append(np.average(rate))
    all_qos.append(np.average(qos))
    all_latency.append(np.average(latency))

    print ("-----------------")
    print ("NumExp \t %i" % ex)
    print ("NumTrajs: \t %i" % len(eprews))
    print ("NumTimesteps: \t %i" % np.sum(eplens))
    print ("MaxRew: \t %s" % np.average([np.max(rew) for rew in eprews]))
    print ("MeanRew: \t %s +- %s" % (np.mean(eprews), np.std(eprews)))
    print ("MeanLen: \t %s +- %s" % (np.mean(eplens), np.std(eplens)))
    print ("IdleRate: \t %s +- %s" % (np.mean(rate), np.std(rate)))
    print ("AverageService: \t %s +- %s" % (np.mean(qos), np.std(qos)))
    print ("AverageLatency: \t %s +- %s" % (np.mean(latency), np.std(latency)))
    print ("-----------------")

def main():

    resume = None
    resume_itr = None

    pa = parameters.Parameters()

    # ----------------------------
    print("Preparing for workers...")
    # ----------------------------

    envs = []

    tf_learner = tf_network.TFLearner(pa, pa.network_input_height, pa.network_input_width, 33)

    if resume is not None:
        print("Find resume data, load from slot "+str(resume))

        file = open(pa.output_filename + "_" + str(resume) + ".pkl", 'rb')
        nw_len_seqs = pickle.load(file)
        nw_ambr_seqs = pickle.load(file)
        file.close()

        file = open(pa.output_filename + "_" + str(resume) + "_etc.pkl", 'rb')
        resume_itr = pickle.load(file)
        resume_itr += 1
        print("Start from iteration "+str(resume_itr))
        max_rew_lr_curve = pickle.load(file)
        mean_rew_lr_curve = pickle.load(file)
        rate_lr_curve = pickle.load(file)
        qos_lr_curve = pickle.load(file)
        latency_lr_curve = pickle.load(file)
        ref_new_job_rate = pickle.load(file)
        pa_change = pickle.load(file)
        file.close()

        if ref_new_job_rate != pa.new_job_rate:
            print("Identified arriving rate change from " + str(ref_new_job_rate) + " to " + str(pa.new_job_rate))
            print("Generating new job sequence")
            nw_len_seqs = job_distribution.generate_sequence_work(pa)
            nw_ambr_seqs = job_distribution.generate_sequence_ue_ambr(pa)
            file = open(pa.output_filename + "_" + str(resume) + ".pkl", 'wb')
            pickle.dump(nw_len_seqs, file)
            pickle.dump(nw_ambr_seqs, file)
            file.close()
            pa_change.append([resume_itr - 1, ref_new_job_rate, pa.new_job_rate])
            

        tf_learner.load_data(pa.output_filename + '_' + str(resume))

    else:
        resume = np.random.randint(100,999)
        print("Cannot find resume data, write in slot "+str(resume))

        file = open(pa.output_filename + "_" + str(resume) + ".pkl", 'wb')
        nw_len_seqs = job_distribution.generate_sequence_work(pa)
        nw_ambr_seqs = job_distribution.generate_sequence_ue_ambr(pa)
        pickle.dump(nw_len_seqs, file)
        pickle.dump(nw_ambr_seqs, file)
        file.close()

        resume_itr = 1
        max_rew_lr_curve = []
        mean_rew_lr_curve = []
        rate_lr_curve = []
        qos_lr_curve = []
        latency_lr_curve = []
        pa_change = []

        tf_learner.save_data(pa.output_filename + '_' + str(resume))

    for ex in range(pa.num_ex):

        print ("-prepare for env-", ex)
        env = environment.Env(pa, nw_len_seqs=nw_len_seqs, nw_ambr_seqs=nw_ambr_seqs, end = 'all_done')
        env.seq_no = ex
        envs.append(env)


    # --------------------------------------
    print("Preparing for reference data...")
    # --------------------------------------

    ref_discount_rews , ref_idle_rate , ref_qos, ref_latency = slow_down_cdf.launch(pa, pg_resume=None, render=False, end='all_done')
    ref_ambr = np.average(nw_ambr_seqs)

    # --------------------------------------
    print("Start training...")
    # --------------------------------------

    timer_start = time.time()

    ex_indices = range(pa.num_ex)

    ts = []

    all_eprews = []
    all_eplens = []
    all_rate = []
    all_qos = []
    all_latency = []
    all_loss = []

    for iteration in range(resume_itr, pa.num_epochs):

        # np.random.shuffle(ex_indices)

        for ex in range(pa.num_ex):
            
            ex_idx = ex_indices[ex]
            thread = threading.Thread(target = mt_worker, args= (tf_learner, envs[ex_idx], pa, all_loss, all_eprews, all_eplens, all_rate, all_qos, all_latency, ex_idx), name= 'Worker-'+str(ex_indices[ex]))
            ts.append(thread)
            ts[ex].start()
        
        for ex in range(pa.num_ex):

            ts[ex].join()
        
        ts = []
        timer_end = time.time()

        print ("-----------------")
        print ("Slot: "+str(resume))
        print ("CompletedIteration: \t %i" % iteration)
        print ("Elapsed time\t %s" % (timer_end - timer_start), "seconds")
        print ("-----------------")

        timer_start = time.time()

        max_rew_lr_curve.append(np.average([np.max(rew) for rew in all_eprews]))
        mean_rew_lr_curve.append(np.mean(all_eprews))
        rate_lr_curve.append(np.mean(all_rate))
        qos_lr_curve.append(np.mean(all_qos))
        latency_lr_curve.append(np.mean(all_latency))

        if iteration % pa.output_freq == 0:

            plot_lr_curve(pa.output_filename,max_rew_lr_curve, mean_rew_lr_curve, ref_discount_rews, rate_lr_curve, ref_idle_rate, qos_lr_curve, ref_qos, latency_lr_curve, ref_latency, resume, pa_change, ref_ambr)
            tf_learner.save_data(pa.output_filename + '_' + str(resume))
            file = open(pa.output_filename + "_" + str(resume) + "_etc.pkl", 'wb')
            pickle.dump(iteration, file)
            pickle.dump(max_rew_lr_curve, file)
            pickle.dump(mean_rew_lr_curve, file)
            pickle.dump(rate_lr_curve, file)
            pickle.dump(qos_lr_curve, file)
            pickle.dump(latency_lr_curve, file)
            pickle.dump(pa.new_job_rate, file)
            pickle.dump(pa_change, file)
            file.close()

        all_eprews = []
        all_eplens = []
        all_loss = []
        all_rate = []
        all_qos = []
        all_latency = []


if __name__ == '__main__':
    main()