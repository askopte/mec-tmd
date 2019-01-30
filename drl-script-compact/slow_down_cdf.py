import numpy as np
import matplotlib.pyplot as plt

import environment
import parameters
import tf_network
import etc

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

def get_traj(test_type, pa, env, episode_max_length, pg_resume=None, render=False):
    """
    Run agent-environment loop for one whole episode (trajectory)
    Return dictionary of results
    """

    if test_type == 'DRL':  # load trained parameters

        pg_learner = pg_resume

    env.reset()
    rews = []

    ob = env.observe()

    for _ in range(episode_max_length):

        if test_type == 'DRL':
            a = pg_learner.choose_action(ob)

        elif test_type == 'Access':
            a = etc.get_access_action(env.machine, env.job_slot)

        elif test_type == 'Quality':
            a = etc.get_quality_action(env.machine, env.job_slot)

        elif test_type == 'Greedy':
            a = etc.get_greedy_action(pa, env.machine, env.job_slot)

        elif test_type == 'Random':
            a = etc.get_random_action(env.job_slot)

        ob, rew, done, info = env.step(a, repeat=True)

        rews.append(rew)

        if done: break
        if render: env.render()
        # env.render()

    return np.array(rews), info

def launch(pa, pg_resume = None, render = False, plot = False, end = "no_new_job"):

    test_types = ['Access', 'Quality', 'Greedy', 'Random']

    if pg_resume is not None:
        test_types = ['DRL-TO'] + test_types
    
    env = environment.Env(pa,end = end)

    all_discount_rews = {}
    jobs_slow_down = {}
    work_complete = {}
    work_remain = {}
    job_len_remain = {}
    num_job_remain = {}
    job_remain_delay = {}

    for test_type in test_types:
        all_discount_rews[test_type] = []
        jobs_slow_down[test_type] = []
        work_complete[test_type] = []
        work_remain[test_type] = []
        job_len_remain[test_type] = []
        num_job_remain[test_type] = []
        job_remain_delay[test_type] = []
    
    for seq_idx in range(pa.num_ex):
        print('\n\n')
        print("=============== " + str(seq_idx) + " ===============")

        for test_type in test_types:

            rews, info = get_traj(test_type, pa, env, pa.episode_max_length, pg_resume)

            print ("---------- " + test_type + " -----------")

            print ("total discount reward : \t %s" % (discount(rews, pa.discount)[0]))

            all_discount_rews[test_type].append(
                discount(rews, pa.discount)[0]
            )

            enter_time = np.array([info.record[i].enter_time for i in range(len(info.record))])
            finish_time = np.array([info.record[i].finish_time for i in range(len(info.record))])
            job_len = np.array([info.record[i].len for i in range(len(info.record))])
            job_total_size = np.array([np.sum(info.record[i].res_vec) for i in range(len(info.record))])

            finished_idx = (finish_time >= 0)
            unfinished_idx = (finish_time < 0)

            jobs_slow_down[test_type].append(
                (finish_time[finished_idx] - enter_time[finished_idx]) / job_len[finished_idx]
            )
            work_complete[test_type].append(
                np.sum(job_len[finished_idx] * job_total_size[finished_idx])
            )
            work_remain[test_type].append(
                np.sum(job_len[unfinished_idx] * job_total_size[unfinished_idx])
            )
            job_len_remain[test_type].append(
                np.sum(job_len[unfinished_idx])
            )
            num_job_remain[test_type].append(
                len(job_len[unfinished_idx])
            )
            job_remain_delay[test_type].append(
                np.sum(pa.episode_max_length - enter_time[unfinished_idx])
            )

        env.seq_no = (env.seq_no + 1) % env.pa.num_ex

    if plot:
        num_colors = len(test_types)
        cm = plt.get_cmap('gist_rainbow')
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_color_cycle([cm(1. * i / num_colors) for i in range(num_colors)])

        for test_type in test_types:
            slow_down_cdf = np.sort(np.concatenate(jobs_slow_down[test_type]))
            slow_down_yvals = np.arange(len(slow_down_cdf))/float(len(slow_down_cdf))
            ax.plot(slow_down_cdf, slow_down_yvals, linewidth=2, label=test_type)

        plt.legend(loc=4)
        plt.xlabel("job slowdown", fontsize=20)
        plt.ylabel("CDF", fontsize=20)
        # plt.show()
        plt.savefig("comparsion_slowdown_fig" + ".pdf")
    
    return all_discount_rews, jobs_slow_down

