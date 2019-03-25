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
    infos = []

    ob = env.observe()

    for _ in range(episode_max_length):

        if test_type == 'DRL':
            a = pg_learner.choose_action(ob)

        elif test_type == 'Max Access':
            a = etc.get_access_action(env.machine, env.job_slot)

        elif test_type == 'Quality':
            a = etc.get_quality_action(env.machine, env.job_slot)

        elif test_type == 'Greedy':
            a = etc.get_greedy_action(pa, env.machine, env.job_slot)

        elif test_type == 'Random':
            a = etc.get_random_action(env.job_slot)

        ob, rew, done, info , info2= env.step(a, repeat=True)

        infos.append(info)
        rews.append(rew)

        if done: break
        if render: env.render()
        # env.render()
    
    max = 0

    for info in infos:
        max = info[0]
    
    worked_info = np.zeros((max))

    for info in infos:
        worked_info[info[0]-1] = info[1]

    return np.array(rews), worked_info, info2

def launch(pa, pg_resume = None, render = False, end = "no_new_job"):

    test_types = ['Max Access', 'Greedy', 'Random']

    if pg_resume is not None:
        test_types = ['DRL-TO'] + test_types
    
    env = environment.Env(pa,end = end)

    all_discount_rews = {}
    all_idle_rate = {}
    all_qos = {}
    all_latency = {}

    for test_type in test_types:
        all_discount_rews[test_type] = []
        all_idle_rate[test_type] = []
        all_qos[test_type] = []
        all_latency[test_type] = []
    
    for seq_idx in range(pa.num_ex):
        print('\n\n')
        print("=============== " + str(seq_idx) + " ===============")

        for test_type in test_types:

            rews, info, info2= get_traj(test_type, pa, env, pa.episode_max_length, pg_resume)

            avg_qos = info2[0]
            avg_latency = info2[1]

            rate = np.average(info)

            print ("---------- " + test_type + " -----------")

            print ("total discount reward : \t %s" % (discount(rews, pa.discount)[0]))
            print ("average idle rate :     \t %s" % (rate))
            print ("average service level : \t %s" % (avg_qos))
            print ("average latency : \t %s" % (avg_latency))

            all_discount_rews[test_type].append(
                discount(rews, pa.discount)[0]
            )
            
            all_idle_rate[test_type].append(rate)
            all_qos[test_type].append(avg_qos)
            all_latency[test_type].append(avg_latency)
            

        env.seq_no = (env.seq_no + 1) % env.pa.num_ex
    
    return all_discount_rews, all_idle_rate, all_qos, all_latency

