import numpy as np
import job_distribution
import math

class Parameters:
    
    def __init__(self):
        # self.output_filename = 'data/tmp'
        self.output_filename = 'D:\Code\data\mec-tmd\data_temp'
        self.num_epochs = 1001         # number of training epochs
        self.simu_len = 250             # length of the busy cycle that repeats itself
        self.num_ex = 4                # number of sequences

        self.output_freq = 10          # interval for output and store parameters

        self.num_seq_per_batch = 10    # number of sequences to compute baseline
        self.episode_max_length = 2000  # enforcing an artificial terminal

        self.num_res = 2               # number of resources in the system
        self.num_nw = 200                # maximum allowed number of work in the queue

        self.time_horizon = 200         # number of time steps in the graph
        self.max_job_len = 100          # maximum duration of new jobs
        self.res_slot = 8             # maximum number of available resource slots

        self.ambr_len = 10             # LTE ambr prediction size

        self.max_track_since_new = 10  # track how many time steps since last new jobs

        self.job_num_cap = 200          # maximum number of distinct colors in current work graph

        self.new_job_rate = 2.2        # lambda in new job arrival Poisson Process

        self.discount = 1           # discount factor

         # distribution for new job arrival
        self.dist = job_distribution.Dist(self.max_job_len)

        self.qos_res_list = [1, 3, 5, 8]
        self.qos_rew_list = [-20, -5, -2, -1]
        self.qos_rew_delta = [-2, -2, -3, -3]
        self.mec_overall_latency = 5
        self.lte_latency = 3

        # graphical representation
        assert self.num_nw % self.time_horizon == 0  # such that it can be converted into an image
        assert self.job_num_cap % self.time_horizon == 0
        # assert self.ambr_len < self.time_horizon

        self.nw_width = int(math.ceil(self.num_nw / float(self.time_horizon)))
        self.job_width = int(math.ceil(self.job_num_cap / float(self.time_horizon)))
        self.network_input_height = self.time_horizon
        self.network_input_width = \
            self.res_slot * self.num_res + \
            self.nw_width * 2 + self.job_width * 2 + \
            1  # for extra info, 1) time since last new job 2) LTE network infomation

        self.delay_penalty = -10       # penalty for delaying things in the current work screen
        self.hold_penalty = -3        # penalty for holding things in the new work screen
        self.dismiss_penalty = -40     # penalty for missing a job because the queue is full

        self.num_frames = 1           # number of frames to combine and process
        self.lr_rate = 0.001          # learning rate
        self.rms_rho = 0.9            # for rms prop
        self.rms_eps = 1e-9           # for rms prop

        self.batch_size = 10
        
    def compute_dependent_parameters(self):
        assert self.num_nw % self.time_horizon == 0  # such that it can be converted into an image
        assert self.job_num_cap % self.time_horizon == 0
        self.nw_width = int(math.ceil(self.num_nw / float(self.time_horizon)))
        self.job_width = int(math.ceil(self.job_num_cap / float(self.time_horizon)))
        self.network_input_height = self.time_horizon
        self.network_input_width = \
            self.res_slot * self.num_res + \
            self.nw_width + self.job_width * 3 + \
            1  # for extra info, 1) time since last new job 2) LTE network infomation