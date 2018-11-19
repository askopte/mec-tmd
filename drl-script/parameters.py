import numpy as np
import math

class Parameters:
    def __init__(self):
        self.output_filename = 'data/tmp'

        self.num_epochs = 10000         # number of training epochs
        self.simu_len = 10             # length of the busy cycle that repeats itself
        self.num_ex = 1                # number of sequences

        self.output_freq = 10          # interval for output and store parameters

        self.num_seq_per_batch = 10    # number of sequences to compute baseline
        self.episode_max_length = 200  # enforcing an artificial terminal

        self.num_nw = 200                # maximum allowed number of work in the queue

        self.time_horizon = 200         # number of time steps in the graph
        self.max_job_len = 150          # maximum duration of new jobs
        self.res_slot = 10             # maximum number of available resource slots

        self.max_track_since_new = 10  # track how many time steps since last new jobs

        self.job_num_cap = 4000          # maximum number of distinct colors in current work graph

        self.new_job_rate = 0.7        # lambda in new job arrival Poisson Process

        self.discount = 1           # discount factor

         # distribution for new job arrival
        self.dist = job_distribution.Dist(self.max_job_len)

        # graphical representation
        assert self.num_nw % self.time_horizon == 0  # such that it can be converted into an image
        assert self.job_num_cap % self.time_horizon == 0
        self.nw_width = int(math.ceil(self.num_nw / float(self.time_horizon)))
        self.job_width = int(math.ceil(self.job_num_cap / float(self.time_horizon)))
        self.network_input_height = self.time_horizon
        self.network_input_width = \
            self.res_slot * 2 + \  
            self.nw_width + self.job_width\
            1  # for extra info, 1) time since last new job 2) LTE network infomation
        
    def compute_dependent_parameters(self):
        assert self.num_nw % self.time_horizon == 0  # such that it can be converted into an image
        assert self.job_num_cap % self.time_horizon == 0
        self.nw_width = int(math.ceil(self.num_nw / float(self.time_horizon)))
        self.job_width = int(math.ceil(self.job_num_cap / float(self.time_horizon)))
        self.network_input_height = self.time_horizon
        self.network_input_width = \
            self.res_slot * 2 + \   
            self.nw_width + self.job_width\
            1  # for extra info, 1) time since last new job 2) LTE network infomation