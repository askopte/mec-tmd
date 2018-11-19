import numpy as np
import math
import matplotlib as plt
import tensorflow

import parameters

class Env:
    def __init__(self, pa, nw_len_seqs = None,  
                 render = False, end = 'no_new_job'):
        
        self.pa = pa
        self.render = render
        self.end = end  # termination type, 'no_new_job' or 'all_done'

        self.nw_dist = pa.dist.job_dist

        self.curr_time = 0

        np.random.seed(seed)

        if nw_len_seqs is None :
            # generate new work
            self.nw_len_seqs = \
                self.generate_sequence_work(self.pa.simu_len * self.pa.num_ex)
