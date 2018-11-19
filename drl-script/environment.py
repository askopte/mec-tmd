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
            
            self.nw_len_seqs = np.reshape(self.nw_len_seqs,
                                           [self.pa.num_ex, self.pa.simu_len])
        
        else:
            self.nw_len_seqs = nw_len_seqs

        self.seq_no = 0  # which example sequence
        self.seq_idx = 0  # index in that sequence

        # initialize system
        self.machine = Machine(pa)
        self.job_slot = JobSlot(pa)
        self.job_backlog = JobBacklog(pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(pa)

        def generate_sequence_work(self, simu_len):

        nw_len_seq = np.zeros(simu_len * 2, dtype=int)

        for i in range(simu_len):

            nw_len_seq[2 * i] = self.nw_dist()

            if np.random.rand() < self.pa.new_job_rate - 1:  # a new job comes

                nw_len_seq[2 * i + 1] = self.nw_dist()

        return nw_len_seq

        def get_new_job_from_seq(self, seq_no, seq_idx):
        new_job = Job(job_len=self.nw_len_seqs[seq_no, seq_idx],
                      job_id=len(self.job_record.record),
                      enter_time=self.curr_time)
        return new_job

        def observe(self):

            image_repr = np.zeros((self.pa.network_input_height, self.pa.network_input_width))

            ir_pt = 0
            
