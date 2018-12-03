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
                self.generate_sequence_work(self.pa.num_ex, self.pa.simu_len, self.pa.new_job_rate)
        
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

    def generate_sequence_work(self, num_ex, simu_len, job_rate):

        nw_len_seq = np.zeros([num_ex, simu_len * np.ceil(job_rate)], dtype=int)
        for i in range(num_ex):
            for j in range(simu_len):
                job_no = 0
                for k in range(np.floor(job_rate)):
                    nw_len_seq[i, np.ceil(job_rate) * seq_idx + job_no] = self.nw_dist()
                    job_no += 1
                
                if np.random.rand() < self.pa.new_job_rate - np.floor(job_rate):  # a new job comes
                    nw_len_seq[i, np.ceil(job_rate) * seq_idx + job_no] = self.nw_dist()
                    job_no += 1

    return nw_len_seq

    def get_new_job_from_seq(self, seq_no, seq_idx, job_no):
        new_job = Job (job_len=self.nw_len_seqs[seq_no, seq_idx * np.ceil(self.pa.new_job_rate) + job_no],
                       job_id=len(self.job_record.record),
                       enter_time=self.curr_time)
    return new_job

    def observe(self):

        image_repr = np.zeros((self.pa.network_input_height, self.pa.network_input_width))

        ir_pt = 0

        for i in xrange(self.pa.num_res): #res_slot repre
                
            image_repr[:, ir_pt: ir_pt + self.pa.res_slot] = self.machine.canvas[i, :, :]
            ir_pt += self.pa.res_slot

        for i in xrange(self.pa.nw_width): # nw_width

            for j in xrange(self.pa.time_horizon):
                image_repr[i * self.pa.time_horizon + j, ir_pt : ir_pt + 1] = 
                
            ir_pt += 1
                
        for i in xrange(self.pa.job_width): # job_width

            for j in xrange(self.pa.time_horizon):
                image_repr[i * self.pa.time_horizon + j, ir_pt : ir_pt + 3] = 
                
            ir_pt += 3
            
        for i in xrange(self.pa.ambr_len):
            image_repr[i, ir_pt:ir_pt + 1] = 

        assert ir_pt == image_repr.shape[1]

        return image_repr
    
    def get_reward(self):

        reward = 0
        for j in self.machine.running_job:
            reward += self.pa.delay_penalty / float(j.len)

        for j in self.job_slot.slot:
            if j is not None:
                reward += self.pa.hold_penalty / float(j.len)

        return reward

    def step(self, a, repeat=False):
        
        status = None

        done = False
        reward = 0
        info = None
        a2 = a % 4

        if a == 32:  # explicit void action
            status = 'MoveOn'
        else:
            if a <= 16:
                a1 = int(np.floor(a/4))
                if self.job_slot.slot[a1] is None:
                    status = 'MoveOn'
                else:
                    allocated = self.machine.allocate_job(self.job_slot.slot[a1],a2, self.curr_time)
                    if not allocated:
                        status = 'MoveOn'
                    else:
                        status = 'Allocate'
                
            if a > 16:
                a1 = int(np.floor((a-16)/4))
                if self.machine.running_job[a1] is None:
                    status = 'MoveOn'
                else:
                    allocated = self.machine.reallocate_job(a1, a % 4,self_curr_time)
                    if not allocated:
                        status = 'MoveOn'
                    else:
                        status = 'Reallocate'
        
        if status == 'MoveOn':
            self.curr_time += 1
            self.machine.time_proceed(self.curr_time)
            
            self.seq_idx += 1
            self.job_idx += 1
            if self.end == "no_new_job":  # end of new job sequence
                if self.seq_idx >= self.pa.simu_len:
                    done = True
            elif self.end == "all_done":  # everything has to be finished
                if self.seq_idx >= self.pa.simu_len and \
                   len(self.machine.running_job) == 0 and \
                   all(s is None for s in self.job_slot.slot):
                   done = True
                elif self.curr_time > self.pa.episode_max_length:  # run too long, force termination
                    done = True
            
            if not done:

                if self.seq_idx < self.pa.simu_len:  # otherwise, end of new job sequence, i.e. no new jobs
                    job_num = 0
                    for i in range(np.ceil(self.pa.new_job_rate)):
                        new_job = self.get_new_job_from_seq(self.seq_no, self.seq_idx, job_num)
                        if new_job.len > 0:  # a new job comes
                            for j in range(self.pa.num_nw):
                                if self.job_slot.slot[j] is None:  # put in new visible job slots
                                    self.job_slot.slot[j] = new_job
                                    self.job_record.record[new_job.id] = new_job
                                    break
                        
                            self.extra_info.new_job_comes()
                        job_num += 1
                
            reward = self.get_reward()
        
        elif status = 'Allocate':
            self.job_record.record[self.job_slot.slot[a1].id] = self.job_slot.slot[a1]
            self.job_slot.slot[a1] = None
        
        elif status = 'Reallocate':

        ob = self.observe()
        info = self.job_record

        if done:
            self.seq_idx = 0

            if not repeat:
                self.seq_no = (self.seq_no + 1) % self.pa.num_ex

            self.reset()
        
        if self.render:
            self.plot_state()

        return ob, reward, done, info


    def reset(self):
        self.seq_idx = 0
        self.curr_time = 0

        # initialize system
        self.machine = Machine(self.pa)
        self.job_slot = JobSlot(self.pa)
        self.job_record = JobRecord()
        self.extra_info = ExtraInfo(self.pa)
        
class Job:
    def __init__(self,job_len, job_id, enter_time):
        self.id = job_id
        self.res = None
        self.len = job_len
        self.enter_time = enter_time
        self.start_time = -1  # not being allocated
        self.finish_time = -1


class JobSlot:
    def __init__(self, pa):
        self.slot = [None] * pa.num_nw

class JobRecord:
    def __init__(self):
        self.record = {}

class Machine:
    def __init__(self, pa):
        self.pa = pa
        self.num_res = pa.num_res
        self.time_horizon = pa.time_horizon
        self.res_slot = pa.res_slot

        self.avbl_slot = np.ones((self.time_horizon, self.num_res)) * self.res_slot

        self.running_job = []

        # graphical representation
        self.canvas = np.zeros((pa.num_res, pa.time_horizon, pa.res_slot))

    def allocate_job(self, job, qos, curr_time):

        allocated = False
        
        res = self.pa.qos_res_list(qos)

        for t in range(1, self.time_horizon - job.len):
            avbl_res = 
        
    def reallocate_job(self, index, qos, curr_time):

        allocated = False

    def time_proceed(self, curr_time):

        self.avbl_slot[:-1, :] = self.avbl_slot[1:, :]
        self.avbl_slot[-1, :] = self.res_slot

        for job in self.running_job:

            if job.finish_time <= curr_time:
                self.running_job.remove(job)

        # update graphical representation

        self.canvas[:, :-1, :] = self.canvas[:, 1:, :]
        self.canvas[:, -1, :] = 0
    

