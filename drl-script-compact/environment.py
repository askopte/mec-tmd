import numpy as np
import math
import matplotlib as plt
import tensorflow

import parameters
import etc

class Env:
    def __init__(self, pa, nw_len_seqs = None,  
                 render = False, end = 'no_new_job'):
        
        self.pa = pa
        self.render = render
        self.end = end  # termination type, 'no_new_job' or 'all_done'

        self.nw_dist = pa.dist.job_dist

        self.curr_time = 0

        if nw_len_seqs is None :
            # generate new work
            self.nw_len_seqs = \
                self.generate_sequence_work(self.pa.num_ex, self.pa.simu_len, self.pa.new_job_rate)
        
        else:
            self.nw_len_seqs = nw_len_seqs
        
        self.nw_ambr_seqs = self.generate_sequence_ue_ambr(self.pa.num_ex,self.pa.episode_max_length)

        self.seq_no = 0  # which example sequence
        self.seq_idx = 0  # index in that sequence

        # initialize system
        self.machine = Machine(pa)
        self.job_slot = JobSlot(pa)
        self.job_record = JobRecord()

    def generate_sequence_ue_ambr(self, num_ex, simu_len):

        nw_ambr_seq = np.zeros([num_ex, simu_len], dtype = int)
        nw_ambr_seq[:,0:9] = 2 
        for i in range(num_ex):
            for j in range(int(simu_len / 10) - 1):
                ran = np.random.ranf()
                if ran < 0.25:
                    if nw_ambr_seq[i, 10*j + 9] <= 2:
                        nw_ambr_seq[i,10*j + 10:10*j + 19] =  nw_ambr_seq[i, 10*j-1] + 1
                if ran > 0.75:
                    if nw_ambr_seq[i, 10*j + 9] >= 1:
                        nw_ambr_seq[i,10*j + 10:10*j + 19] =  nw_ambr_seq[i, 10*j-1] - 1
        
        return nw_ambr_seq


    def generate_sequence_work(self, num_ex, simu_len, job_rate):

        nw_len_seq = np.zeros([num_ex, int(simu_len * np.ceil(job_rate))], dtype=int)
        for i in range(num_ex):
            for j in range(simu_len):
                job_no = 0
                for k in range(int(np.floor(job_rate))):
                    nw_len_seq[i, int(np.ceil(job_rate) * j + job_no)] = self.nw_dist()
                    job_no += 1
                
                if np.random.ranf() < job_rate - np.floor(job_rate):  # a new job comes
                    nw_len_seq[i, int(np.ceil(job_rate) * j + job_no)] = self.nw_dist()
                    job_no += 1
        
        return nw_len_seq

    def get_new_job_from_seq(self, seq_no, seq_idx, job_no):
        new_job = Job (job_len=self.nw_len_seqs[seq_no, int(seq_idx * np.ceil(self.pa.new_job_rate)) + job_no],
                       job_id=len(self.job_record.record),
                       enter_time=self.curr_time)
        return new_job

    def observe(self):

        image_repr = np.zeros((self.pa.network_input_height, self.pa.network_input_width))

        ir_pt = 0

        for i in range(self.pa.num_res): #res_slot repre
                
            image_repr[:, ir_pt: ir_pt + self.pa.res_slot] = self.machine.canvas[i, :, :]
            ir_pt += self.pa.res_slot

        for i in range(self.pa.nw_width): # nw_width

            for j in range(self.pa.time_horizon):
                if self.job_slot.slot[i * self.pa.time_horizon + j] is not None:
                    image_repr[j, ir_pt : ir_pt + 1] = self.job_slot.slot[i * self.pa.time_horizon + j].len
                    image_repr[j, ir_pt + 1 : ir_pt + 2] = self.curr_time - self.job_slot.slot[i * self.pa.time_horizon + j].enter_time
                
            ir_pt += 2
                
        for i in range(self.pa.job_width): # job_width

            for j in range(self.pa.time_horizon):
                #if self.machine.running_job[i * self.pa.time_horizon + j] is not None:

                if len(self.machine.running_job) > i * self.pa.time_horizon + j:
                    if self.machine.running_job[i * self.pa.time_horizon + j] is not None:
                        image_repr[j, ir_pt : ir_pt + 1] = self.machine.running_job[i * self.pa.time_horizon + j].remain_len
                        image_repr[j, ir_pt + 1 : ir_pt + 2] = self.machine.running_job[i * self.pa.time_horizon + j].res
                
            ir_pt += 2
            
        for i in range(self.pa.ambr_len):
            image_repr[i, ir_pt:ir_pt + 1] = self.nw_ambr_seqs[self.seq_no, self.curr_time + i]
        
        ir_pt += 1

        assert ir_pt == image_repr.shape[1]

        return image_repr.ravel()
    
    def get_reward(self):

        reward = 0
        for j in self.machine.running_job:
            reward += self.pa.qos_rew_list[j.res] 
            if j.res > self.nw_ambr_seqs[self.seq_no, self.curr_time]:
                for i in range(self.nw_ambr_seqs[self.seq_no, self.curr_time], j.res):
                    reward += self.pa.qos_rew_delta[i]

        for j in self.machine.pending_job:
            reward += self.pa.hold_penalty 

        for j in self.job_slot.slot:
            if j is not None:
                if self.curr_time - j.enter_time <= 8:
                    reward -= 2
                elif self.curr_time - j.enter_time > 20:
                    reward -= 64
                else:
                    reward -= 2**int((self.curr_time - j.enter_time - 8)/2)
        
        # reward += self.machine.avbl_slot[0,1] - (2**self.machine.res_slot)
        
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
            if a < 16:
                a1 = int(np.floor(a/4))
                if self.job_slot.slot[a1] is None:
                    status = 'MoveOn'
                else:
                    allocated = self.machine.allocate_job(self.job_slot.slot[a1],a2, self.curr_time)
                    if not allocated:
                        status = 'MoveOn'
                    else:
                        status = 'Allocate'
                
            if a >= 16:
                a1 = int(np.floor((a-16)/4))
                if len(self.machine.running_job) < a1:
                    status = 'MoveOn'
                else:
                    allocated = self.machine.reallocate_job(a1, a2,self.curr_time)
                    if not allocated:
                        status = 'MoveOn'
                    else:
                        status = 'Reallocate'
        
        if status == 'MoveOn':
            self.curr_time += 1
            self.machine.time_proceed(self.curr_time)
            
            self.seq_idx += 1
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
                    for i in range(int(np.ceil(self.pa.new_job_rate))):
                        new_job = self.get_new_job_from_seq(self.seq_no, self.seq_idx, job_num)
                        if new_job.len > 0:  # a new job comes
                            
                            exceed = True
                            for j in range(self.pa.num_nw):
                                if self.job_slot.slot[j] is None:  # put in new visible job slots
                                    self.job_slot.slot[j] = new_job
                                    self.job_record.record[new_job.id] = new_job
                                    exceed = False
                                    break
                            
                            if exceed:
                                print("Resource Shortage.")
                        job_num += 1
                
            reward = self.get_reward()
        
        elif status is 'Allocate':
            self.job_record.record[self.job_slot.slot[a1].id] = self.job_slot.slot[a1]
            self.job_slot.slot[a1] = None
            for i in range(a1,self.pa.num_nw - 1):
                self.job_slot.slot[i] = self.job_slot.slot[i+1]
        
        elif status is 'Reallocate':
            self.job_record.record[self.machine.running_job[a1].id] = self.machine.running_job[a1]
            self.machine.running_job.remove(self.machine.running_job[a1])

        ob = self.observe()
        info = self.job_record

        if done:
            self.seq_idx = 0

            if not repeat:
                self.seq_no = (self.seq_no + 1) % self.pa.num_ex

            self.reset()

        return ob, reward, done, info


    def reset(self):
        self.seq_idx = 0
        self.curr_time = 0

        # initialize system
        self.machine = Machine(self.pa)
        self.job_slot = JobSlot(self.pa)
        self.job_record = JobRecord()
        
class Job:
    def __init__(self,job_len, job_id, enter_time):
        self.id = job_id
        self.res = None
        self.len = job_len
        self.enter_time = enter_time
        self.start_time = -1  # not being allocated
        self.finish_time = -1
        self.remain_len = -1



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

        self.avbl_slot = np.ones((self.time_horizon, self.num_res)) * (2**self.res_slot)

        self.pending_job = []
        self.running_job = []

        # graphical representation
        self.canvas = np.zeros((pa.num_res, pa.time_horizon, pa.res_slot))

    def allocate_job(self, job, qos, curr_time):

        allocated = False
        
        res = self.pa.qos_res_list[qos]

        all_latency = self.pa.lte_latency + self.pa.mec_overall_latency

        for t in range(1, self.time_horizon - job.len - all_latency):
            new_avbl_res = np.zeros(job.len)
            backhaul_res = np.zeros(all_latency)

            for i in range(0,all_latency):
                backhaul_res[i] = self.avbl_slot[t+i,1] - 1

            for i in range(0, job.len):
                new_avbl_res[i] = self.avbl_slot[t+i+all_latency,0] - res
            
            if np.all(new_avbl_res[:] >= 0) and np.all(backhaul_res[:] >= 0):

                allocated = True
                for i in range(0,all_latency):
                    self.avbl_slot[t+i,1] = backhaul_res[i]
                
                for i in range(0,job.len):
                    self.avbl_slot[t+i+all_latency,0] = new_avbl_res[i]
                
                job.start_time = curr_time + t + all_latency
                job.finish_time = job.start_time + job.len
                job.remain_len = job.len
                job.res = qos

                self.pending_job.append(job)

                # update graphical representation(incompleted)

                assert job.start_time != -1
                assert job.finish_time != -1
                assert job.finish_time >= job.start_time
                canvas_start_time = job.start_time - curr_time
                canvas_end_time = job.finish_time - curr_time

                for res in range(self.num_res):
                    for i in range(canvas_start_time, canvas_end_time):
                        self.canvas[res, i, :] = etc.dec_to_bin(self.avbl_slot[i, res],self.pa.res_slot)

                break

        return allocated
        
    def reallocate_job(self, index, qos, curr_time):

        allocated = False
        
        res = self.pa.qos_res_list[qos]

        if len(self.running_job) > index:

            job = self.running_job[index]

            all_latency = self.pa.lte_latency

            released_res = np.zeros(self.time_horizon)

            for i in range(0,job.remain_len):
                released_res[i] = self.avbl_slot[i,0] + self.pa.qos_res_list[job.res]
        
            for i in range(job.remain_len, self.time_horizon):
                released_res[i] = self.avbl_slot[i,0]

            for t in range(1, self.time_horizon - job.remain_len - all_latency):
                new_avbl_res = np.zeros(job.remain_len)

                for i in range(0, job.remain_len):
                    new_avbl_res[i] = released_res[i+t+all_latency] - res
            
                if np.all(new_avbl_res[:] >= 0):
                    allocated = True
                
                    for i in range(0,self.time_horizon):
                        self.avbl_slot[i,0] = released_res[i]

                    for i in range(0,job.remain_len):
                        self.avbl_slot[t+i+all_latency,0] = new_avbl_res[i]
                
                    job.start_time = curr_time + t + all_latency
                    job.finish_time = job.start_time + job.remain_len

                    self.pending_job.append(job)

                    # update graphical representation(incompleted)

                    assert job.start_time != -1
                    assert job.finish_time != -1
                    assert job.finish_time >= job.start_time
                    canvas_start_time = job.start_time - curr_time
                    canvas_end_time = job.finish_time - curr_time

                    for res in range(self.num_res):
                        for i in range(canvas_start_time, canvas_end_time):
                            self.canvas[res, i, :] = etc.dec_to_bin(self.avbl_slot[i, res],self.pa.res_slot)

                    break
        
        return allocated

    def time_proceed(self, curr_time):

        self.avbl_slot[:-1, :] = self.avbl_slot[1:, :]
        self.avbl_slot[-1, :] = 2**self.res_slot

        for job in self.pending_job:

            if job.start_time <= curr_time:
                self.running_job.append(job)
                self.pending_job.remove(job)

        for job in self.running_job:
            
            job.remain_len -= 1 
            if job.finish_time <= curr_time:
                self.running_job.remove(job)

        # update graphical representation

        self.canvas[:, :-1, :] = self.canvas[:, 1:, :]
        self.canvas[:, -1, :] = 0

