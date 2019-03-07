import numpy as np
import parameters

class Dist:

    def __init__(self, job_len):

        self.job_len = job_len

        self.job_small_chance = 0.8

        self.job_len_big_lower = job_len * 1 / 2
        self.job_len_big_upper = job_len * 3 / 4

        self.job_len_small_lower = 10
        self.job_len_small_upper = job_len / 4
    
    def job_dist(self):

        # -- job length --
        if np.random.ranf() < self.job_small_chance:  # small job
            nw_len = np.random.randint(self.job_len_small_lower,
                                       self.job_len_small_upper + 1)
        else:  # big job
            nw_len = np.random.randint(self.job_len_big_lower,
                                       self.job_len_big_upper + 1)

        return nw_len

def generate_sequence_work(pa):

    num_ex = pa.num_ex
    simu_len = pa.simu_len
    job_rate = pa.new_job_rate

    nw_dist = pa.dist.job_dist

    nw_len_seq = np.zeros([num_ex, int(simu_len * np.ceil(job_rate))], dtype=int)
    for i in range(num_ex):
        for j in range(simu_len):
            job_no = 0
            for k in range(int(np.floor(job_rate))):
                nw_len_seq[i, int(np.ceil(job_rate) * j + job_no)] = nw_dist()
                job_no += 1
                
            if np.random.ranf() < job_rate - np.floor(job_rate): 
                nw_len_seq[i, int(np.ceil(job_rate) * j + job_no)] = nw_dist()
                job_no += 1
        
    return nw_len_seq

def generate_sequence_ue_ambr(pa):

    num_ex = pa.num_ex
    simu_len = pa.episode_max_length

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
