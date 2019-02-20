import numpy as np

class Dist:

    def __init__(self, job_len):

        self.job_len = job_len

        self.job_small_chance = 0.6

        self.job_len_big_lower = job_len * 2 / 3
        self.job_len_big_upper = job_len

        self.job_len_small_lower = 10
        self.job_len_small_upper = job_len / 5
    
    def job_dist(self):

        # -- job length --
        if np.random.ranf() < self.job_small_chance:  # small job
            nw_len = np.random.randint(self.job_len_small_lower,
                                       self.job_len_small_upper + 1)
        else:  # big job
            nw_len = np.random.randint(self.job_len_big_lower,
                                       self.job_len_big_upper + 1)

        return nw_len

def generate_sequence_work(self, pa):

    simu_len = pa.simu_len * pa.num_ex

    nw_len_seq = np.zeros(simu_len, dtype=int)

    for i in range(simu_len):

        if np.random.ranf() < pa.new_job_rate:  # a new job comes

            nw_len_seq[i] = self.job_dist()

    nw_len_seq = np.reshape(nw_len_seq,
                            [pa.num_ex, pa.simu_len])

    return nw_len_seq
