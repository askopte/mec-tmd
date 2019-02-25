import numpy as np

def bin_to_dec(input):
    output = 0
    l = 2**(len(input)-1)

    for i in range(len(input)):

        if input[i] == 1:
            output += l

        l = l / 2
    
    return output

def dec_to_bin(input, dim):
    output = np.zeros(dim)
    l = 2**(dim-1)
    
    for i in range(dim):

        if input >= l:
            output[i] = 1
            input = input - l
        else:
            output[i] = 0

        l=l/2
    
    return output

def get_access_action(machine, job_slot):
    
    for i in reversed(range(4)):

        if job_slot.slot[i] is not None:

            return 4 * i 
    
    return 32

def get_quality_action(machine, job_slot):

    for i in reversed(range(4)):

        if job_slot.slot[i] is not None:

            return 4 * i + 3
    
    return 32

def get_random_action(job_slot):
    
    return int(np.floor(np.random.ranf()*33))

def get_greedy_action(pa, machine, job_slot):

    all_latency = pa.lte_latency + pa.mec_overall_latency

    if job_slot.slot[0] is None:
        return 32

    min_start_time = 0

    for i in range(1,4):
        if job_slot.slot[i] is not None:
            if job_slot.slot[i].enter_time < job_slot.slot[min_start_time].enter_time:
                min_start_time = i
    
    i = min_start_time

    for j in reversed(range(4)):

        new_avbl_slot = np.zeros(pa.time_horizon)

        for t in range(pa.time_horizon):
            new_avbl_slot[t] = machine.avbl_slot[t,0]

        for temp_job in job_slot.slot:
            if temp_job is not None:
                new_avbl_slot[all_latency : all_latency + temp_job.len] = new_avbl_slot[all_latency : all_latency + temp_job.len] - pa.qos_res_list[j]

        if np.min(new_avbl_slot[all_latency : all_latency + job_slot.slot[i].len]) >= 0:

            test1 = np.min(machine.avbl_slot[all_latency : all_latency+job_slot.slot[i].len, 0])

            return i * 4 + j
    
    return 32
                

def main():

    # Test Program

    inp = 1564
    out1 = dec_to_bin(inp, 11)
    out1
    out2 = bin_to_dec(out1)
    out2

if __name__ == '__main__':
    main()


