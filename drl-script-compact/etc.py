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
    
    for i in range(4):
        if job_slot.slot[i] is not None:
            return 4 * i + 1
    
    return 0

def get_quality_action(machine, job_slot):

    for i in range(4):
        if job_slot.slot[i] is not None:
            return 4 * i + 3
    
    return 0

def get_random_action(machine):

    return int(np.random.ranf()*32)

def get_greedy_action(machine, job_slot):

    for i in range(4):
        if job_slot.slot[i] is not None:
            for j in range(4):
                if machine.avbl_slot[8:8+]
                

def main():

    # Test Program

    inp = 1564
    out1 = dec_to_bin(inp, 11)
    out1
    out2 = bin_to_dec(out1)
    out2

if __name__ == '__main__':
    main()


