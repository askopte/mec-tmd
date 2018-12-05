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

def main():

    # Test Program

    inp = 1564
    out1 = dec_to_bin(inp, 11)
    out1
    out2 = bin_to_dec(out1)
    out2

if __name__ == '__main__':
    main()


