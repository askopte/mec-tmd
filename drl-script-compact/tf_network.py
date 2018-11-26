import tensorflow as tf
import numpy as numpy

from functools import reduce
from operator import mul

class TFLearner:
    def __init__(self, pa):

        self.input_height = pa.network_input_height
        self.input_width = pa.network_input_width
        self.output_height = pa.network_output_dim

        self.num_frames = pa.num_frames

        self.update_counter = 0

        self.lr_rate = pa.lr_rate
        self.rms_rho = pa.rms_rho
        self.rms_eps = pa.rms_eps

        

