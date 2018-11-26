import tensorflow as tf
import numpy as np

class TFLearner:
    def __init__(self,pa):
        
        self.input_height = pa.network_input_height
        self.input_width = pa.network_input_width
        self.output_height = pa.network_output_dim

        self.num_frames = pa.num_frames

        self.update_counter = 0

        states = np.zeros()
        actions = np.zeros()

        print 'network_input_height=', pa.network_input_height
        print 'network_input_width=', pa.network_input_width

        Input = tf.keras.layers.input(shape = (pa.network_input_height, pa.network_input_width))

        Layer1 = tf.keras.layers.dense(32,activation = "relu")(Input)
        LayerA2 = tf.keras.layers.dense(32,activation = "relu")(Layer1)
        LayerB2 = tf.keras.layers.dense(32,activation = "relu")(Layer1)
        OutputA = tf.keras.layers.dense(32,activation = "softmax")(LayerA2)
        OutputB = tf.keras.layers.dense(32,activation = "softmax")(LayerB2)

        self.model = tf.keras.models.Model(inputs = self.Input, outputs = [self.OutputA, self.OutputB])

        self.lr_rate = pa.lr_rate
        self.rms_rho = pa.rms_rho
        self.rms_eps = pa.rms_eps

        params = self.model.get_weights()

        print ' params=', params, ' count=', self.model.count_params()

        # ===================================
        # training function part
        # ===================================

        prob_act = self.model.predict()

        

        









        

