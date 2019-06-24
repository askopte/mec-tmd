from tensorflow import keras
import numpy as np

class TFLearner:
    def __init__(self, pa, in_height, in_width, out_dim):
        
        self.input_height = in_height
        self.input_width = in_width
        self.output_height = out_dim
        self.num_features = self.input_height * self.input_width

        self.num_frames = pa.num_frames

        self.update_counter = 0

        states = np.zeros()
        actions = np.zeros()
        values = np.zeros()

        print ('network_input_height=', pa.network_input_height)
        print ('network_input_width=', pa.network_input_width)

        Input = keras.layers.input(shape = (self.num_feature,))
        Layer1 = keras.layers.dense(32,activation = "tanh")(Input)
        Layer2 = keras.layers.dense(32,activation = "tanh")(Layer1)
        Output = keras.layers.dense(32,activation = "softmax")(Layer2)

        self.model = keras.models.Model(inputs = self.Input, outputs = self.Output)

        self.lr_rate = pa.lr_rate
        self.rms_rho = pa.rms_rho
        self.rms_eps = pa.rms_eps

        model.compile(optimizer=keras.optimizers.Adam(lr=self.lr_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), loss='categorical_crossentropy', metrics=['accuracy'])
    
    def choose_action(self, observe):
    
    def learn(self, obs, acts, vals):
    
    def get_num_params(self):
    
    def load_data(self, resume):




        

        









        

