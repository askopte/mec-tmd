import tensorflow as tf
import numpy as np

from functools import reduce
from operator import mul

class TFLearner:
    def __init__(self, pa, in_height, in_width, out_dim):

        self.input_height = in_height
        self.input_width = in_width
        self.output_height = out_dim
        self.num_features = self.input_height * self.input_width

        output_graph = 1

        self.num_frames = pa.num_frames

        self.update_counter = 0

        self.lr_rate = pa.lr_rate
        self.rms_rho = pa.rms_rho
        self.rms_eps = pa.rms_eps

        self.build_network()

        self.sess = tf.Session()
        self.saver = tf.train.Saver()

        if output_graph:
            tf.summary.FileWriter(pa.output_filename+"_graph.tmp", self.sess.graph)
        
        self.sess.run(tf.global_variables_initializer())

    
    def build_network(self):
        with tf.name_scope('inputs'):
            self.states = tf.placeholder(tf.float32,[None, self.num_features], name="observe")
            self.actions = tf.placeholder(tf.int32,[None, ], name="actions")
            self.values = tf.placeholder(tf.float32, [None, ],name="values")

        layer = tf.layers.dense(
            inputs = self.states,
            units = 32,
            activation = tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(value=0.1))
        
        act = tf.layers.dense(
            inputs = layer,
            units = self.output_height,
            activation = tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(value=0.1))

        self.all_act_prob = tf.nn.softmax(act, name = 'act_prob')

        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=act, labels=self.actions)
            self.loss = tf.reduce_mean(neg_log_prob * self.values)

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr_rate).minimize(self.loss)
        
    def choose_action(self, observe):

        obs = np.expand_dims(observe, axis=0)
        prob_weights = self.sess.run(self.all_act_prob, feed_dict={self.states: obs})
        action = np.random.choice(range(prob_weights.shape[1]), p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action

    def learn(self, obs, acts, vals):

        _, loss = self.sess.run([self.train_op, self.loss], feed_dict= {
        self.states: np.array(obs), 
        self.actions: np.array(acts), 
        self.values: np.array(vals),} 
        )

        return loss
    
    def get_num_params(self):
        num_params = 0
        for variable in tf.trainable_variables():
            shape = variable.get_shape()
            num_params += reduce(mul, [dim.value for dim in shape], 1)
        
        return num_params
    