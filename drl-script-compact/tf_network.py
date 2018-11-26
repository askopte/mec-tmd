import tensorflow as tf
import numpy as numpy

from functools import reduce
from operator import mul

class TFLearner:
    def __init__(self, pa):

        self.input_height = pa.network_input_height
        self.input_width = pa.network_input_width
        self.output_height = pa.network_output_dim
        self.num_features = self.input_height * self.input_width

        self.num_frames = pa.num_frames

        self.update_counter = 0

        self.lr_rate = pa.lr_rate
        self.rms_rho = pa.rms_rho
        self.rms_eps = pa.rms_eps

        self.states = []
        self.actions = []
        self.values = []

        self
    
    def build_network():
        with tf.name_scope('inputs'):
            self.states = tf.placeholder(tf.int16,[None, self.num_features], name="observe")
            self.actions = tf.placeholder(tf.int16,[None, 2, ], name="actions")
            self.values = tf.placeholder(tf.float16, [None, ],name="values")

        layer1 = tf.layers.dense(
            inputs = self.states,
            units = 32,
            activation = tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1))
        
        layer2a = tf.layers.dense(
            inputs = self.layer1,
            units = 32,
            activation = tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1))

        layer2b = tf.layers.dense(
            inputs = self.layer1,
            units = 32,
            activation = tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1))
        
        action1 = tf.layers.dense(
            inputs = self.layer2a,
            units = 32,
            activation = tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1))

        action2 = f.layers.dense(
            inputs = self.layer2b,
            units = 8,
            activation = tf.nn.relu,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1))

        self.all_act_prob = tf.nn.softmax(action1, name = 'act1_prob'), tf.nn.softmax(action2, name = 'act2_prob')

        with tf.name_scope('loss'):

        