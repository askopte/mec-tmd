import numpy as np
import time
import tensorflow
import matplotlin.pyplot as plt

import environment
import pg_network
import slow_down_cdf

def launch(pa, pg_resume=None, render=False, repre='image', end='no_new_job'):

    env = environment.Env(pa, render=render, repre=repre, end=end)
    