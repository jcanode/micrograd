"""Add rnn class as decribed in http://karpathy.github.io/2015/05/21/rnn-effectiveness/ """
from micrograd.engine import Value
from micrograd import nn
import numpy as np

class RNN:
    def __init__(self,hidden_size, input_size, output_size):    
        self.h = np.zeros((hidden_size, 1)) # initalize hidden state to 0.
        self.W_hh = np.random.randn(hidden_size,hidden_size)*.01 # hidden to hiden weights 
        self.W_xh = np.random.randn(hidden_size,input_size)*.01 # input to hidden weights
        self.W_hy = np.random.randn(output_size,hidden_size)*.01 # hidden to output weights 
        self.x =  np.random.randn(1,input_size)

    def step(self,x):
        # update the hidden state
        self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x)) # np.dot((25,25),(25,1)) + np.dot((25,106),(106,25))
        # converts h from (hidden size,1) to (hidden size, hiddensize)
        
        # compute the output vector
        y = np.dot(self.W_hy, self.h)

        return y
    

