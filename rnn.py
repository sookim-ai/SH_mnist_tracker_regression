from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from function import *
from tensorflow.contrib import rnn

h=64
w=64
feature_size=w*h; channels=1
testing_step=100;
training_steps = 200000
# Network Parameters
number_of_layers=2; #Start from only one layer

def ConvLSTM(x):
#SH: you can tune sub parameters in ConvLSTM such as channels and so on.
    convlstm_layer1 = tf.contrib.rnn.ConvLSTMCell(
                 conv_ndims=2,
                 input_shape=[h,w,channels],
                 output_channels=4,
                 kernel_shape=[3,3],
                 use_bias=True,
                 skip_connection=False,
                 forget_bias=0.5,
                 initializers=None,
                 name="conv_lstm_cell1")
    convlstm_layer2 = tf.contrib.rnn.ConvLSTMCell(
                 conv_ndims=2,
                 input_shape=[h,w,4],
                 output_channels=8,
                 kernel_shape=[3,3],
                 use_bias=True,
                 skip_connection=False,
                 forget_bias=0.5,
                 initializers=None,
                 name="conv_lstm_cell2")
    convlstm_layer3 = tf.contrib.rnn.ConvLSTMCell(
                 conv_ndims=2,
                 input_shape=[h,w,8],
                 output_channels=16,
                 kernel_shape=[7,7],
                 use_bias=True,
                 skip_connection=False,
                 forget_bias=0.5,  
                 initializers=None,
                 name="conv_lstm_cell3")
    convlstm_layer4 = tf.contrib.rnn.ConvLSTMCell(
                 conv_ndims=2,
                 input_shape=[h,w,16],
                 output_channels=1,
                 kernel_shape=[7,7],
                 use_bias=True,
                 skip_connection=False,
                 forget_bias=0.5,
                 initializers=None,
                 name="conv_lstm_cell4")
    stacked_lstm = tf.contrib.rnn.MultiRNNCell(
        [convlstm_layer1, convlstm_layer2, convlstm_layer3, convlstm_layer4])
    print(stacked_lstm)
    initial_state=stacked_lstm.zero_state(FLAGS.batch_size, dtype=tf.float32 )
    #SH: Things to Do
    #Put FC-layer which getting outputs(below) of dynamic_rnn as input and obtain the x,y,and confidence score. I think we need seperate FC-layer for condifence score.
    #We do not need the output with same sized as input, so the ConvLSTM can be desidned ad encodding format. (something like getting shallower as deeper layer..) 
    outputs, states=tf.nn.dynamic_rnn(cell=stacked_lstm, inputs=x, sequence_length=None, dtype=tf.float32, initial_state=initial_state)
    return outputs, states
# Didn't apply drop_out here

