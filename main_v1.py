from __future__ import print_function
import os
import tensorflow as tf
from tensorflow.contrib import rnn
from function import *
from read_input import *
from train import *
from testing import *
from rnn import *
import numpy as np
h=64; w=64;

#1: Log files
fout_log= open("log.txt","w")
fout_log.write("TEST LOG PRINT\nSOO\n")

#2: Graph
#Training Parameters
validation_step=10;
learning_rate =0.001
X = tf.placeholder("float", [FLAGS.batch_size, None, h,w,channels]) 
#SH: Need to correct y placeholder to get the exact corrdinate value: something like [FLAGS.batch_size, None, 1,1,1] where axis=2 getting x-coordinate, axis=3 getting y-coordinate, axis=4 getting confidence score: design by your own depending on shape of output label
Y = tf.placeholder("float", [FLAGS.batch_size, None, h,w,1]) 
timesteps = tf.shape(X)[1]
h=tf.shape(X)[2] 
w=tf.shape(X)[3] 

#SH: Need to correct ConvLSTM function in a way to regress out the exact coordinate(in rnn.py)
prediction, last_state = ConvLSTM(X)
loss_op=tf.losses.mean_pairwise_squared_error(Y,prediction)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

#3 :Training
with tf.Session() as sess:
    # Initialize all variables
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    sess.run(init)
    train_X,train_Y,test_X,test_Y,val_X,val_Y=read_input("./")
    print("finished collecting data")
    test("temp",sess,loss_op,train_op,X,Y,test_X,test_Y,prediction,last_state,fout_log)
    for ii in range(1000):
        train(sess,loss_op,train_op,X,Y,train_X,train_Y,val_X,val_Y,prediction, last_state,fout_log)
        name=str(ii)
        test(name,sess,loss_op,train_op,X,Y,test_X,test_Y,prediction,last_state,fout_log)
fout_log.close();

