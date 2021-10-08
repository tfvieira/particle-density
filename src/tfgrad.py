#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 18:46:33 2021

@author: vieira
"""

#%%
import tensorflow as tf
print(f'Executing tensorflow eagarly?\n{tf.executing_eagerly()}')

#%%
w = tf.Variable(tf.random.normal((3,2)), name='w')
b = tf.Variable(tf.zeros(2, dtype=tf.float32), name='b')
x = [[1., 2., 3.]]

with tf.GradientTape(persistent = True) as tape:
    y = x @ w + b
    loss = tf.reduce_mean(y**2)

#%%
x = tf.linspace(-1,1,32)
y = tf.linspace(-1,1,32)
X, Y = tf.meshgrid(x, y)

u1 = tf.Variable(0.0, dtype=tf.float64)
u2 = tf.Variable(0.0, dtype=tf.float64)
S  = tf.Variable(1.0, dtype=tf.float64)

O = [u1, u2, S]
I = tf.exp(-0.5*(((X-u1)**2)*S + ((Y-u2)**2)*S))