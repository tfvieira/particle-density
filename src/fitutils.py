import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

import datetime

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

NEW_SHAPE = (32, 32)

#%%
def gaussian(O, x = np.linspace(0,32,32)):
    
    u1, u2, sig21, sig22, p = tuple(O)
    
    g1 = p*tf.exp(-0.5*(x-u1)**2/sig21)
    g2 = (1-p)*tf.exp(-0.5*(x-u2)**2/sig22)
    g  = g1 + g2
    
    return g

def plot_donut_1D():

    O = [16, 16, 10, 2, 1.5]
    x = np.linspace(0,32,100)
    y = gaussian(O, x)
    plt.plot(x, y, 'k-')
    plt.show()

def donut(O, N = 32):

    p1, S1, p2, S2 = O

    x = tf.linspace(-1,1,32)
    y = tf.linspace(-1,1,32)
    X, Y = tf.meshgrid(x, y)

    # I = p1 * tf.exp(-0.5*(((X-u1_1)**2)*S1 + ((Y-u1_2)**2)*S1)) - p2 * tf.exp(-0.5*(((X-u2_1)**2)*S2 + ((Y-u2_2)**2)*S2))
    I = p1 * tf.exp(-0.5*(((X)**2)*S1 + ((Y)**2)*S1)) - p2 * tf.exp(-0.5*(((X)**2)*S2 + ((Y)**2)*S2))
    
    return I

def gaussian2D(O, N = 32):

    u1, u2, S, p = tuple(O)

    x = tf.linspace(-1,1,32)
    y = tf.linspace(-1,1,32)
    X, Y = tf.meshgrid(x, y)

    I = p * tf.exp(-0.5*(((X-u1)**2)*S + ((Y-u2)**2)*S))
    
    return I

def fit_gaussian_on_vector(y, starting_O = [16, 16, 10, 2, 3]):


    if starting_O == None:
        u1 = tf.Variable(0.0)
        sig21 = tf.Variable(1.0)
        u2 = tf.Variable(0.0)
        sig22 = tf.Variable(1.0)
        p = tf.Variable(0.0)
    else:
        u1    = tf.Variable(starting_O[0])
        u2    = tf.Variable(starting_O[1])
        sig21 = tf.Variable(starting_O[2])
        sig22 = tf.Variable(starting_O[3])
        p     = tf.Variable(starting_O[4])
        O = [u1, u2, sig21, sig22, p]

    losses = fit(gaussian, O, y, max_epoch=2000)

    return losses

def fit_gaussian_on_image (img):

    bg = (img[0,0] + img[-1,-1] + img[-1,0] + img[0,-1]) / 4.0

    img = img - bg
    img = img / np.sqrt(np.mean(img**2))

    u1_1  = tf.Variable(0.0, dtype=tf.float64)
    u1_2  = tf.Variable(0.0, dtype=tf.float64)
    S1    = tf.Variable(1.0, dtype=tf.float64)
    p1    = tf.Variable(1.0, dtype=tf.float64)

    O = [u1_1, u1_2, S1, p1]

    losses = fit(gaussian2D, O, img)

    return (O, losses)

def fit_donut_on_image (img):

    u1_1  = tf.Variable(0.0, dtype=tf.float64)
    u1_2  = tf.Variable(0.0, dtype=tf.float64)
    p1  = tf.Variable(0.0, dtype=tf.float64)
    S1  = tf.Variable(0.0, dtype=tf.float64)

    p2  = tf.Variable(0.0, dtype=tf.float64)
    S2  = tf.Variable(0.0, dtype=tf.float64)

    O = [p1, S1, p2, S2]

    losses = fit(donut, O, img, max_epoch=200)

    return (O, losses)

def domain(start=-1.0, stop=-1.0, shape=NEW_SHAPE):

    t    = tf.linspace(-1.0, 1.0, shape[0])
    x, y = tf.cast(tf.meshgrid(t, t), tf.float64)

    return x, y

def model(O):
    """

    """
    # ux, uy, s2, p = O

    # Get required x, y domain and model parameters
    x, y = domain(shape=NEW_SHAPE)

    # Compute the model function
    z = O[3] * tf.exp( -( ( ((x - O[0])**2.0)/(2.0 * O[2]) + 
                            ((y - O[1])**2.0)/(2.0 * O[2]) ) ) ** O[4] )

    return z

def fit(y):

    function      = model
    max_epoch     = 1000
    loss_function = tf.keras.losses.mean_squared_error
    optimizer     = tf.keras.optimizers.SGD(learning_rate = 2.0)
    callbacks     = [tensorboard_callback]

    # Define parameters
    ux    = tf.Variable(0.0, dtype=tf.float64)
    uy    = tf.Variable(0.0, dtype=tf.float64)
    s2    = tf.Variable(0.5, dtype=tf.float64)
    p     = tf.Variable(0.1, dtype=tf.float64)
    expo  = tf.constant(4.0, dtype=tf.float64)

    O         = [ux,   uy,   s2,   p,    expo]
    trainable = [True, True, True, True, False]

    i = 0
    losses = []

    for epoch in range(max_epoch):

        with tf.GradientTape(persistent=True) as tape:

            vars = [x[0] for x in zip(O, trainable) if x[1]] 
            
            tape.watch(vars)

            y_hat = function(O)
          
            loss = tf.reduce_mean(loss_function(y, y_hat)) 

        grads = tape.gradient(loss, vars)
        optimizer.apply_gradients(zip(grads, vars))

        if i%100 == 0:
            print('batch : %d of %d (loss = %.4f)'%(i, max_epoch, loss))
            losses.append(loss.numpy())

        i = i + 1
    
    return O, losses
