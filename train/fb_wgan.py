# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:01:31 2019

@author: Jeric
"""

import tensorflow as tf
from fb_wgan_data import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


#=============================================
#Hyperparameters
#=============================================

mb_size = 200   #minibatch size
Z_dim = 2       #noise vevtor length
X_dim = 2       #output vector length
y_dim = 2       #condition vector length 
h_dim = 16      #no. of neurons on hidden layers
D_iter = 5      #critic iterations per generator iteration 
lam = 10        #gradient penalty weight

#=============================================
#Parameters for output files
#============================================= 

epoch_iter = int(Train_dim/mb_size)
weight_out = 5*epoch_iter
max_step = 100*epoch_iter + 1
un_normal(seed = None)


def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


""" Discriminator Net model """
X = tf.placeholder(tf.float32, shape=[None, X_dim])
y = tf.placeholder(tf.float32, shape=[None, y_dim])

D_W1 = tf.Variable(xavier_init([X_dim + y_dim, h_dim]))
D_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
D_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W3 = tf.Variable(xavier_init([h_dim, h_dim]))
D_b3 = tf.Variable(tf.zeros(shape=[h_dim]))

D_W4 = tf.Variable(xavier_init([h_dim, 1]))
D_b4 = tf.Variable(tf.zeros(shape=[1]))

theta_D = [D_W1, D_W2, D_W3, D_W4, D_b1, D_b2, D_b3, D_b4]

#=============================================
#Discriminator structure
#============================================= 

def discriminator(x, y):
    inputs = tf.concat(axis=1, values=[x, y])
    D_h1 = tf.nn.leaky_relu(tf.matmul(inputs, D_W1) + D_b1)
    D_h2 = tf.nn.leaky_relu(tf.matmul(D_h1, D_W2) + D_b2)
    D_h3 = tf.nn.leaky_relu(tf.matmul(D_h2, D_W3) + D_b3)
    D_logit = tf.matmul(D_h3, D_W4) + D_b4

    return D_logit


""" Generator Net model """
Z = tf.placeholder(tf.float32, shape=[None, Z_dim])

G_W1 = tf.Variable(xavier_init([Z_dim + y_dim, h_dim]))
G_b1 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W2 = tf.Variable(xavier_init([h_dim, h_dim]))
G_b2 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W3 = tf.Variable(xavier_init([h_dim, h_dim]))
G_b3 = tf.Variable(tf.zeros(shape=[h_dim]))

G_W4 = tf.Variable(xavier_init([h_dim, X_dim]))
G_b4 = tf.Variable(tf.zeros(shape=[X_dim]))

theta_G = [G_W1, G_W2, G_W3, G_W4, G_b1, G_b2, G_b3, G_b4]


#=============================================
#Generator structure
#=============================================

def generator(z, y):
    inputs = tf.concat(axis=1, values=[z, y])
    G_h1 = tf.nn.leaky_relu(tf.matmul(inputs, G_W1) + G_b1)
    G_h2 = tf.nn.leaky_relu(tf.matmul(G_h1, G_W2) + G_b2)
    G_h3 = tf.nn.leaky_relu(tf.matmul(G_h2, G_W3) + G_b3)
    G_log_prob = tf.matmul(G_h3, G_W4) + G_b4

    return G_log_prob


def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])

#=============================================
#Building the optimization problem
#=============================================

G_sample = generator(Z, y)
D_real = discriminator(X, y)
D_fake = discriminator(G_sample, y)

eps = tf.random_uniform([mb_size, 1], minval=0., maxval=1.)
X_inter = eps*X + (1. - eps)*G_sample
grad = tf.gradients(discriminator(X_inter,y), [X_inter])[0]
grad_norm = tf.sqrt(tf.reduce_sum((grad)**2, axis=1))
grad_pen = lam * tf.reduce_mean((grad_norm - 1)**2)     #gradient penalty for WGAN

D_loss = tf.reduce_mean(D_fake) - tf.reduce_mean(D_real) + grad_pen
G_loss = -tf.reduce_mean(D_fake)

D_solver = (tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(D_loss, var_list=theta_D)) 
G_solver = (tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.5).minimize(G_loss, var_list=theta_G))

#=============================================
#Training
#=============================================

result_dir = './results/' 
saver = tf.train.Saver()

sess = tf.Session()
sess.run(tf.global_variables_initializer())


#outputing contours for online assessment of convergence

x_train = sample_all_train()
plt.figure()
xax = plt.scatter(x_plot[:,0], x_plot[:,1])
xax_train = plt.hist2d(x_train[0][0],x_train[1][0], bins = (500,500), cmap = plt.cm.jet)
plt.colorbar()
plt.title('Real Data (Train)')
plt.tight_layout()
plt.savefig('plots/iterations/real_data_train.png')
plt.close()

x_valid = sample_all_valid()
plt.figure()
#xax = plt.scatter(x_plot[:,0], x_plot[:,1])
xax_valid = plt.hist2d(x_valid[0][0],x_valid[1][0], bins = (500,500), cmap = plt.cm.jet)
plt.colorbar()
plt.title('Real Data (Valid)')
plt.tight_layout()
plt.savefig('plots/iterations/real_data_valid.png')
plt.close()

i = 0


#save file for generator/critic loss
lossfile = open('loss_net.dat','w')

#save file for Wasserteins distance wrt validations set
wlossfile = open('wloss_net.dat','w')

n_sample = int(Train_dim-1) 
n_sample_valid = lenx - Train_dim - 1
  
y_plot = sample_all_train()[0]
y_sample = np.vstack(y_plot).T

y_plot_valid = sample_all_valid()[0]
y_sample_valid = np.vstack(y_plot_valid).T
   

for it in range(max_step):
    if it % epoch_iter == 0:
        epoch_num = int(it/epoch_iter)
        Z_sample = sample_Z(n_sample, Z_dim)
        Z_sample_valid = sample_Z(n_sample_valid, Z_dim)
               
        samples_train = sess.run(G_sample, feed_dict={Z: Z_sample, y:y_sample})
        samples_valid = sess.run(G_sample, feed_dict={Z: Z_sample_valid, y:y_sample_valid})
        
        plt.figure()
        xax = plt.hist2d(y_plot[0],samples_train[:,0], bins = (500,500), cmap = plt.cm.jet)
        plt.colorbar()
        plt.title('TSamples at Iteration %d'%it)
        plt.tight_layout()
        plt.savefig('plots/iterations/Titeration_%d.png'%epoch_num)
        plt.close()
        
        plt.figure()
        xax = plt.hist2d(y_plot_valid[0],samples_valid[:,0], bins = (500,500), cmap = plt.cm.jet)
        plt.colorbar()
        plt.title('VSamples at Iteration %d'%it)
        plt.tight_layout()
        plt.savefig('plots/iterations/Viteration_%d.png'%epoch_num)
        plt.close()

        wlossfile.write(str(it))
        wlossfile.write(" ")
        wlossfile.write(str(wloss))
        wlossfile.write("\n")
        

    draw_mb = sample_data_wor(it, n=mb_size)
    X_mb = np.vstack(draw_mb[1]).T
    y_mb = np.vstack(draw_mb[0]).T

    for _ in range(D_iter):

        Z_sample = sample_Z(mb_size, Z_dim)
        _, D_loss_curr,grad_pen_curr = sess.run([D_solver, D_loss, grad_pen], feed_dict={X: X_mb, Z: Z_sample, y:y_mb})
    _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: Z_sample, y:y_mb})
    
    if it % int(epoch_iter/10) == 0:
        print('Iter: {}'.format(it),'D loss: {:.4}'.format(D_loss_curr), 'G_loss: {:.4}'.format(G_loss_curr))
        
        lossfile.write(str(it))
        lossfile.write(" ")
        lossfile.write(str(D_loss_curr))
        lossfile.write(" ")
        lossfile.write(str(G_loss_curr))
        lossfile.write("\n")
        
        
    
    if it % weight_out == 0 or i == max_step:
            checkpoint_file = os.path.join(result_dir, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=i)
            epoch_num = int(it/epoch_iter)
            
            theta_G_val = sess.run(theta_G)
            
            #saving weights for parametrization in the reduced equation
            np.savetxt('G_W1_%d.txt'%epoch_num, theta_G_val[0], delimiter=" ")
            np.savetxt('G_W2_%d.txt'%epoch_num, theta_G_val[1], delimiter=" ")
            np.savetxt('G_W3_%d.txt'%epoch_num, theta_G_val[2], delimiter=" ")
            np.savetxt('G_W4_%d.txt'%epoch_num, theta_G_val[3], delimiter=" ")
            np.savetxt('G_b1_%d.txt'%epoch_num, theta_G_val[4], delimiter=" ")
            np.savetxt('G_b2_%d.txt'%epoch_num, theta_G_val[5], delimiter=" ")
            np.savetxt('G_b3_%d.txt'%epoch_num, theta_G_val[6], delimiter=" ")
            np.savetxt('G_b4_%d.txt'%epoch_num, theta_G_val[7], delimiter=" ")
            
            #for visualization of joint histogram
            np.savetxt('G1_%d.txt'%epoch_num, samples_valid[:,0], delimiter=" ")
            np.savetxt('G2_%d.txt'%epoch_num, samples_valid[:,1], delimiter=" ")

lossfile.close()
wlossfile.close()