import tensorflow as tf
import numpy as np


class Model:

    def __init__(self, input_data):

        # Load the input activity, the target data, and the training mask for this batch of trials
        self.input_data = input_data

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()

    def run_model(self):

        x = self.input_data
        for n in range(len(par['encoder_dims'])-1):
            with tf.variable_scope('encoder' + str(n)):

                # Get layer variables
                W = tf.get_variable('W', (par['encoder_dims'][n+1], par['encoder_dims'][n]), \
                    initializer=tf.random_uniform_initializer(-0.05, 0.05))
                b = tf.get_variable('b', (par['encoder_dims'][n+1], 1), initializer=tf.constant_initializer(0))
                x = tf.nn.relu(tf.matmul(W, x) + b)


        with tf.variable_scope('latent'):
            W_mu = tf.get_variable('W', (par['latent_dims'], par['encoder_dims'][-1]), \
                initializer=tf.random_uniform_initializer(-0.05, 0.05))
            b_mu = tf.get_variable('b', (par['latent_dims'], 1), initializer=tf.constant_initializer(0))
            W_sigma = tf.get_variable('W', (par['latent_dims'], par['encoder_dims'][-1]), \
                initializer=tf.random_uniform_initializer(-0.05, 0.05))
            b_sigma = tf.get_variable('b', (par['latent_dims'], 1), initializer=tf.constant_initializer(0))


        self.latent_mu = tf.matmul(W_mu,x) + b_mu
        self.latent_sigma = tf.matmul(W_sigma,x) + b_sigma
        self.latent_loss = -0.5*tf.reduce_sum(1 + self.latent_sigma - tf.square(self.latent_mu) - tf.exp(self.latent_sigma))


        self.sample_latent = self.latent_mu + tf.exp(self.latent_sigma)*tf.random_normal([par['n_latent'], par['batch_train_size']], \
            0, 1 , dtype=tf.float32)


        self.x_hat = self.sample_latent
        for n in range(len(par['decoder_dims'])-1):
            with tf.variable_scope('decoder' + str(n)):
                # Get layer variables
                W = tf.get_variable('W', (par['decoder_dims'][n+1], par['decoder_dims'][n]), \
                    initializer=tf.random_uniform_initializer(-0.05, 0.05))
                b = tf.get_variable('b', (par['decoder_dims'][n+1], 1), initializer=tf.constant_initializer(0))
                self.x_hat = tf.nn.relu(tf.matmul(W, self.x_hat) + b)



    def optimize(self):


        self.recotruction_loss = tf.reduce_mean(tf.square(self.x_hat - self.input_data))
        self.loss = par['beta']*self.latent_loss + self.recotruction_loss

        opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        self.train_op = opt.minimize(self.loss)
