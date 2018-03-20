import tensorflow as tf
import numpy as np
from parameters import *
import os
import pickle
import sys
import time
import matplotlib.pyplot as plt


class Model:

    def __init__(self, input_data, target_accuracy):

        # Load the input activity, the target data, and the training mask for this batch of trials
        self.input_data = input_data
        self.target_accuracy = target_accuracy

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
            print(par['encoder_dims'])
            W_mu = tf.get_variable('W_mu', (par['n_latent'], par['encoder_dims'][-1]), \
                initializer=tf.random_uniform_initializer(-0.05, 0.05))
            b_mu = tf.get_variable('b_mu', (par['n_latent'], 1), initializer=tf.constant_initializer(0))
            W_sigma = tf.get_variable('W_sigma', (par['n_latent'], par['encoder_dims'][-1]), \
                initializer=tf.random_uniform_initializer(-0.05, 0.05))
            b_sigma = tf.get_variable('b_sigma', (par['n_latent'], 1), initializer=tf.constant_initializer(0))


        self.latent_mu = tf.matmul(W_mu,x) + b_mu
        self.latent_sigma = tf.matmul(W_sigma,x) + b_sigma
        self.latent_loss = -0.5*tf.reduce_mean(1 + self.latent_sigma - tf.square(self.latent_mu) - tf.exp(self.latent_sigma))


        self.sample_latent = self.latent_mu + tf.exp(self.latent_sigma)*tf.random_normal([par['n_latent'], par['batch_train_size']], \
            0, 1 , dtype=tf.float32)

        # Reconstructing network paramaters from latent variables
        self.x_hat = self.sample_latent
        for n in range(len(par['decoder_dims'])-1):
            with tf.variable_scope('decoder' + str(n)):
                # Get layer variables
                W = tf.get_variable('W', (par['decoder_dims'][n+1], par['decoder_dims'][n]), \
                    initializer=tf.random_uniform_initializer(-0.05, 0.05))
                b = tf.get_variable('b', (par['decoder_dims'][n+1], 1), initializer=tf.constant_initializer(0))
                self.x_hat = tf.nn.relu(tf.matmul(W, self.x_hat) + b)

        # Predicting network accuracy from latent variables
        self.pred_acc = self.sample_latent
        for n in range(len(par['accuracy_dims'])-1):
            with tf.variable_scope('accuracy' + str(n)):
                W = tf.get_variable('W', (par['accuracy_dims'][n+1], par['accuracy_dims'][n]), \
                    initializer=tf.random_uniform_initializer(-0.05, 0.05))
                b = tf.get_variable('b', (par['accuracy_dims'][n+1], 1), initializer=tf.constant_initializer(0))
                self.pred_acc = tf.nn.relu(tf.matmul(W, self.pred_acc) + b)



    def optimize(self):


        self.recotruction_loss = tf.reduce_mean(tf.square(self.x_hat - self.input_data))
        self.accuracy_loss = tf.reduce_mean(tf.square(self.pred_acc - self.target_accuracy))
        self.loss = par['beta']*self.latent_loss + self.recotruction_loss + par['accuracy_cost']*self.accuracy_loss

        opt = tf.train.AdamOptimizer(learning_rate = par['learning_rate'])
        self.train_op = opt.minimize(self.loss)


def main(gpu_id = None):

    print('\nRunning model.\n')

    ##################
    ### Setting Up ###
    ##################
    tf.reset_default_graph()

    """ Set up GPU """
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    """ Reset TensorFlow before running anything """
    tf.reset_default_graph()

    """ Set up performance recording """
    model_performance = {'recotruction_loss': [], 'latent_loss': [], 'accuracy_loss': [], 'loss': []}
    weights, accuracy  =  load_network_weights(par['save_dir'], par['file_prefix'], par['accuracy_threshold'])
    accuracy = np.array(accuracy)
    print('Weight shape ', weights.shape)
    print('Weight variance = ', np.mean(np.var(weights,axis=0)))

    x = tf.placeholder(tf.float32, shape=[par['num_weights'], par['batch_train_size']]) # network paramaters
    y = tf.placeholder(tf.float32, shape=[par['batch_train_size']]) # network accuracies


    """ Start TensorFlow session """
    with tf.Session() as sess:
        if gpu_id is None:
            model = Model(x, y)
        else:
            with tf.device("/gpu:0"):
                model = Model(x, y)

        # Initialize session variables
        init = tf.global_variables_initializer()
        sess.run(init)
        t_start = time.time()

        for i in range(par['num_iterations']):

            # generate batch of batch_train_size
            network_params, accuracies = generate_batch(weights, accuracy)

            """
            Run the model
            """
            _, recotruction_loss, latent_loss, accuracy_loss, loss, latent_mu = sess.run([model.train_op, model.recotruction_loss, \
                model.latent_loss, model.accuracy_loss, model.loss, model.latent_mu], {x: network_params, y: accuracies})
            model_performance['recotruction_loss'].append(recotruction_loss)
            model_performance['latent_loss'].append(latent_loss)
            model_performance['accuracy_loss'].append(accuracy_loss)
            model_performance['loss'].append(loss)

            if (i)%500==0:
                iteration_time = time.time() - t_start
                iterstr = 'Iter. {:>4}'.format(i)
                timestr = 'Time. {:>7.4}'.format(iteration_time)
                lossstr = 'Total Loss: {:>7.4}'.format(loss)
                reconstr = 'Perf. Loss: {:>7.4}'.format(np.mean(recotruction_loss))
                latentstr = 'Latent Loss: {:>7.4}'.format(np.mean(latent_loss))
                accstr = 'Accuracy Loss: {:>7.4}'.format(np.mean(accuracy_loss))
                print(' | '.join([str(x) for x in [iterstr, timestr, lossstr, reconstr, latentstr, accstr]]))

        W_deocder, b_decoder, W_accuracy, b_accuracy = eval_weights()
        z = output_prototypes(W_deocder, b_decoder)
        results = {'W_deocder': W_deocder, 'b_decoder': b_decoder, 'W_accuracy': W_accuracy, 'b_accuracy': b_accuracy, \
            'z': z, 'model_performance': model_performance, 'latent_mu': latent_mu}
        pickle.dump(results, open(par['save_dir'] + 'autoencoder_results.pkl', 'wb') )
        for j in range(len(z)):
            plt.imshow(z[j], aspect = 'auto', interpolation = 'none')
            plt.colorbar()
            plt.show()



def generate_batch(weights, accuracy):

    N = weights.shape[1]
    ind = np.random.permutation(N)
    ind = ind[: par['batch_train_size']]
    return weights[:, ind], accuracy[ind]


def load_network_weights(data_dir, file_prefix, accuracy_threshold):

    weights = []
    accuracy = []

    fns = os.listdir(data_dir)
    for f in fns:
        if f.startswith(file_prefix):
            x = pickle.load(open(data_dir + f, 'rb'))
            for n in range(len(x['accuracy'])):
                if x['accuracy'][n] > accuracy_threshold:
                    accuracy.append(x['accuracy'][n])
                    w0 = np.maximum(0,x['W_rnn'][n, :])
                    w1 = np.maximum(0,x['W_out'][n, :])
                    weights.append(np.hstack((w0, w1)))
    weights = np.transpose(np.stack(weights, axis=0))

    return weights, accuracy

def output_prototypes(W_deocder, b_decoder):

    z = []
    for i in range(par['n_latent']):
        x = np.zeros((par['n_latent'],1))
        x[i] = 1
        for j in range(len(W_deocder)):
            x = np.maximum(0, np.dot(W_deocder[j], x) + b_decoder[j])
        #x -= b_decoder[-1]
        z.append(np.reshape(x[:50**2], (50,50)))
    return z

def eval_weights():

    W_deocder = []
    b_decoder = []
    W_accuracy = []
    b_accuracy = []

    for n in range(len(par['decoder_dims'])-1):
        with tf.variable_scope('decoder' + str(n), reuse = True):
            # Get layer variables
            W = tf.get_variable('W')
            b = tf.get_variable('b')
            W_deocder.append(W.eval())
            b_decoder.append(b.eval())

    for n in range(len(par['accuracy_dims'])-1):
        with tf.variable_scope('accuracy' + str(n), reuse = True):
            W = tf.get_variable('W')
            b = tf.get_variable('b')
            W_accuracy.append(W.eval())
            b_accuracy.append(b.eval())

    return W_deocder, b_decoder, W_accuracy, b_accuracy

try:

    main(sys.argv[1])
except KeyboardInterrupt:
    quit('Quit by KeyboardInterrupt')
