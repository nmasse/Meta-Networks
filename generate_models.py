<<<<<<< HEAD
import numpy as np
import pickle
import parameters as p
import stimulus
import matplotlib.pyplot as plt


def run_model(W_rnn, W_out, b_rnn, b_out, motion_input, rule_input):

    network_output = []

    if p.par['EI']:
        W_rnn = np.dot(W_rnn, p.par['EI_matrix'])

    hidden_state_hist = []
    output_rec = []

    h = p.par['h_init']

    for t, (x, r) in enumerate(zip(motion_input, rule_input)):

        # Calculate new and recurrent rnn inputs
        motion_input = np.dot(p.par['W_in'], np.maximum(0, x))
        #rule_input = np.dot(W_rule, np.maximum(0, r))
        rule_input = 0
        rnn_recur = np.dot(W_rnn, h)

        # Compute rnn state
        h = np.maximum(0, h*(1-p.par['alpha_neuron']) \
            + p.par['alpha_neuron']*(motion_input + rule_input + rnn_recur + b_rnn) \
            + np.random.normal(0, p.par['noise_rnn'], (p.par['n_hidden'],p.par['batch_train_size'])))

        # Compute output state
        network_output.append(np.dot(W_out, h) + b_out)

    return network_output
=======
"""
Nicolas Masse 2018
Contributions from Gregory Grant, Catherine Lee
"""

import tensorflow as tf
import numpy as np
import stimulus
import time
import parameters as p
#from parameters import *
import os, sys
import pickle
import AdamOpt
import importlib

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class Model:

    def __init__(self, input_data, target_data, mask, generator_var):

        # Load the input activity, the target data, and the training mask for this batch of trials
        self.input_data = tf.unstack(input_data, axis=1)
        self.target_data = tf.unstack(target_data, axis=1)
        self.mask = tf.unstack(mask, axis=0)
        self.generator_var = tf.unstack(generator_var, axis=0)

        # Load meta network state
        self.W_in = tf.constant(p.par['W_in'], dtype=tf.float32)
        self.W_ei = tf.constant(p.par['EI_matrix'], dtype=tf.float32)
        self.hidden_init = tf.constant(p.par['h_init'], dtype=tf.float32)
        self.w_rnn_mask  = tf.constant(p.par['w_rnn_mask'], dtype=tf.float32)

        # Load the initial synaptic depression and facilitation to be used at the start of each trial
        self.synapse_x_init = tf.constant(p.par['syn_x_init'])
        self.synapse_u_init = tf.constant(p.par['syn_u_init'])

        # Declare all necessary variables for each network
        self.declare_variables()

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()


    def declare_variables(self):

        for n in range(len(p.par['generator_dims']) - 1):
            with tf.variable_scope('generator'+str(n)):
                w = tf.random_uniform([p.par['generator_dims'][n+1],p.par['generator_dims'][n]], \
                    -0.01, 0.01, dtype=tf.float32)
                tf.get_variable('W', initializer=w, trainable=True)
        with tf.variable_scope('generator_output'):
            w = tf.random_uniform([p.par['wrnn_generator_dims'][1],p.par['wrnn_generator_dims'][0]], \
                -0.01, 0.01, dtype=tf.float32)
            tf.get_variable('W_rnn', initializer=w, trainable=True)
            w = tf.random_uniform([p.par['wout_generator_dims'][1],p.par['wout_generator_dims'][0]], \
                -0.01, 0.01, dtype=tf.float32)
            tf.get_variable('W_out', initializer=w, trainable=True)
            w = tf.random_uniform([p.par['brnn_generator_dims'][1],p.par['brnn_generator_dims'][0]], \
                -0.01, 0.01, dtype=tf.float32)
            tf.get_variable('b_rnn', initializer=w, trainable=True)
            w = tf.random_uniform([p.par['bout_generator_dims'][1],p.par['bout_generator_dims'][0]], \
                -0.01, 0.01, dtype=tf.float32)
            tf.get_variable('b_out', initializer=w, trainable=True)



    def run_model(self):

        self.networks_hidden = []
        self.networks_output = []
        self.networks_syn_x = []
        self.networks_syn_u = []

        for n in range(p.par['num_networks']):
            z = tf.reshape(self.generator_var[n],(p.par['generator_dims'][0], 1))
            for m in range(len(p.par['generator_dims']) - 1):
                with tf.variable_scope('generator'+str(m), reuse=True):
                    W = tf.get_variable('W')
                    z = tf.nn.relu(tf.matmul(W, z))

            with tf.variable_scope('generator_output', reuse=True):
                W_rnn_gen = tf.get_variable('W_rnn')
                W_out_gen = tf.get_variable('W_out')
                b_rnn_gen = tf.get_variable('b_rnn')
                b_out_gen = tf.get_variable('b_out')

                W_rnn = tf.matmul(W_rnn_gen, z)
                W_out = tf.matmul(W_out_gen, z)
                b_rnn = tf.matmul(b_rnn_gen, z)
                b_out = tf.matmul(b_out_gen, z)

                W_rnn = tf.reshape(W_rnn,(p.par['n_hidden'], p.par['n_hidden']))
                W_out = tf.reshape(W_out,(p.par['n_output'], p.par['n_hidden']))
                b_rnn = tf.reshape(b_rnn,(p.par['n_hidden'], 1))
                b_out = tf.reshape(b_out,(p.par['n_output'], 1))

            if p.par['EI']:
                W_rnn *= self.w_rnn_mask
                W_rnn = tf.matmul(tf.nn.relu(W_rnn), self.W_ei)

            hidden_state_hist = []
            syn_x_hist = []
            syn_u_hist = []
            output_rec = []

            h = self.hidden_init
            syn_x = self.synapse_x_init
            syn_u = self.synapse_u_init

            for t, x in enumerate(self.input_data):

                # Calculate effect of STP
                if p.par['synapse_config'] == 'std_stf':
                    # implement both synaptic short term facilitation and depression
                    syn_x += p.par['alpha_std']*(1-syn_x) - p.par['dt_sec']*syn_u*syn_x*h
                    syn_u += p.par['alpha_stf']*(p.par['U']-syn_u) + p.par['dt_sec']*p.par['U']*(1-syn_u)*h
                    syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
                    syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
                    h_post = syn_u*syn_x*h

                elif p.par['synapse_config'] == 'std':
                    # implement synaptic short term derpression, but no facilitation
                    # we assume that syn_u remains constant at 1
                    syn_x += p.par['alpha_std']*(1-syn_x) - p.par['dt_sec']*syn_x*h
                    syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
                    syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
                    h_post = syn_x*h

                elif p.par['synapse_config'] == 'stf':
                    # implement synaptic short term facilitation, but no depression
                    # we assume that syn_x remains constant at 1
                    syn_u += p.par['alpha_stf']*(p.par['U']-syn_u) + p.par['dt_sec']*p.par['U']*(1-syn_u)*h
                    syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
                    h_post = syn_u*h

                else:
                    # no synaptic plasticity
                    h_post = h

                # Calculate new and recurrent rnn inputs
                rnn_input = tf.matmul(tf.nn.relu(self.W_in), tf.nn.relu(x))
                rnn_recur = tf.matmul(W_rnn, h_post)

                # Compute rnn state
                h = tf.nn.relu(h*(1-p.par['alpha_neuron']) \
                    + p.par['alpha_neuron']*(rnn_input + rnn_recur + b_rnn) \
                    + tf.random_normal([p.par['n_hidden'],p.par['batch_train_size']], 0, p.par['noise_rnn'], dtype=tf.float32))

                # Compute output state
                output = tf.matmul(tf.nn.relu(W_out), h) + b_out

                # Record the outputs of this time step
                hidden_state_hist.append(h)
                syn_x_hist.append(syn_x)
                syn_u_hist.append(syn_u)
                output_rec.append(output)

            self.networks_hidden.append(hidden_state_hist)
            self.networks_output.append(output_rec)
            #self.networks_syn_x.append(syn_x_hist)
            #self.networks_syn_u.append(syn_u_hist)


    def optimize(self):

        self.perf_losses = []
        self.spike_losses = []
        self.wiring_losses = []
        self.total_loss = tf.constant(0.)

        self.variables = [var for var in tf.trainable_variables() if not var.op.name.find('conv')==0]
        #adam_optimizer = AdamOpt.AdamOpt(self.variables, learning_rate = p.par['learning_rate'])

        adam_optimizer = tf.train.AdamOptimizer(learning_rate = p.par['learning_rate'])

        for n in range(p.par['num_networks']):

            # Calculate performance loss
            perf_loss = [mask*tf.nn.softmax_cross_entropy_with_logits(logits = y_hat, labels = desired_output, dim=0) \
                         for (y_hat, desired_output, mask) in zip(self.networks_output[n], self.target_data, self.mask)]
            perf_loss = tf.reduce_mean(tf.stack(perf_loss, axis=0))

            # Calculate spiking loss
            spike_loss = [p.par['spike_cost']*tf.reduce_mean(tf.square(h), axis=0) for h in self.networks_hidden[n]]
            spike_loss = tf.reduce_mean(tf.stack(spike_loss, axis=0))

            # Calculate wiring cost
            #wiring_loss = [p.par['wiring_cost']*tf.nn.relu(W_rnn*p.par['W_rnn_dist']) for W_rnn in tf.trainable_variables() if 'W_rnn' in W_rnn.name]
            #wiring_loss = tf.reduce_mean(tf.stack(wiring_loss, axis=0))
            wiring_loss = 0.

            # Add losses to record
            self.perf_losses.append(perf_loss)
            self.spike_losses.append(spike_loss)
            self.wiring_losses.append(wiring_loss)

            # Collect total loss
            self.total_loss += perf_loss + spike_loss + wiring_loss


        #self.train_op = adam_optimizer.compute_gradients(self.total_loss)
        self.train_op = adam_optimizer.minimize(self.total_loss)
        #self.reset_adam_op = adam_optimizer.reset_params()


def main(gpu_id = None):
    gpu_id = None
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
    model_performance = {'accuracy': [], 'par': [], 'task_list': []}
    stim = stimulus.Stimulus()

    mask = tf.placeholder(tf.float32, shape=[p.par['num_time_steps'], p.par['batch_train_size']])
    x = tf.placeholder(tf.float32, shape=[p.par['n_input'], p.par['num_time_steps'], p.par['batch_train_size']])
    y = tf.placeholder(tf.float32, shape=[p.par['n_output'], p.par['num_time_steps'], p.par['batch_train_size']])
    z = tf.placeholder(tf.float32, shape=[p.par['num_networks'], p.par['generator_dims'][0]])

    """ Start TensorFlow session """
    with tf.Session() as sess:
        if gpu_id is None:
            model = Model(x, y, mask, z)
        else:
            with tf.device("/gpu:0"):
                model = Model(x, y, mask, z)

        # Initialize session variables
        init = tf.global_variables_initializer()
        sess.run(init)
        t_start = time.time()

        # Restore variables from previous model if desired
        saver = tf.train.Saver()
        if p.par['load_previous_model']:
            saver.restore(sess, p.par['save_dir'] + p.par['ckpt_load_fn'])
            print('Model ' +  p.par['ckpt_load_fn'] + ' restored.')

        results = create_results_dict()

        for k in range(p.par['num_network_iters']):
            print('NETWORK ITERATION ', k)
            for i in range(p.par['num_iterations']):

                # generate batch of batch_train_size
                trial_info = stim.generate_trial()
                generator_var = np.random.normal(0,1,size=(p.par['num_networks'], p.par['generator_dims'][0]))

                # get the intiial weights
                if i == -1:
                    W_rnn0, W_out0 = get_initial_weights()

                """
                Run the model
                """
                _, total_loss, perf_loss, spike_loss, network_output = sess.run([model.train_op, model.total_loss, model.perf_losses, \
                    model.spike_losses, model.networks_output], {x: trial_info['neural_input'], \
                    y: trial_info['desired_output'], mask: trial_info['train_mask'], z: generator_var})

                if (i+1)%p.par['iters_between_outputs'] == 0:# and i != 0:
                    accuracy = np.array([get_perf(trial_info['desired_output'], y_hat, trial_info['train_mask']) for y_hat in network_output])
                    iteration_time = time.time() - t_start
                    iterstr = 'Iter. {:>4}'.format(i)
                    timestr = 'Time. {:>7.4}'.format(iteration_time)
                    lossstr = 'Total Loss: {:>7.4}'.format(total_loss)
                    #perfstr = 'Perf. Loss: {:>7.4} +/- {:<7.4}'.format(np.mean(perf_loss), np.std(perf_loss))
                    #spikstr = 'Spike Loss: {:>7.4} +/- {:<7.4}'.format(np.mean(spike_loss), np.std(spike_loss))
                    #wirestr = 'Wiring Loss: {:>7.4} +/- {:<7.4}'.format(np.mean(wiring_loss), np.std(wiring_loss))
                    perfstr = 'Perf. Loss: {:>7.4}'.format(np.mean(perf_loss))
                    spikstr = 'Spike Loss: {:>7.4}'.format(np.mean(spike_loss))
                    #wirestr = 'Wiring Loss: {:>7.4}'.format(np.mean(wiring_loss))
                    accuracystr = 'Accuracy: {:>7.4} +/- {:<7.4}'.format(np.mean(accuracy), np.std(accuracy))

                    print(' | '.join([str(x) for x in [iterstr, timestr, perfstr, spikstr, accuracystr]]))

            print('Saving data and reseting variables...')
            save_data(results, k, network_output, trial_info, W_rnn0, W_out0)
            #sess.run(model.reset_adam_op)
            init = tf.global_variables_initializer()
            sess.run(init)



def get_initial_weights():

    W_rnn0 = np.zeros((p.par['num_networks'], p.par['n_hidden']*p.par['n_hidden']), dtype = np.float32)
    W_out0 = np.zeros((p.par['num_networks'], p.par['n_hidden']*p.par['n_output']), dtype = np.float32)
    for n in range(p.par['num_networks']):
        with tf.variable_scope('network'+str(n), reuse=True):
            w = tf.get_variable('W_rnn')
            w = w.eval()
            W_rnn0[n,:] = np.reshape(w,(p.par['n_hidden']*p.par['n_hidden']))
            w = tf.get_variable('W_out')
            w = w.eval()
            W_out0[n,:] = np.reshape(w,(p.par['n_hidden']*p.par['n_output']))

    return W_rnn0, W_out0

def save_data(results, k, network_output, trial_info, W_rnn0, W_out0):

    # save data
    ind = range(k*p.par['num_networks'], (k+1)*p.par['num_networks'])
    accuracy = [get_perf(trial_info['desired_output'], y_hat, trial_info['train_mask']) for y_hat in network_output]
    W_rnn, W_out, b_rnn, b_out = eval_weights()
    results['W_rnn'][ind, :] = W_rnn
    results['W_out'][ind, :] = W_out
    results['W_rnn0'][ind, :] = W_rnn0
    results['W_out0'][ind, :] = W_out0
    results['b_rnn'][ind, :] = b_rnn
    results['b_out'][ind, :] = b_out
    results['accuracy'][ind] = np.array(accuracy)
    pickle.dump(results, open(p.par['save_dir'] + p.par['save_fn'], 'wb') )


def create_results_dict():

    results = {
        'W_rnn'     : np.zeros((p.par['num_networks']*p.par['num_network_iters'], p.par['n_hidden']*p.par['n_hidden']), dtype = np.float32),
        'W_out'     : np.zeros((p.par['num_networks']*p.par['num_network_iters'], p.par['n_hidden']*p.par['n_output']), dtype = np.float32),
        'W_rnn0'     : np.zeros((p.par['num_networks']*p.par['num_network_iters'], p.par['n_hidden']*p.par['n_hidden']), dtype = np.float32),
        'W_out0'     : np.zeros((p.par['num_networks']*p.par['num_network_iters'], p.par['n_hidden']*p.par['n_output']), dtype = np.float32),
        'b_rnn'     : np.zeros((p.par['num_networks']*p.par['num_network_iters'], p.par['n_hidden']), dtype = np.float32),
        'b_out'     : np.zeros((p.par['num_networks']*p.par['num_network_iters'], p.par['n_output']), dtype = np.float32),
        'accuracy'  : np.zeros((p.par['num_networks']*p.par['num_network_iters']), dtype = np.float32)}

    return results

def eval_weights():

    W_rnn = np.zeros((p.par['num_networks'], p.par['n_hidden']*p.par['n_hidden']), dtype = np.float32)
    W_out = np.zeros((p.par['num_networks'], p.par['n_hidden']*p.par['n_output']), dtype = np.float32)
    b_rnn = np.zeros((p.par['num_networks'], p.par['n_hidden']), dtype = np.float32)
    b_out = np.zeros((p.par['num_networks'], p.par['n_output']), dtype = np.float32)

    for n in range(p.par['num_networks']):
        with tf.variable_scope('network'+str(n), reuse=True):
            x = tf.get_variable('W_rnn')
            W_rnn[n, :] = np.reshape(x.eval(), (1, p.par['n_hidden']*p.par['n_hidden']))
            x = tf.get_variable('W_out')
            W_out[n, :] = np.reshape(x.eval(), (1, p.par['n_hidden']*p.par['n_output']))
            x = tf.get_variable('b_rnn')
            b_rnn[n, :] = np.reshape(x.eval(), (1, p.par['n_hidden']))
            x = tf.get_variable('b_out')
            b_out[n, :] = np.reshape(x.eval(), (1, p.par['n_output']))

    return W_rnn, W_out, b_rnn, b_out
>>>>>>> 27a40217286fd3968ffa11d23fb7e24832567a6e

def get_perf(y, y_hat, mask):

    """
    Calculate task accuracy by comparing the actual network output to the desired output
    only examine time points when test stimulus is on
    in another words, when y[0,:,:] is not 0
    y is the desired output
    y_hat is the actual output
    """
    y_hat = np.stack(y_hat, axis=1)
<<<<<<< HEAD
    #plt.imshow(y_hat[0,:,:], aspect='auto', interpolation='none')
    #plt.show()
    mask *= y[0,:,:]==0
    y = np.argmax(y, axis = 0)
    y_hat = np.argmax(y_hat, axis = 0)
    #print(np.sum(mask))
    accuracy = np.sum(np.float32(y == y_hat)*np.squeeze(mask))/np.sum(mask)

    return accuracy

N = 1000
x = pickle.load(open(p.par['save_dir'] + 'autoencoder_results_11.pkl', 'rb') )
n_latent = x['W_deocder'][0].shape[1]
stim = stimulus.Stimulus()
trial_info = stim.generate_trial()
motion_input = trial_info['neural_input'][:p.par['num_motion_tuned'], :, :]
rule_input = trial_info['neural_input'][p.par['num_motion_tuned']:, :, :]

motion_input = np.squeeze(np.split(motion_input,p.par['num_time_steps'],axis=1))
rule_input = np.split(rule_input,p.par['num_time_steps'],axis=1)


n0 = p.par['n_hidden']**2
n1 = p.par['n_hidden']**2+p.par['n_hidden']*p.par['n_output']
n2 = p.par['n_hidden']**2+p.par['n_hidden']*p.par['n_output']+p.par['n_hidden']

print(x['latent_mu'].shape)
mu_mean = np.mean(x['latent_mu'], axis=1)
print(mu_mean)

print(x.keys())

for i in range(N):

    #z = np.random.normal(np.reshape(mu_mean,(n_latent,1)),0.1*np.ones((n_latent,1)),size = (n_latent, 1))
    z = np.random.normal(0,1,size = (n_latent, 1))
    for n in range(len(x['W_deocder'])):
        if n < len(x['W_deocder']) - 1:
            z = np.maximum(0, np.dot(x['W_deocder'][n], z) + x['b_decoder'][n])
        else:
            z = np.dot(x['W_deocder'][n], z) + x['b_decoder'][n]
    W_rnn = np.maximum(0, np.reshape(z[:n0], (p.par['n_hidden'], p.par['n_hidden'])))
    W_rnn *= p.par['w_rnn_mask']
    W_out = np.maximum(0, np.reshape(z[n0:n1], (p.par['n_output'], p.par['n_hidden'])))
    W_out *= p.par['w_out_mask']
    b_rnn = np.reshape(z[n1:n2], (p.par['n_hidden'], 1))
    b_out = np.reshape(z[n2:], (p.par['n_output'], 1))

    network_output = run_model(W_rnn, W_out, b_rnn, b_out, motion_input, rule_input)
    accuracy = get_perf(trial_info['desired_output'], network_output, trial_info['train_mask'])
    print(i, accuracy)
=======
    mask *= y[0,:,:]==0
    mask_non_match = mask*(y[1,:,:]==1)
    mask_match = mask*(y[2,:,:]==1)
    y = np.argmax(y, axis = 0)
    y_hat = np.argmax(y_hat, axis = 0)
    accuracy = np.sum(np.float32(y == y_hat)*np.squeeze(mask))/np.sum(mask)

    #accuracy_non_match = np.sum(np.float32(y == y_hat)*np.squeeze(mask_non_match))/np.sum(mask_non_match)
    #accuracy_match = np.sum(np.float32(y == y_hat)*np.squeeze(mask_match))/np.sum(mask_match)

    return accuracy


if __name__ == '__main__':
    main(str(sys.argv[1]))
>>>>>>> 27a40217286fd3968ffa11d23fb7e24832567a6e
