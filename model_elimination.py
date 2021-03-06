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
import analysis

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class Model:

    def __init__(self, input_data, target_data, mask, gate):

        # Load the input activity, the target data, and the training mask for this batch of trials
        self.input_data = tf.unstack(input_data, axis=1)
        self.target_data = tf.unstack(target_data, axis=1)
        self.mask = tf.unstack(mask, axis=0)
        self.gate = tf.tile(gate,[1, p.par['batch_train_size']])

        # Load meta network state
        self.W_ei = tf.constant(p.par['EI_matrix'], dtype=tf.float32)
        self.hidden_init = tf.constant(p.par['h_init'], dtype=tf.float32)

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
        with tf.variable_scope('network'):
            tf.get_variable('W_rnn', initializer=p.par['w_rnn0'], trainable=True)
            tf.get_variable('W_out', initializer=p.par['w_out0'], trainable=True)
            tf.get_variable('W_in', initializer=p.par['w_in0'], trainable=True)
            tf.get_variable('b_rnn', shape=[p.par['n_hidden'], 1], initializer=tf.random_uniform_initializer(-1e-6,1e-6), trainable=True)
            tf.get_variable('b_out', shape=[p.par['n_output'], 1], initializer=tf.random_uniform_initializer(-1e-6,1e-6), trainable=True)


    def run_model(self):

        self.network_hidden = []
        self.network_output = []
        self.network_syn_x = []
        self.network_syn_u = []

        with tf.variable_scope('network', reuse=True):
            W_rnn = tf.get_variable('W_rnn')
            W_in = tf.get_variable('W_in')
            W_out = tf.get_variable('W_out')
            b_rnn = tf.get_variable('b_rnn')
            b_out = tf.get_variable('b_out')

        if p.par['EI']:
            W_rnn = tf.matmul(tf.nn.relu(W_rnn), self.W_ei)

        h = self.hidden_init
        syn_x = self.synapse_x_init
        syn_u = self.synapse_u_init

        for x in self.input_data:

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
            rnn_input = tf.matmul(tf.nn.relu(W_in), tf.nn.relu(x))
            rnn_recur = tf.matmul(W_rnn, h_post)

            # Compute rnn state
            h = tf.nn.relu(h*(1-p.par['alpha_neuron']) \
                + p.par['alpha_neuron']*(rnn_input + rnn_recur + b_rnn) \
                + tf.random_normal([p.par['n_hidden'],p.par['batch_train_size']], 0, p.par['noise_rnn'], dtype=tf.float32))

            h *= self.gate
            # Compute output state
            output = tf.matmul(tf.nn.relu(W_out), h) + b_out

            # Record the outputs of this time step
            self.network_hidden.append(h)
            self.network_output.append(output)
            self.network_syn_x.append(syn_x)
            self.network_syn_u.append(syn_u)


    def optimize(self):


        self.variables = [var for var in tf.trainable_variables() if not var.op.name.find('conv')==0]
        adam_optimizer = AdamOpt.AdamOpt(self.variables, learning_rate = p.par['learning_rate'])


        #print('mask', self.mask)
        #print('target_data', self.target_data)
        #print('network_output', self.network_output)
        # Calculate performance loss
        perf_loss = tf.stack([mask*tf.nn.softmax_cross_entropy_with_logits(logits = y_hat, labels = desired_output, dim=0) \
                        for (y_hat, desired_output, mask) in zip(self.network_output, self.target_data, self.mask)])
        self.perf_loss = tf.reduce_mean(perf_loss)

        # Calculate spiking loss
        self.spike_loss = [p.par['spike_cost']*tf.reduce_mean(tf.square(h), axis=0) for h in self.network_hidden]
        self.spike_loss = tf.reduce_mean(self.spike_loss)/tf.reduce_mean(self.gate)

        # Calculate wiring cost
        self.wiring_loss = [p.par['wiring_cost']*tf.nn.relu(W_rnn) \
            for W_rnn in tf.trainable_variables() if 'W_rnn' in W_rnn.name]
        self.wiring_loss = tf.reduce_mean(self.wiring_loss)

        # Collect total loss
        self.total_loss = self.perf_loss + self.spike_loss + self.wiring_loss

        self.train_op = adam_optimizer.compute_gradients(self.total_loss)
        self.reset_adam_op = adam_optimizer.reset_params()


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
    model_performance = {'accuracy': [], 'par': [], 'task_list': []}
    stim = stimulus.Stimulus()

    mask = tf.placeholder(tf.float32, shape=[p.par['num_time_steps'], p.par['batch_train_size']])
    x = tf.placeholder(tf.float32, shape=[p.par['n_input'], p.par['num_time_steps'], p.par['batch_train_size']])
    y = tf.placeholder(tf.float32, shape=[p.par['n_output'], p.par['num_time_steps'], p.par['batch_train_size']])
    gate = tf.placeholder(tf.float32, shape=[p.par['n_hidden'], 1])


    """ Start TensorFlow session """
    with tf.Session(config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
        if gpu_id is None:
            model = Model(x, y, mask, gate)
        else:
            with tf.device("/gpu:0"):
                model = Model(x, y, mask, gate)

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
        current_gate = np.ones((p.par['n_hidden'],1), dtype=np.float32)

        # history of info to save
        results = {'gate_hist': [], 'accuracy_hist': [], 'weights_hist': []}

        for k in range(p.par['n_hidden']-5):
            print('NETWORK ITERATION ', k)
            if k == 0:
                add_iter = 500
            else:
                add_iter = 0
            for i in range(p.par['num_iterations'] + add_iter):

                # generate batch of batch_train_size
                trial_info = stim.generate_trial()

                """
                Run the model
                """
                _, total_loss, perf_loss, spike_loss, wiring_loss, network_output = sess.run([model.train_op, \
                    model.total_loss, model.perf_loss, \
                    model.spike_loss, model.wiring_loss, model.network_output], {x:trial_info['neural_input'], \
                    y: trial_info['desired_output'], mask: trial_info['train_mask'], gate: current_gate})


                if (i+1)%p.par['iters_between_outputs'] == 0:# and i != 0:
                    accuracy = get_perf(trial_info['desired_output'], network_output, trial_info['train_mask'])
                    iteration_time = time.time() - t_start
                    iterstr = 'Iter. {:>4}'.format(i)
                    timestr = 'Time. {:>7.4}'.format(iteration_time)
                    lossstr = 'Total Loss: {:>7.4}'.format(total_loss)
                    perfstr = 'Perf. Loss: {:>7.4}'.format(np.mean(perf_loss))
                    spikstr = 'Spike Loss: {:>7.4}'.format(np.mean(spike_loss))
                    wirestr = 'Wiring Loss: {:>7.4}'.format(np.mean(wiring_loss))
                    accuracystr = 'Accuracy: {:>7.4} +/- {:<7.4}'.format(np.mean(accuracy), np.std(accuracy))
                    print(' | '.join([str(x) for x in [iterstr, timestr, perfstr, spikstr, wirestr, accuracystr]]))

            print('Saving data...')
            save_data(results, accuracy, current_gate)
            weights = eval_weights()

            print('Selecting next unit to gate')
            acc = np.zeros((p.par['n_hidden']))
            for m in range(p.par['n_hidden']):
                if current_gate[m] == 0:
                    continue
                temp_gate = np.array(current_gate)
                temp_gate[m,0] = 0
                y_hat, _, _, _ = analysis.run_model(trial_info['neural_input'], p.par['h_init'], p.par['syn_x_init'], \
                    p.par['syn_u_init'], weights, temp_gate)
                acc[m] = get_perf(trial_info['desired_output'], y_hat, trial_info['train_mask'])
            print('gating accuracy ', acc)
            next_to_gate = np.argmax(acc)
            print('next_to_gate ', next_to_gate, ' acc =', acc[next_to_gate])
            current_gate[next_to_gate,0] = 0
            print('Current gate ',current_gate[:,0])



def get_initial_weights():

    W_rnn0 = np.zeros((p.par['num_networks'], p.par['n_hidden']*p.par['n_hidden']), dtype = np.float32)
    W_out0 = np.zeros((p.par['num_networks'], p.par['n_hidden']*p.par['n_output']), dtype = np.float32)
    W_rule0 = np.zeros((p.par['num_networks'], p.par['n_hidden']*p.par['num_rule_tuned']), dtype = np.float32)
    for n in range(p.par['num_networks']):
        with tf.variable_scope('network'+str(n), reuse=True):
            w = tf.get_variable('W_rnn')
            w = w.eval()
            W_rnn0[n,:] = np.reshape(w,(p.par['n_hidden']*p.par['n_hidden']))
            w = tf.get_variable('W_out')
            w = w.eval()
            W_out0[n,:] = np.reshape(w,(p.par['n_hidden']*p.par['n_output']))
            w = tf.get_variable('W_rule')
            w = w.eval()
            W_rule0[n,:] = np.reshape(w,(p.par['n_hidden']*p.par['num_rule_tuned']))

    return W_rnn0, W_out0, W_rule0

def save_data(results, accuracy, current_gate):

    # save data
    weights = eval_weights()
    results['weights_hist'].append(weights)
    results['gate_hist'].append(np.array(current_gate))
    results['accuracy_hist'].append(accuracy)
    results['par'] = p.par
    pickle.dump(results, open(p.par['save_dir'] + p.par['save_fn'], 'wb') )

def create_results_dict():

    results = {
        'W_rnn'     : np.zeros((p.par['num_networks']*p.par['num_network_iters'], p.par['n_hidden']*p.par['n_hidden']), dtype = np.float32),
        'W_out'     : np.zeros((p.par['num_networks']*p.par['num_network_iters'], p.par['n_hidden']*p.par['n_output']), dtype = np.float32),
        'W_rule'     : np.zeros((p.par['num_networks']*p.par['num_network_iters'], p.par['n_hidden']*p.par['num_rule_tuned']), dtype = np.float32),
        'W_rnn0'     : np.zeros((p.par['num_networks']*p.par['num_network_iters'], p.par['n_hidden']*p.par['n_hidden']), dtype = np.float32),
        'W_out0'     : np.zeros((p.par['num_networks']*p.par['num_network_iters'], p.par['n_hidden']*p.par['n_output']), dtype = np.float32),
        'W_rule0'     : np.zeros((p.par['num_networks']*p.par['num_network_iters'], p.par['n_hidden']*p.par['num_rule_tuned']), dtype = np.float32),
        'b_rnn'     : np.zeros((p.par['num_networks']*p.par['num_network_iters'], p.par['n_hidden']), dtype = np.float32),
        'b_out'     : np.zeros((p.par['num_networks']*p.par['num_network_iters'], p.par['n_output']), dtype = np.float32),
        'accuracy'  : np.zeros((p.par['num_networks']*p.par['num_network_iters']), dtype = np.float32)}

    return results

def eval_weights():

    weights = {}
    with tf.variable_scope('network', reuse=True):
        weights['w_rnn'] = tf.get_variable('W_rnn').eval()
        weights['w_in'] = tf.get_variable('W_in').eval()
        weights['w_out'] = tf.get_variable('W_out').eval()
        weights['b_rnn'] = tf.get_variable('b_rnn').eval()
        weights['b_out'] = tf.get_variable('b_out').eval()

    return weights

def get_perf(y, y_hat, mask):

    """
    Calculate task accuracy by comparing the actual network output to the desired output
    only examine time points when test stimulus is on
    in another words, when y[0,:,:] is not 0
    y is the desired output
    y_hat is the actual output
    """
    y_hat = np.stack(y_hat, axis=1)
    mask *= y[0,:,:]==0
    mask_non_match = mask*(y[1,:,:]==1)
    mask_match = mask*(y[2,:,:]==1)
    y = np.argmax(y, axis = 0)
    y_hat = np.argmax(y_hat, axis = 0)
    accuracy = np.sum(np.float32(y == y_hat)*np.squeeze(mask))/np.sum(mask)

    #accuracy_non_match = np.sum(np.float32(y == y_hat)*np.squeeze(mask_non_match))/np.sum(mask_non_match)
    #accuracy_match = np.sum(np.float32(y == y_hat)*np.squeeze(mask_match))/np.sum(mask_match)

    return accuracy
