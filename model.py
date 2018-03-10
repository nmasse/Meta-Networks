"""
Nicolas Masse 2018
Contributions from Gregory Grant, Catherine Lee
"""

import tensorflow as tf
import numpy as np
import stimulus
import time
from parameters import *
import os, sys
import pickle
import AdamOpt

# Ignore "use compiled version of TensorFlow" errors
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

class Model:

    def __init__(self, input_data, target_data, mask):

        # Load the input activity, the target data, and the training mask for this batch of trials
        self.input_data = tf.unstack(input_data, axis=1)
        self.target_data = tf.unstack(target_data, axis=1)
        self.mask = tf.unstack(mask, axis=0)

        # Load meta network state
        self.W_in = tf.constant(par['W_in'], dtype=tf.float32)
        self.W_ei = tf.constant(par['EI_matrix'], dtype=tf.float32)
        self.hidden_init = tf.constant(par['h_init'], dtype=tf.float32)

        # Load the initial synaptic depression and facilitation to be used at the start of each trial
        self.synapse_x_init = tf.constant(par['syn_x_init'])
        self.synapse_u_init = tf.constant(par['syn_u_init'])

        # Declare all necessary variables for each network
        self.declare_variables()

        # Build the TensorFlow graph
        self.run_model()

        # Train the model
        self.optimize()


    def declare_variables(self):
        for n in range(par['num_networks']):
            with tf.variable_scope('network'+str(n)):
                tf.get_variable('W_rnn', initializer=par['w_rnn0'][n], trainable=True)
                tf.get_variable('W_out', initializer=par['w_out0'][n], trainable=True)
                tf.get_variable('b_rnn', shape=[par['n_hidden'], 1], initializer=tf.random_uniform_initializer(-1e-6,1e-6), trainable=True)
                tf.get_variable('b_out', shape=[par['n_output'], 1], initializer=tf.random_uniform_initializer(-1e-6,0.001), trainable=True)


    def run_model(self):

        self.networks_hidden = []
        self.networks_output = []
        self.networks_syn_x = []
        self.networks_syn_u = []

        for n in range(par['num_networks']):
            with tf.variable_scope('network'+str(n), reuse=True):
                W_rnn = tf.get_variable('W_rnn')
                W_out = tf.get_variable('W_out')
                b_rnn = tf.get_variable('b_rnn')
                b_out = tf.get_variable('b_out')

            if par['EI']:
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
                if par['synapse_config'] == 'std_stf':
                    # implement both synaptic short term facilitation and depression
                    syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_u*syn_x*h
                    syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
                    syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
                    syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
                    h_post = syn_u*syn_x*h

                elif par['synapse_config'] == 'std':
                    # implement synaptic short term derpression, but no facilitation
                    # we assume that syn_u remains constant at 1
                    syn_x += par['alpha_std']*(1-syn_x) - par['dt_sec']*syn_x*h
                    syn_x = tf.minimum(np.float32(1), tf.nn.relu(syn_x))
                    syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
                    h_post = syn_x*h

                elif par['synapse_config'] == 'stf':
                    # implement synaptic short term facilitation, but no depression
                    # we assume that syn_x remains constant at 1
                    syn_u += par['alpha_stf']*(par['U']-syn_u) + par['dt_sec']*par['U']*(1-syn_u)*h
                    syn_u = tf.minimum(np.float32(1), tf.nn.relu(syn_u))
                    h_post = syn_u*h

                else:
                    # no synaptic plasticity
                    h_post = h

                # Calculate new and recurrent rnn inputs
                rnn_input = tf.matmul(tf.nn.relu(self.W_in), tf.nn.relu(x))
                rnn_recur = tf.matmul(W_rnn, h_post)

                # Compute rnn state
                h = tf.nn.relu(h*(1-par['alpha_neuron']) \
                    + par['alpha_neuron']*(rnn_input + rnn_recur + b_rnn) \
                    + tf.random_normal([par['n_hidden'],par['batch_train_size']], 0, par['noise_rnn'], dtype=tf.float32))

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
        adam_optimizer = AdamOpt.AdamOpt(self.variables, learning_rate = par['learning_rate'])

        for n in range(par['num_networks']):

            # Calculate performance loss
            perf_loss = [mask*tf.nn.softmax_cross_entropy_with_logits(logits = y_hat, labels = desired_output, dim=0) \
                         for (y_hat, desired_output, mask) in zip(self.networks_output[n], self.target_data, self.mask)]
            perf_loss = tf.reduce_mean(tf.stack(perf_loss, axis=0))

            # Calculate spiking loss
            spike_loss = [par['spike_cost']*tf.reduce_mean(tf.square(h), axis=0) for h in self.networks_hidden[n]]
            spike_loss = tf.reduce_mean(tf.stack(spike_loss, axis=0))

            # Calculate wiring cost
            wiring_loss = [par['wiring_cost']*tf.nn.relu(W_rnn*par['W_rnn_dist']) for W_rnn in tf.trainable_variables() if 'W_rnn' in W_rnn.name]
            wiring_loss = tf.reduce_mean(tf.stack(wiring_loss, axis=0))

            # Add losses to record
            self.perf_losses.append(perf_loss)
            self.spike_losses.append(spike_loss)
            self.wiring_losses.append(wiring_loss)

            # Collect total loss
            self.total_loss += perf_loss + spike_loss + wiring_loss


        self.train_op = adam_optimizer.compute_gradients(self.total_loss)
        self.reset_adam_op = adam_optimizer.reset_params()
        self.reset_weights()


    def reset_weights(self):

        reset_weights = []
        for var in tf.trainable_variables():
            if 'b' in var.op.name:
                # reset biases to 0
                reset_weights.append(tf.assign(var, var*0.))
            else:
                for n in range(par['num_networks']):
                    if 'network' + str(n) + '/W_rnn' in var.op.name:
                        reset_weights.append(tf.assign(var, par['w_rnn0'][n]))
                    elif 'network' + str(n) + '/W_out' in var.op.name:
                        reset_weights.append(tf.assign(var, par['w_out0'][n]))

        self.reset_weights = tf.group(*reset_weights)


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

    mask = tf.placeholder(tf.float32, shape=[par['num_time_steps'], par['batch_train_size']])
    x = tf.placeholder(tf.float32, shape=[par['n_input'], par['num_time_steps'], par['batch_train_size']])
    y = tf.placeholder(tf.float32, shape=[par['n_output'], par['num_time_steps'], par['batch_train_size']])

    """ Start TensorFlow session """
    with tf.Session() as sess:
        if gpu_id is None:
            model = Model(x, y, mask)
        else:
            with tf.device("/gpu:0"):
                model = Model(x, y, mask)

        # Initialize session variables
        init = tf.global_variables_initializer()
        sess.run(init)
        t_start = time.time()

        # Restore variables from previous model if desired
        saver = tf.train.Saver()
        if par['load_previous_model']:
            saver.restore(sess, par['save_dir'] + par['ckpt_load_fn'])
            print('Model ' +  par['ckpt_load_fn'] + ' restored.')

        results = create_results_dict()

        for k in range(par['num_network_iters']):
            print('NETWORK ITERATION ', k)
            for i in range(par['num_iterations']):

                # generate batch of batch_train_size
                trial_info = stim.generate_trial()

                """
                Run the model
                """
                _, total_loss, perf_loss, spike_loss, wiring_loss, network_output = sess.run([model.train_op, model.total_loss, model.perf_losses, \
                    model.spike_losses, model.wiring_losses, model.networks_output], {x: trial_info['neural_input'], \
                    y: trial_info['desired_output'], mask: trial_info['train_mask']})

                if (i+1)%par['iters_between_outputs'] == 0:# and i != 0:
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
                    wirestr = 'Wiring Loss: {:>7.4}'.format(np.mean(wiring_loss))
                    accuracystr = 'Accuracy: {:>7.4} +/- {:<7.4}'.format(np.mean(accuracy), np.std(accuracy))

                    print(' | '.join([str(x) for x in [iterstr, timestr, perfstr, spikstr, wirestr, accuracystr]]))

            save_data(results, k, network_output, trial_info)
            update_dependencies()
            sess.run(model.reset_weights)
            sess.run(model.reset_adam_op)



def save_data(results, k, network_output, trial_info):

    # save data
    ind = range(k*par['num_networks'], (k+1)*par['num_networks'])
    accuracy = [get_perf(trial_info['desired_output'], y_hat, trial_info['train_mask']) for y_hat in network_output]
    W_rnn, W_out, b_rnn, b_out = eval_weights()
    results['W_rnn'][ind, :] = W_rnn
    results['W_out'][ind, :] = W_out
    results['b_rnn'][ind, :] = b_rnn
    results['b_out'][ind, :] = b_out
    results['accuracy'][ind] = np.array(accuracy)
    pickle.dump(results, open(par['save_dir'] + par['save_fn'], 'wb') )


def create_results_dict():

    results = {
        'W_rnn'     : np.zeros((par['num_networks']*par['num_network_iters'], par['n_hidden']*par['n_hidden']), dtype = np.float32),
        'W_out'     : np.zeros((par['num_networks']*par['num_network_iters'], par['n_hidden']*par['n_output']), dtype = np.float32),
        'b_rnn'     : np.zeros((par['num_networks']*par['num_network_iters'], par['n_hidden']), dtype = np.float32),
        'b_out'     : np.zeros((par['num_networks']*par['num_network_iters'], par['n_output']), dtype = np.float32),
        'accuracy'  : np.zeros((par['num_networks']*par['num_network_iters']), dtype = np.float32)}

    return results

def eval_weights():

    W_rnn = np.zeros((par['num_networks'], par['n_hidden']*par['n_hidden']), dtype = np.float32)
    W_out = np.zeros((par['num_networks'], par['n_hidden']*par['n_output']), dtype = np.float32)
    b_rnn = np.zeros((par['num_networks'], par['n_hidden']), dtype = np.float32)
    b_out = np.zeros((par['num_networks'], par['n_output']), dtype = np.float32)

    for n in range(par['num_networks']):
        with tf.variable_scope('network'+str(n), reuse=True):
            x = tf.get_variable('W_rnn')
            W_rnn[n, :] = np.reshape(x.eval(), (1, par['n_hidden']*par['n_hidden']))
            x = tf.get_variable('W_out')
            W_out[n, :] = np.reshape(x.eval(), (1, par['n_hidden']*par['n_output']))
            x = tf.get_variable('b_rnn')
            b_rnn[n, :] = np.reshape(x.eval(), (1, par['n_hidden']))
            x = tf.get_variable('b_out')
            b_out[n, :] = np.reshape(x.eval(), (1, par['n_output']))

    return W_rnn, W_out, b_rnn, b_out

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

"""
if __name__ == '__main__':
    main('testing', str(sys.argv[1]))
"""
