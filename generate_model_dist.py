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
import matplotlib.pyplot as plt

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
        self.w_rnn_mask = tf.constant(p.par['w_rnn_mask'], dtype=tf.float32)
        self.w_out_mask = tf.constant(p.par['w_out_mask'], dtype=tf.float32)

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
                    -0.1, 0.1, dtype=tf.float32)
                tf.get_variable('W', initializer=w, trainable=True)
                b = tf.random_uniform([p.par['generator_dims'][n+1],1], -0.1, 0.1, dtype=tf.float32)
                #b = tf.zeros([p.par['generator_dims'][n+1], 1], dtype=tf.float32)
                tf.get_variable('b', initializer=b, trainable=True)
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

                    W = tf.nn.relu(tf.get_variable('W'))
                    b = tf.nn.tanh(tf.get_variable('b'))
                    z = tf.nn.relu(tf.matmul(W, z) + b)
                    if m == 0:
                        self.z0 = z
                    if m == 1:
                        self.z1 = z

                    z = tf.nn.dropout(z, 0.9999)

            with tf.variable_scope('generator_output', reuse=True):
                W_rnn_gen = tf.get_variable('W_rnn')
                W_out_gen = tf.get_variable('W_out')
                b_rnn_gen = tf.get_variable('b_rnn')
                b_out_gen = tf.get_variable('b_out')

                W_rnn = tf.matmul(tf.nn.relu(W_rnn_gen), z)
                W_out = tf.matmul(W_out_gen, z)
                b_rnn = tf.matmul(b_rnn_gen, z)
                b_out = tf.matmul(b_out_gen, z)

                W_rnn = tf.reshape(W_rnn,(p.par['n_hidden'], p.par['n_hidden']))
                W_out = tf.reshape(W_out,(p.par['n_output'], p.par['n_hidden']))
                b_rnn = tf.reshape(b_rnn,(p.par['n_hidden'], 1))
                b_out = tf.reshape(b_out,(p.par['n_output'], 1))
                #b_out = tf.constant(np.zeros((3,1)), dtype = tf.float32)

            if p.par['EI']:
                W_rnn *= self.w_rnn_mask
                W_rnn = tf.matmul(tf.nn.relu(W_rnn), self.W_ei)

            W_out *= self.w_out_mask

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
        print(self.variables)
        adam_optimizer = AdamOpt.AdamOpt(self.variables, learning_rate = p.par['learning_rate'])

        #adam_optimizer = tf.train.AdamOptimizer(learning_rate = p.par['learning_rate'])

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


        self.train_op = adam_optimizer.compute_gradients(self.total_loss)
        self.grads = adam_optimizer.return_delta_grads()
        #self.train_op = adam_optimizer.minimize(self.total_loss)
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

        generator_var = np.random.normal(0,1,size=(p.par['num_networks'], p.par['generator_dims'][0]))

        for k in range(p.par['num_network_iters']):
            print('NETWORK ITERATION ', k)

            for i in range(p.par['num_iterations']):

                # generate batch of batch_train_size
                trial_info = stim.generate_trial()

                # each iteration, regenerate a new generator_var with a fixed prob
                ind_regen = np.where(np.random.rand(p.par['num_networks']) < p.par['regenerate_var_prob'])[0]
                if len(ind_regen) > 0:
                    new_gen_vars = np.random.normal(0,1,size=(len(ind_regen), p.par['generator_dims'][0]))
                    generator_var[ind_regen, :] = new_gen_vars
                    print('Regenerating ', len(ind_regen), ' vectors')

                # get the intiial weights
                if i == -1:
                    W_rnn0, W_out0 = get_initial_weights()

                """
                Run the model
                """
                _, total_loss, perf_loss, spike_loss, network_output, grads, z0, z1 = sess.run([model.train_op, model.total_loss, model.perf_losses, \
                    model.spike_losses, model.networks_output, model.grads, model.z0, model.z1], {x: trial_info['neural_input'], \
                    y: trial_info['desired_output'], mask: trial_info['train_mask'], z: generator_var})

                """
                print(type(grads))
                for v,g in grads.items():
                    print('v',v)
                    print('g',np.sum(g))
                    if v == 'generator_output/W_rnn':
                        plt.imshow(g, aspect='auto', interpolation = 'none')
                        plt.colorbar()
                        plt.show()
                        print(z0.shape, z1.shape)
                        #plt.imshow(z, aspect='auto', interpolation = 'none')
                        plt.plot(z0[:,0],'b')
                        plt.plot(z1[:,0],'r')
                        plt.show()
                """

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

                if (i+1)%500 == 0:

                    accuracy = np.array([get_perf(trial_info['desired_output'], y_hat, trial_info['train_mask']) for y_hat in network_output])
                    novel_generator_var = np.random.normal(0,1,size=(p.par['num_networks'], p.par['generator_dims'][0]))
                    trial_info = stim.generate_trial()
                    network_output = sess.run(model.networks_output, {x: trial_info['neural_input'], \
                        y: trial_info['desired_output'], mask: trial_info['train_mask'], z: novel_generator_var})
                    novel_accuracy = np.array([get_perf(trial_info['desired_output'], y_hat, trial_info['train_mask']) for y_hat in network_output])
                    accuracystr = 'Accuracy: {:>7.4} +/- {:<7.4}'.format(np.mean(novel_accuracy), np.std(novel_accuracy))
                    print('Novel ', accuracystr)
                    print('Saving data...')
                    save_data(accuracy, novel_accuracy)






def save_data(train_accuracy, novel_accuracy):

    # save data
    results = {}
    results['weight_dict'] = eval_weights()
    results['train_accuracy'] = train_accuracy
    results['novel_accuracy'] = novel_accuracy
    results['par'] = p.par
    pickle.dump(results, open(p.par['save_dir'] + p.par['save_fn'], 'wb') )



def eval_weights():

    weight_dict = {}

    for n in range(len(p.par['generator_dims']) - 1):
        with tf.variable_scope('generator'+str(n), reuse=True):
            W = tf.get_variable('W')
            weight_dict['W'+str(n)] = W.eval()
            b = tf.get_variable('b')
            weight_dict['b'+str(n)] = b.eval()
    with tf.variable_scope('generator_output', reuse=True):
        W = tf.get_variable('W_rnn')
        weight_dict['W_rnn'] = W.eval()
        W = tf.get_variable('W_out')
        weight_dict['W_out'] = W.eval()
        W = tf.get_variable('b_rnn')
        weight_dict['b_rnn'] = W.eval()
        W = tf.get_variable('b_out')
        weight_dict['b_out'] = W.eval()

    return weight_dict

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


if __name__ == '__main__':
    try:
        # GPU designated by first argument (must be integer 0-3)
        try:
            print('Selecting GPU ',  sys.argv[1])
            assert(int(sys.argv[1]) in [0,1,2,3])
        except AssertionError:
            quit('Error: Select a valid GPU number.')

        # Run model
        main(sys.argv[1])
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt')
