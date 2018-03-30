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

    def __init__(self, input_data, target_data, mask, generator_var, gating):

        # Load the input activity, the target data, and the training mask for this batch of trials
        self.input_data = tf.unstack(input_data, axis=1)
        self.target_data = tf.unstack(target_data, axis=1)
        self.mask = tf.unstack(mask, axis=0)
        self.generator_var = generator_var
        self.gating = gating

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

        C = 0.01
        for n in range(len(p.par['generator_dims']) - 1):
            with tf.variable_scope('generator'+str(n)):
                w = tf.random_uniform([p.par['generator_dims'][n+1],p.par['generator_dims'][n]], \
                    -C, C, dtype=tf.float32)
                tf.get_variable('W', initializer=w, trainable=True)
                #b = tf.zeros([p.par['generator_dims'][n+1], 1], dtype=tf.float32)
                #tf.get_variable('b', initializer=b, trainable=True)
        with tf.variable_scope('generator_output'):
            w = tf.random_uniform([p.par['wrnn_generator_dims'][1],p.par['wrnn_generator_dims'][0]], \
                -C, C, dtype=tf.float32)
            tf.get_variable('W_rnn', initializer=w, trainable=True)
            w = tf.random_uniform([p.par['wout_generator_dims'][1],p.par['wout_generator_dims'][0]], \
                -C, C, dtype=tf.float32)
            tf.get_variable('W_out', initializer=w, trainable=True)
            w = tf.random_uniform([p.par['brnn_generator_dims'][1],p.par['brnn_generator_dims'][0]], \
                -C, C, dtype=tf.float32)
            tf.get_variable('b_rnn', initializer=w, trainable=True)
            w = tf.random_uniform([p.par['bout_generator_dims'][1],p.par['bout_generator_dims'][0]], \
                -C, C, dtype=tf.float32)
            tf.get_variable('b_out', initializer=w, trainable=True)



    def run_model(self):

        self.networks_hidden = []
        self.networks_output = []
        self.networks_syn_x = []
        self.networks_syn_u = []


        z = tf.reshape(self.generator_var,(p.par['generator_dims'][0], 1))
        z = tf.nn.dropout(z, 0.999999)
        for m in range(len(p.par['generator_dims']) - 1):
            with tf.variable_scope('generator'+str(m), reuse=True):
                W = tf.get_variable('W')
                #b = tf.get_variable('b')
                print('W', W)
                print('old z',z)
                z = tf.nn.relu(tf.matmul(W, z))
                #z = tf.nn.dropout(z, 0.999999)
                #gate = tf.tile(tf.reshape(self.gating[m+1],[1,p.par['generator_dims'][m+1]]),[p.par['batch_train_size'],1])
                print('new z',z)
                #print(gate)
                z = z*tf.reshape(self.gating[m+1],[p.par['generator_dims'][m+1], 1])
                if m == 0:
                    self.z0 = z
                if m == 1:
                    self.z1 = z

        with tf.variable_scope('generator_output', reuse=True):
            W_rnn_gen = tf.get_variable('W_rnn')
            W_out_gen = tf.get_variable('W_out')
            b_rnn_gen = tf.get_variable('b_rnn')
            b_out_gen = tf.get_variable('b_out')

            W_rnn = tf.nn.relu(tf.matmul(W_rnn_gen, z))
            W_out = tf.nn.relu(tf.matmul(W_out_gen, z))
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

        #hidden_state_hist = []
        syn_x_hist = []
        syn_u_hist = []
        #output_rec = []

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
                + tf.random_normal([p.par['n_hidden'],p.par['batch_train_size']], \
                0, p.par['noise_rnn'], dtype=tf.float32))
            #h = tf.nn.l2_normalize(h, dim=0)

            # Compute output state
            output = tf.matmul(tf.nn.relu(W_out), h) + b_out

            # Record the outputs of this time step
            #hidden_state_hist.append(h)
            syn_x_hist.append(syn_x)
            syn_u_hist.append(syn_u)
            #output_rec.append(output)
            self.networks_hidden.append(h)
            self.networks_output.append(output)

        #self.networks_hidden.append(hidden_state_hist)
        #self.networks_output.append(output_rec)
        #self.networks_syn_x.append(syn_x_hist)
        #self.networks_syn_u.append(syn_u_hist)


    def optimize(self):

        self.variables = [var for var in tf.trainable_variables() if not var.op.name.find('conv')==0]
        print(self.variables)
        adam_optimizer = AdamOpt.AdamOpt(self.variables, learning_rate = p.par['learning_rate'])

        previous_weights_mu_minus_1 = {}
        reset_prev_vars_ops = []
        self.big_omega_var = {}
        aux_losses = []

        for var in self.variables:
            self.big_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            previous_weights_mu_minus_1[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            aux_losses.append(p.par['omega_c']*tf.reduce_sum(tf.multiply(self.big_omega_var[var.op.name], \
               tf.square(previous_weights_mu_minus_1[var.op.name] - var) )))
            reset_prev_vars_ops.append( tf.assign(previous_weights_mu_minus_1[var.op.name], var ) )

        self.aux_loss = tf.add_n(aux_losses)

        # Calculate performance loss
        self.perf_loss = [mask*tf.nn.softmax_cross_entropy_with_logits(logits = y_hat, labels = desired_output, dim=0) \
            for (y_hat, desired_output, mask) in zip(self.networks_output, self.target_data, self.mask)]
        self.perf_loss = tf.reduce_mean(self.perf_loss)

        # Calculate spiking loss
        self.spike_loss = [p.par['spike_cost']*tf.reduce_mean(tf.square(h), axis=0) for h in self.networks_hidden]
        self.spike_loss = tf.reduce_mean(self.spike_loss)


        # Collect total loss
        self.total_loss = self.perf_loss +self. spike_loss


        # Gradient of the loss+aux function, in order to both perform training and to compute delta_weights
        with tf.control_dependencies([self.total_loss, self.aux_loss]):
            self.train_op = adam_optimizer.compute_gradients(self.total_loss + self.aux_loss)

        # Zenke method
        self.pathint_stabilization(adam_optimizer, previous_weights_mu_minus_1)


        self.reset_prev_vars = tf.group(*reset_prev_vars_ops)
        self.reset_adam_op = adam_optimizer.reset_params()


    def pathint_stabilization(self, adam_optimizer, previous_weights_mu_minus_1):
        # Zenke method

        optimizer_task = tf.train.GradientDescentOptimizer(learning_rate =  1.0)
        small_omega_var = {}

        reset_small_omega_ops = []
        update_small_omega_ops = []
        update_big_omega_ops = []
        initialize_prev_weights_ops = []

        for var in self.variables:

            small_omega_var[var.op.name] = tf.Variable(tf.zeros(var.get_shape()), trainable=False)
            reset_small_omega_ops.append( tf.assign( small_omega_var[var.op.name], small_omega_var[var.op.name]*0.0 ) )
            update_big_omega_ops.append( tf.assign_add( self.big_omega_var[var.op.name], tf.div(tf.nn.relu(small_omega_var[var.op.name]), \
            	(p.par['omega_xi'] + tf.square(var-previous_weights_mu_minus_1[var.op.name])))))


        # After each task is complete, call update_big_omega and reset_small_omega
        self.update_big_omega = tf.group(*update_big_omega_ops)

        # Reset_small_omega also makes a backup of the final weights, used as hook in the auxiliary loss
        self.reset_small_omega = tf.group(*reset_small_omega_ops)

        # This is called every batch
        with tf.control_dependencies([self.train_op]):
            self.delta_grads = adam_optimizer.return_delta_grads()
            self.gradients = optimizer_task.compute_gradients(self.total_loss)
            self.grads = adam_optimizer.return_delta_grads()
            for grad,var in self.gradients:
                update_small_omega_ops.append(tf.assign_add(small_omega_var[var.op.name], -self.delta_grads[var.op.name]*grad ) )
            self.update_small_omega = tf.group(*update_small_omega_ops) # 1) update small_omega after each train!


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
    gating = [tf.placeholder(tf.float32, [p.par['generator_dims'][n]], 'gating') for n in range(len(p.par['generator_dims']))]

    """ Start TensorFlow session """
    with tf.Session() as sess:
        if gpu_id is None:
            model = Model(x, y, mask, z, gating)
        else:
            with tf.device("/gpu:0"):
                model = Model(x, y, mask, z, gating)

        # Initialize session variables
        init = tf.global_variables_initializer()
        sess.run(init)
        t_start = time.time()

        # Restore variables from previous model if desired
        saver = tf.train.Saver()
        if p.par['load_previous_model']:
            saver.restore(sess, p.par['save_dir'] + p.par['ckpt_load_fn'])
            print('Model ' +  p.par['ckpt_load_fn'] + ' restored.')

        generator_var_hist = []

        for k in range(p.par['num_network_iters']):
            print('NETWORK ITERATION ', k)
            generator_var = np.random.normal(0,1,size=(p.par['num_networks'], p.par['generator_dims'][0]))
            generator_var_hist.append(generator_var)
            gating_dict = {k:v for k,v in zip(gating, p.par['gating'][k])}

            for i in range(p.par['num_iterations']):

                # generate batch of batch_train_size
                trial_info = stim.generate_trial()

                """
                Run the model
                """
                _,_, total_loss, perf_loss, spike_loss, aux_loss, network_output, grads, z0, z1 = sess.run([model.train_op, \
                    model.update_small_omega, model.total_loss, model.perf_loss, \
                    model.spike_loss,  model.aux_loss, model.networks_output, model.grads, model.z0, model.z1], {x: trial_info['neural_input'], \
                    y: trial_info['desired_output'], mask: trial_info['train_mask'], z: generator_var, **gating_dict})

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


                if (i+1)%p.par['iters_between_outputs'] == 0:# and i != 0:
                    accuracy = get_perf(trial_info['desired_output'], network_output, trial_info['train_mask'])
                    iteration_time = time.time() - t_start
                    iterstr = 'Iter. {:>4}'.format(i)
                    timestr = 'Time. {:>7.4}'.format(iteration_time)
                    lossstr = 'Total Loss: {:>7.4}'.format(total_loss)
                    auxstr = 'Aux Loss: {:>7.4}'.format(aux_loss)
                    #perfstr = 'Perf. Loss: {:>7.4} +/- {:<7.4}'.format(np.mean(perf_loss), np.std(perf_loss))
                    #spikstr = 'Spike Loss: {:>7.4} +/- {:<7.4}'.format(np.mean(spike_loss), np.std(spike_loss))
                    #wirestr = 'Wiring Loss: {:>7.4} +/- {:<7.4}'.format(np.mean(wiring_loss), np.std(wiring_loss))
                    perfstr = 'Perf. Loss: {:>7.4}'.format(np.mean(perf_loss))
                    spikstr = 'Spike Loss: {:>7.4}'.format(np.mean(spike_loss))
                    #wirestr = 'Wiring Loss: {:>7.4}'.format(np.mean(wiring_loss))
                    accuracystr = 'Accuracy: {:>7.4} +/- {:<7.4}'.format(np.mean(accuracy), np.std(accuracy))

                    print(' | '.join([str(x) for x in [iterstr, timestr, perfstr, spikstr, accuracystr, auxstr]]))

                if (i+1)%2000 == 0:
                    num_reps = 50
                    accuracy = get_perf(trial_info['desired_output'], network_output, trial_info['train_mask'])


                    novel_accuracy = []
                    for r in range(num_reps):
                        novel_var = np.random.normal(0,1,size=(1, p.par['generator_dims'][0]))
                        gating_novel = []
                        for n in range(len(p.par['generator_dims'])):
                            gating_layer = np.zeros((p.par['generator_dims'][n]), dtype = np.float32)
                            for i in range(p.par['generator_dims'][n]):
                                if np.random.rand() < 1-p.par['gate_pct']:
                                    gating_layer[i] = 1
                            gating_novel.append(gating_layer)
                        trial_info = stim.generate_trial()
                        gating_dict = {k:v for k,v in zip(gating, gating_novel)}
                        network_output = sess.run(model.networks_output, {x: trial_info['neural_input'], \
                            y: trial_info['desired_output'], mask: trial_info['train_mask'], z: novel_var, \
                            **gating_dict})
                        novel_accuracy.append(get_perf(trial_info['desired_output'], network_output, \
                            trial_info['train_mask']))
                    previous_accuracy = []
                    for r in range(len(generator_var_hist)):
                        gating_dict = {k:v for k,v in zip(gating, p.par['gating'][r])}
                        network_output = sess.run(model.networks_output, {x: trial_info['neural_input'], \
                            y: trial_info['desired_output'], mask: trial_info['train_mask'], z: generator_var_hist[r], \
                            **gating_dict})

                        previous_accuracy.append(get_perf(trial_info['desired_output'], network_output, \
                            trial_info['train_mask']))
                    accuracystr = 'Accuracy: {:>7.4} +/- {:<7.4}'.format(np.mean(novel_accuracy), np.std(novel_accuracy))
                    prev_accuracystr = 'Accuracy: {:>7.4} +/- {:<7.4}'.format(np.mean(previous_accuracy), np.std(previous_accuracy))
                    print('Novel ', accuracystr)
                    print('Previous ', prev_accuracystr)
                    print('Saving data...')
                    save_data(accuracy, novel_accuracy, previous_accuracy)

            # Update big omegaes, and reset other values before starting new task
            big_omegas = sess.run([model.update_big_omega, model.big_omega_var])

            # Reset the Adam Optimizer, and set the previous parater values to their current values
            sess.run(model.reset_adam_op)
            sess.run(model.reset_prev_vars)
            sess.run(model.reset_small_omega)

def save_data(train_accuracy, novel_accuracy, previous_accuracy):

    # save data
    results = {}
    results['weight_dict'] = eval_weights()
    results['train_accuracy'] = train_accuracy
    results['novel_accuracy'] = novel_accuracy
    results['previous_accuracy'] = previous_accuracy
    results['par'] = p.par
    pickle.dump(results, open(p.par['save_dir'] + p.par['save_fn'], 'wb') )



def eval_weights():

    weight_dict = {}

    for n in range(len(p.par['generator_dims']) - 1):
        with tf.variable_scope('generator'+str(n), reuse=True):
            W = tf.get_variable('W')
            weight_dict['W'+str(n)] = W.eval()
            #b = tf.get_variable('b')
            #weight_dict['b'+str(n)] = b.eval()
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
