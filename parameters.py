import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from itertools import product
import os

print("--> Loading parameters...")

global par, analysis_par

"""
Independent parameters
"""

rnd_save_suffix = np.random.randint(10000)

par = {
    # Setup parameters
    'save_dir'              : 'C:/Users/nicol/Projects/GitHub/Meta-Networks/savedir/',
    'debug_model'           : False,
    'load_previous_model'   : False,
    'analyze_model'         : True,

    # Network configuration
    'synapse_config'        : None, # Full is 'std_stf'
    'exc_inh_prop'          : 0.8,       # Literature 0.8, for EI off 1
    'var_delay'             : True,

    # Network shape
    'num_networks'          : 25,
    'num_motion_tuned'      : 36,
    'num_fix_tuned'         : 0,
    'num_rule_tuned'        : 0,
    'n_hidden'              : 50,
    'n_output'              : 3,
    'n_latent'              : 10,
    'num_weights'           : int(50**2 + 50*3),


    # Timings and rates
    'dt'                    : 10,
    'learning_rate'         : 1e-3,
    'membrane_time_constant': 100,
    'connection_prob'       : 1.,         # Usually 1

    # Variance values
    'clip_max_grad_val'     : 1,
    'input_mean'            : 0.0,
    'noise_in_sd'           : 0.05,
    'noise_rnn_sd'          : 0.1,

    # Tuning function data
    'num_motion_dirs'       : 8,
    'tuning_height'         : 4,        # magnitutde scaling factor for von Mises
    'kappa'                 : 2,        # concentration scaling factor for von Mises

    # Cost parameters
    'spike_cost'            : 1e-7,
    'wiring_cost'           : 1e-1,
    'beta'                  : 1e-8,
    'accuracy_cost'         : 1.,

    # Synaptic plasticity specs
    'tau_fast'              : 200,
    'tau_slow'              : 1500,
    'U_stf'                 : 0.15,
    'U_std'                 : 0.45,

    # Training specs
    'batch_train_size'      : 256,
    'num_iterations'        : 20000,
    'iters_between_outputs' : 100,
    'num_network_iters'     : 40,

    # Task specs
    'trial_type'            : 'DMS', # allowable types: DMS, DMRS45, DMRS90, DMRS180, DMC, DMS+DMRS, ABBA, ABCA, dualDMS
    'rotation_match'        : 0,  # angular difference between matching sample and test
    'dead_time'             : 50,
    'fix_time'              : 50,
    'sample_time'           : 100,
    'delay_time'            : 500,
    'test_time'             : 100,
    'variable_delay_max'    : 300,
    'mask_duration'         : 30,  # duration of traing mask after test onset
    'catch_trial_pct'       : 0.0,
    'num_receptive_fields'  : 1,
    'num_rules'             : 1, # this will be two for the DMS+DMRS task
    'decoding_test_mode'    : False,

    # Save paths
    'save_fn'               : 'model_results.pkl',
    'ckpt_save_fn'          : 'model' + str(rnd_save_suffix) + '.ckpt',
    'ckpt_load_fn'          : 'model' + str(rnd_save_suffix) + '.ckpt',

    # Analysis
    'svm_normalize'         : True,
    'decoding_reps'         : 0,
    'simulation_reps'       : 0,
    'decode_test'           : False,
    'decode_rule'           : False,
    'decode_sample_vs_test' : False,
    'suppress_analysis'     : False,
    'analyze_tuning'        : False,

    'accuracy_threshold'    : 0.75,
    'file_prefix'           : 'DM'
}



"""
Dependent parameters
"""

def update_parameters(updates):
    """
    Takes a list of strings and values for updating parameters in the parameter dictionary
    Example: updates = [(key, val), (key, val)]
    """
    print('Updating parameters...')
    for key, val in updates.items():
        par[key] = val
        print('Updating ', key)

    update_trial_params()
    update_dependencies()

def update_trial_params():

    """
    Update all the trial parameters given trial_type
    """

    par['num_rules'] = 1
    par['num_rule_tuned'] = 0

    if par['trial_type'] == 'DMS' or par['trial_type'] == 'DMC':
        par['rotation_match'] = 0

    elif par['trial_type'] == 'DMRS45':
        par['rotation_match'] = 45

    elif par['trial_type'] == 'DMRS90':
        par['rotation_match'] = 90

    elif par['trial_type'] == 'DMRS90ccw':
        par['rotation_match'] = -90

    elif  par['trial_type'] == 'DMRS180':
        par['rotation_match'] = 180

    elif par['trial_type'] == 'dualDMS':
        par['catch_trial_pct'] = 0
        par['num_receptive_fields'] = 2
        par['num_rules'] = 2
        par['probe_trial_pct'] = 0
        par['probe_time'] = 10
        par['num_rule_tuned'] = 12
        par['sample_time'] = 500
        par['test_time'] = 500
        par['delay_time'] = 1000
        par['analyze_rule'] = True
        par['num_motion_tuned'] = 36
        par['noise_in_sd']  = 0.1
        par['noise_rnn_sd'] = 0.5
        par['num_iterations'] = 4000

        par['dualDMS_single_test'] = False

    elif par['trial_type'] == 'ABBA' or par['trial_type'] == 'ABCA':
        par['catch_trial_pct'] = 0
        par['match_test_prob'] = 0.5
        par['max_num_tests'] = 3
        par['sample_time'] = 400
        par['delay_time'] = 2400
        #par['spike_cost'] = 1e-2
        par['ABBA_delay'] = par['delay_time']//par['max_num_tests']//2
        par['repeat_pct'] = 0
        par['analyze_test'] = True
        if par['trial_type'] == 'ABBA':
            par['repeat_pct'] = 0.5

    elif par['trial_type'] == 'DMS+DMRS' or par['trial_type'] == 'DMS+DMRS_early_cue':

        par['num_rules'] = 2
        par['num_rule_tuned'] = 12
        if par['trial_type'] == 'DMS+DMRS':
            par['rotation_match'] = [0, 90]
            par['rule_onset_time'] = par['dead_time']+par['fix_time']+par['sample_time'] + 500
            par['rule_offset_time'] = par['dead_time']+par['fix_time']+par['sample_time'] + 750
        else:
            par['rotation_match'] = [0, 45]
            par['rule_onset_time'] = par['dead_time']
            par['rule_offset_time'] = par['dead_time']+par['fix_time']+par['sample_time']

    elif par['trial_type'] == 'DMS+DMC':
        par['num_rules'] = 2
        par['num_rule_tuned'] = 12
        par['rotation_match'] = [0, 0]
        par['rule_onset_time'] = par['dead_time']+par['fix_time']+par['sample_time'] + 500
        par['rule_offset_time'] = par['dead_time']+par['fix_time']+par['sample_time'] + par['delay_time'] + par['test_time']

    elif par['trial_type'] == 'DMS+DMRS+DMC':
        par['num_rules'] = 3
        par['num_rule_tuned'] = 18
        par['rotation_match'] = [0, 90, 0]
        par['rule_onset_time'] = par['dead_time']
        par['rule_offset_time'] = par['dead_time']+par['fix_time']+par['sample_time'] + par['delay_time'] + par['test_time']

    else:
        print(par['trial_type'], ' not a recognized trial type')
        quit()


def update_dependencies():
    """
    Updates all parameter dependencies
    """

    # Number of input neurons
    par['n_input'] = par['num_motion_tuned'] + par['num_fix_tuned'] + par['num_rule_tuned']
    # General network shape
    par['shape'] = (par['n_input'], par['n_hidden'], par['n_output'])

    par['encoder_dims'] = [par['num_weights'], 1000, 1000]
    par['decoder_dims'] = [par['n_latent'], 1000, 1000, par['num_weights']]
    par['accuracy_dims'] = [par['n_latent'], 1]

    # Possible rules based on rule type values
    #par['possible_rules'] = [par['num_receptive_fields'], par['num_categorizations']]

    # If num_inh_units is set > 0, then neurons can be either excitatory or
    # inihibitory; is num_inh_units = 0, then the weights projecting from
    # a single neuron can be a mixture of excitatory or inhibitory
    if par['exc_inh_prop'] < 1:
        par['EI'] = True
    else:
        par['EI']  = False

    par['num_exc_units'] = int(np.round(par['n_hidden']*par['exc_inh_prop']))
    par['num_inh_units'] = par['n_hidden'] - par['num_exc_units']

    par['EI_list'] = np.ones(par['n_hidden'], dtype=np.float32)
    print('Note that EI is currently hard-coded.')
    #par['EI_list'][-par['num_inh_units']:] = -1.
    par['EI_list'][20:25] = -1
    par['EI_list'][45:50] = -1

    par['EI_matrix'] = np.diag(par['EI_list'])
    par['ind_inh'] = np.where(par['EI_list'] == -1)[0]

    # Membrane time constant of RNN neurons
    par['alpha_neuron'] = np.float32(par['dt'])/par['membrane_time_constant']
    # The standard deviation of the Gaussian noise added to each RNN neuron
    # at each time step
    par['noise_rnn'] = np.sqrt(2*par['alpha_neuron'])*par['noise_rnn_sd']
    par['noise_in'] = np.sqrt(2/par['alpha_neuron'])*par['noise_in_sd'] # since term will be multiplied by par['alpha_neuron']

    par['num_exc'] = int(par['n_hidden']*par['exc_inh_prop'])
    par['num_inh'] = int(par['n_hidden'] - par['num_exc'])


    # General event profile info
    #par['name_of_stimulus'], par['date_stimulus_created'], par['author_of_stimulus_profile'] = get_profile(par['profile_path'])
    # List of events that occur for the network
    #par['events'] = get_events(par['profile_path'])
    # The time step in seconds
    par['dt_sec'] = par['dt']/1000
    # Length of each trial in ms
    if par['trial_type'] == 'dualDMS' and not par['dualDMS_single_test']:
        par['trial_length'] = par['dead_time']+par['fix_time']+par['sample_time']+2*par['delay_time']+2*par['test_time']
    else:
        par['trial_length'] = par['dead_time']+par['fix_time']+par['sample_time']+par['delay_time']+par['test_time']
    # Length of each trial in time steps
    par['num_time_steps'] = par['trial_length']//par['dt']


    ####################################################################
    ### Setting up assorted intial weights, biases, and other values ###
    ####################################################################

    par['h_init'] = 0.5*np.ones((par['n_hidden'], par['batch_train_size']), dtype=np.float32)
    par['h_init'][par['ind_inh']] = 2.0

    par['input_to_hidden_dims'] = [par['n_hidden'], par['n_input']]
    par['hidden_to_hidden_dims'] = [par['n_hidden'], par['n_hidden']]


    # Building input weights
    par['W_in'] = np.zeros(par['input_to_hidden_dims'])


    stim_dirs = np.float32(np.arange(0,360,360/par['n_input']))[np.newaxis,:]
    exc_dirs = np.float32(np.arange(0,360,360/20))[:,np.newaxis]
    inh_dirs = np.float32(np.arange(0,360,360/5))[:,np.newaxis]

    d_exc = np.cos((stim_dirs - exc_dirs)/180*np.pi)
    d_inh = np.cos((stim_dirs - inh_dirs)/180*np.pi)

    k_exc = 4
    k_inh = 1.5

    par['W_in'][:20,:]   = np.exp(k_exc * d_exc)
    par['W_in'][20:25,:] = np.exp(k_inh * d_inh)

    par['W_in'][:20,:]   /= np.sum(par['W_in'][:20,:], axis=1)[:,np.newaxis]
    par['W_in'][20:25,:] /= np.sum(par['W_in'][20:25,:], axis=1)[:,np.newaxis]


    # Building spatial embedding for recurrent weights

    par['W_rnn_dist'] = np.zeros(par['hidden_to_hidden_dims'])

    r_inh = 2.0     # Corresponds to inh_dirs
    r_exc = 2.5     # Corresponds to exc_dirs
    h     = 1.0
    """
    neuron_location = np.zeros((par['n_hidden'], 3))
    num_exc = par['n_hidden']*par['exc_inh_prop']
    num_inh = par['n_hidden'] - num_exc
    for i in range(range(par['n_hidden']):
        if i >=  par['n_hidden']/2:
            neuron_location[i,2] = h
        if par['EI_list'][i] == 1:
            neuron_location[i,0] = r_exc*np.cos(2*np.pi*)
        else:

    """
    for i, j in product(range(par['n_hidden']), range(par['n_hidden'])):

        # Determining r1 and t1
        if i in range(20,25) or i in range(45,50):
            r1 = r_inh
            t1 = inh_dirs[(i%25)%5]
        else:
            r1 = r_exc
            t1 = exc_dirs[i%25]

        # Determining r2
        if j in range(20,25) or j in range(45,50):
            r2 = r_inh
            t2 = inh_dirs[(j%25)%5]
        else:
            r2 = r_exc
            t2 = exc_dirs[j%25]

        # Finding whether the heights are different
        if i < 25 and j < 25:
            rh = 0
        elif i >= 25 and j >= 25:
            rh = 0
        else:
            rh = h

        # Calculating distance
        par['W_rnn_dist'][i,j] = np.sqrt( (r1 * np.sin(t1/180*np.pi) - r2 * np.sin(t2/180*np.pi))**2 \
                                        + (r1 * np.cos(t1/180*np.pi) - r2 * np.cos(t2/180*np.pi))**2 + rh)

    par['w_out_mask'] = np.ones((par['n_output'], par['n_hidden']), dtype=np.float32)
    if par['EI']:
        par['w_out_mask'][:, par['ind_inh']] = 0

    print('Generating random initial weights...')
    par['w_rnn0'] = []
    par['w_out0'] = []
    for n in range(par['num_networks']):
        par['w_out0'].append(initialize([par['n_output'], par['n_hidden']], par['connection_prob']))
        if par['EI']:
            par['w_rnn0'].append(initialize(par['hidden_to_hidden_dims'], par['connection_prob']))
            par['w_rnn0'][-1][:, par['ind_inh']] *= 4
            par['w_out0'][-1][:, par['ind_inh']] = 0
            #if par['synapse_config'] == None:
                #par['w_rnn0'][-1] = par['w_rnn0'][-1]/(spectral_radius(par['w_rnn0']))
            for i in range(par['n_hidden']):
                par['w_rnn0'][-1][i,i] = 0
            par['w_rnn_mask'] = np.ones((par['hidden_to_hidden_dims']), dtype=np.float32) - np.eye(par['n_hidden'])
        else:
            par['w_rnn0'].append(0.54*np.eye(par['n_hidden']))
            par['w_rnn_mask'] = np.ones((par['hidden_to_hidden_dims']), dtype=np.float32)




    """
    Setting up synaptic parameters
    0 = static
    1 = facilitating
    2 = depressing
    """
    par['synapse_type'] = np.zeros(par['n_hidden'], dtype=np.int8)

    # only facilitating synapses
    if par['synapse_config'] == 'stf':
        par['synapse_type'] = np.ones(par['n_hidden'], dtype=np.int8)

    # only depressing synapses
    elif par['synapse_config'] == 'std':
        par['synapse_type'] = 2*np.ones(par['n_hidden'], dtype=np.int8)

    # even numbers facilitating, odd numbers depressing
    elif par['synapse_config'] == 'std_stf':
        par['synapse_type'] = np.ones(par['n_hidden'], dtype=np.int8)
        par['ind'] = range(1,par['n_hidden'],2)
        par['synapse_type'][par['ind']] = 2

    par['alpha_stf'] = np.ones((par['n_hidden'], 1), dtype=np.float32)
    par['alpha_std'] = np.ones((par['n_hidden'], 1), dtype=np.float32)
    par['U'] = np.ones((par['n_hidden'], 1), dtype=np.float32)

    # initial synaptic values
    par['syn_x_init'] = np.zeros((par['n_hidden'], par['batch_train_size']), dtype=np.float32)
    par['syn_u_init'] = np.zeros((par['n_hidden'], par['batch_train_size']), dtype=np.float32)

    for i in range(par['n_hidden']):
        if par['synapse_type'][i] == 1:
            par['alpha_stf'][i,0] = par['dt']/par['tau_slow']
            par['alpha_std'][i,0] = par['dt']/par['tau_fast']
            par['U'][i,0] = 0.15
            par['syn_x_init'][i,:] = 1
            par['syn_u_init'][i,:] = par['U'][i,0]

        elif par['synapse_type'][i] == 2:
            par['alpha_stf'][i,0] = par['dt']/par['tau_fast']
            par['alpha_std'][i,0] = par['dt']/par['tau_slow']
            par['U'][i,0] = 0.45
            par['syn_x_init'][i,:] = 1
            par['syn_u_init'][i,:] = par['U'][i,0]

def initialize(dims, connection_prob):
    #w = np.random.gamma(shape=0.25, scale=1.0, size=dims)
    w = np.random.uniform(0,0.25, size=dims)
    #w = np.random.uniform(0,0.5, size=dims)**2
    #w = np.minimum(1, -np.log(w))
    w *= (np.random.rand(*dims) < connection_prob)
    return np.float32(w)


def spectral_radius(A):

    return np.max(abs(np.linalg.eigvals(A)))

update_trial_params()
update_dependencies()

print("--> Parameters successfully loaded.\n")
