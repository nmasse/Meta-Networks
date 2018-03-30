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

def get_perf(y, y_hat, mask):

    """
    Calculate task accuracy by comparing the actual network output to the desired output
    only examine time points when test stimulus is on
    in another words, when y[0,:,:] is not 0
    y is the desired output
    y_hat is the actual output
    """
    y_hat = np.stack(y_hat, axis=1)
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
