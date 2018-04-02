import numpy as np
from itertools import product
from itertools import permutations
import matplotlib.pyplot as plt
import pickle
import os
import stimulus
import parameters
import analysis


class Motifs:

    def __init__(self, data_dir, file_prefix, N = None):

        self.motifs = {}
        self.motif_sizes = [2,3,4]
        data_files = os.listdir(data_dir)

        for f in data_files:
            if f.startswith(file_prefix):
                print('Processing ', f)
                self.current_filename = f
                W, v = self.make_matrix(data_dir + f, 'elim_lesion', N)
                print(type(W))
                if type(W) is list:
                    for i,w1 in enumerate(W):
                        self.find_motifs(w1, v)
                else:
                    self.find_motifs(W, v)

        self.print_motif_list()


    def make_matrix(self, filename, method, N):
        x = pickle.load(open(filename, 'rb'))
        beh_threshold = 0.1
        val_th = 0.1
        ind_accurate = np.where(np.array(x['accuracy_hist']) > 0.98)[0]
        #N = np.argmax(ind_accurate)
        #N = 6
        print('N = ', N)

        if method == 'elim_lesion' or  method == 'elim':
            parameters.update_parameters(x['par'])
            s = stimulus.Stimulus()
            trial_info = s.generate_trial()


        if method == 'lesion':
            significant_weights_rnn = x['model_performance']['accuracy'][-1] - x['lesion_accuracy_rnn'][0,:,:] > beh_threshold
            significant_weights_out = x['model_performance']['accuracy'][-1] - x['lesion_accuracy_out'][0,:,:] > beh_threshold
            v = np.array([0]*x['parameters']['num_exc_units'] + [1]*x['parameters']['num_inh_units'] \
                + [2]*x['parameters']['n_output'])
            W = np.vstack((significant_weights_rnn, significant_weights_out))
            d = W.shape[0] - W.shape[1]
            W = np.hstack((W, np.zeros((W.shape[0], d))))

        elif method == 'elim':
            num_units = 50 - N
            w1 = np.zeros((num_units, num_units))
            w2 = np.zeros((3, num_units))
            ind = np.where(x['gate_hist'][N]>0)[0]
            for i in range(num_units):
                for j in range(num_units):
                    w1[i,j] = x['weights_hist'][N]['w_rnn'][ind[i], ind[j]] > val_th
                for j in range(3):
                    w2[j,i] = x['weights_hist'][N]['w_out'][j, ind[i]] > val_th
            n_exc = int(np.sum(x['gate_hist'][N][:x['par']['num_exc']]))
            n_inh = int(np.sum(x['gate_hist'][N][x['par']['num_exc']:]))
            v = np.array([0]*n_exc + [1]*n_inh + [2]*x['par']['n_output'])
            W = np.vstack((w1, w2))
            d = W.shape[0] - W.shape[1]
            W = np.hstack((W, np.zeros((W.shape[0], d))))



        elif method == 'elim_lesion':
            num_units = 50 - N
            r = analysis.lesion_weights(trial_info, x['par']['h_init'], x['par']['syn_x_init'], x['par']['syn_u_init'], \
                x['weights_hist'][N], x['gate_hist'][N])
            #plt.imshow(np.squeeze(r['lesion_accuracy_rnn']), aspect='auto', interpolation = 'none')
            #plt.colorbar()
            #plt.show()
            w1_full = np.tile(x['accuracy_hist'][N],(x['par']['n_hidden'],x['par']['n_hidden'])) - np.squeeze(r['lesion_accuracy_rnn']) > beh_threshold
            w2_full = np.tile(x['accuracy_hist'][N],(x['par']['n_output'],x['par']['n_hidden'])) - np.squeeze(r['lesion_accuracy_out']) > beh_threshold
            w1 = np.zeros((num_units, num_units))
            w2 = np.zeros((3, num_units))
            ind = np.where(x['gate_hist'][N]>0)[0]
            for i in range(num_units):
                for j in range(num_units):
                    w1[i,j] = w1_full[ind[i], ind[j]]
                for j in range(3):
                    w2[j,i] = w2_full[j, ind[i]]
            #plt.imshow(w1, aspect='auto', interpolation = 'none')
            #plt.colorbar()
            #plt.show()
            print('accuracy ', x['accuracy_hist'][N])
            n_exc = int(np.sum(x['gate_hist'][N][:x['par']['num_exc']]))
            n_inh = int(np.sum(x['gate_hist'][N][x['par']['num_exc']:]))
            v = np.array([0]*n_exc + [1]*n_inh + [2]*x['par']['n_output'])
            W = np.vstack((w1, w2))
            d = W.shape[0] - W.shape[1]
            W = np.hstack((W, np.zeros((W.shape[0], d))))
            plt.imshow(W, aspect='auto', interpolation = 'none')
            plt.colorbar()
            plt.show()
            print(v)

        elif method == 'stacked':
            W = []
            for i in range(x['W_rnn'].shape[0]):
                w1 = np.reshape(x['W_rnn'][i,:], (50,50))>0.2
                w2 = np.reshape(x['W_out'][i,:], (3,50))>0.2
                v = np.array([0]*40 + [1]*10 + [2]*3)
                W1 = np.vstack((w1, w2))
                d = W1.shape[0] - W1.shape[1]
                W1 = np.hstack((W1, np.zeros((W1.shape[0], d))))
                W.append(W1)

        return W, v


    def connection_probs(self):

        unique_labels = np.unique(self.v).tolist() # [Inhibitory, Excitatory, Output]
        N = len(unique_labels)

        total      = np.zeros([N,N], dtype=np.float32)
        connection = np.zeros([N,N], dtype=np.float32)
        for (i, v_in), (j, v_out) in product(enumerate(input_labels), enumerate(output_labels)):
            l_in  = unique_labels.index(v_in)
            l_out = unique_labels.index(v_out)
            if i != j:
                total[l_in, l_out] += 1
                if self.W[j,i] > 0:
                    connection[l_in, l_out] += 1

        self.p_connection = np.zeros((N,N), dtype = np.float32)
        for n1, n2 in product(range(N), range(N)):
            self.p_connection[n1, n2] = connection[n1, n2]/total[n1,n2] if total[n1,n2] != 0 else -1


    def find_motifs(self, W ,v):

        W, v = self.prune_network(W, v)
        for i in self.motif_sizes:
            self.find_motif_set_size(W, v, i)

    def return_motifs(self):

        return self.motifs


    def find_motif_set_size(self,W, v, c):


        N = W.shape[0]
        for i0 in range(N):
            ind0 = np.where((W[:, i0] > 0) + (W[i0, :] > 0))[0]
            for i1 in np.setdiff1d(ind0, i0):
                if c == 2:
                    self.motif_properties(W, v, [i0, i1])
                else:
                    ind1 = np.where((W[:, i1] > 0) + (W[i1, :] > 0))[0]
                    for i2 in np.setdiff1d(ind1,[i0,i1]):
                        if c == 3:
                            self.motif_properties(W, v, [i0, i1, i2])
                        else:
                            ind2 = np.where((W[:, i2] > 0) + (W[i2, :] > 0))[0]
                            for i3 in np.setdiff1d(ind2,[i0,i1,i2]):
                                if c == 4:
                                    self.motif_properties(W, v, [i0, i1, i2, i3])
                                else:
                                    ind3 = np.where((W[:, i3] > 0) + (W[i3, :] > 0))[0]
                                    for i4 in np.setdiff1d(ind3,[i0,i1,i2,i3]):
                                        if c == 5:
                                            self.motif_properties(W, v, [i0, i1, i2, i3, i4])
                                        else:
                                            ind4 = np.where((W[:, i4] > 0) + (W[i4, :] > 0))[0]
                                            for i5 in np.setdiff1d(ind4,[i0,i1,i2,i3,i4]):
                                                if c == 6:
                                                    self.motif_properties(W, v, [i0, i1, i2, i3, i4, i5])


    def motif_properties(self, W, v, u):

        u = sorted(u)
        W1 = W[:, u]
        W1 = W1[u, :]
        v1 = v[u]

        if np.sum(W1) < len(u):
            return

        # check for loops
        #for i in range(len(v)):


        s = [str(int(i)) for i in v1]
        id0 = ''.join(s)

        s = [str(int(i)) for i in np.reshape(np.where(W1 > 0, 1, 0), (len(v1)**2), order='F')]
        id1 = ''.join(s)

        s = [str(int(i)) for i in np.sort(u)]
        location = [''.join(s)]

        #print(id0, id1, W1)

        if id0 not in self.motifs.keys():
            self.motifs[id0] = {id1: {}}
            self.motifs[id0][id1]['count'] = 1
            self.motifs[id0][id1]['W'] = W1
            self.motifs[id0][id1]['v'] = v1
            self.motifs[id0][id1]['location'] = {self.current_filename : [location]}

        else:
            if id1 not in self.motifs[id0].keys():
                for key, val in self.motifs[id0].items():
                    if self.is_isomorphic(W1, v1, val['W'], val['v']):
                        for k, v in self.motifs[id0][key]['location'].items():
                            if self.current_filename == k and location in v:
                                return
                        self.motifs[id0][key]['count'] += 1
                        if self.current_filename in self.motifs[id0][key]['location'].keys():
                            self.motifs[id0][key]['location'][self.current_filename].append(location)
                        else:
                            self.motifs[id0][key]['location'][self.current_filename] = location
                        return

                self.motifs[id0][id1] = {}
                self.motifs[id0][id1]['count'] = 1
                self.motifs[id0][id1]['W'] = W1
                self.motifs[id0][id1]['v'] = v1
                self.motifs[id0][id1]['location'] = {self.current_filename : [location]}
            else:
                for k, v in self.motifs[id0][id1]['location'].items():
                    if self.current_filename == k and location in v:
                        return
                self.motifs[id0][id1]['count'] += 1
                if self.current_filename in self.motifs[id0][id1]['location'].keys():
                    self.motifs[id0][id1]['location'][self.current_filename].append(location)
                else:
                    self.motifs[id0][id1]['location'][self.current_filename] = location

    def is_isomorphic(self, W1, v1, W2, v2):

        N = len(v1)
        if not np.sum(W1) == np.sum(W2):
            return False

        perms = list(permutations(range(N)))
        for ind in perms:
            #print(ind, v1.shape, v1, type(v1), type(ind))
            ind = np.array(ind)
            #print(v1[ind])
            v_test = v1[ind]
            if not (v_test == v2).all():
                continue
            W_test = W1[:,ind]
            W_test = W_test[ind,:]
            if (W_test == W2).all():
                return True
            #print('compare ', W1, W2)

        return False


    def prune_network(self,W,v):

        inputs = np.sum(W, axis = 0)
        outputs = np.sum(W, axis = 1)
        connections = inputs + outputs
        neurons_with_connections = np.where(connections > 0)[0]
        W = W[:, neurons_with_connections]
        W = W[neurons_with_connections, :]
        v = v[neurons_with_connections]
        print('neurons_with_connections', len(neurons_with_connections))
        return W, v


    def print_motif_list(self):

        short_ids = sorted(list(self.motifs.keys()))
        long_ids = []
        for s in short_ids:
            long_ids.append(self.motifs[s].keys())

        for s, l in zip(short_ids, long_ids):
            print('\nShort ID:', s, '\t(Neuron types: Inh=0, Exc=1, Out=2)')
            print('Long ID:\t(Rounded weights: if w > 0, is 1)')
            for lid in sorted(l, key=lambda k : -self.motifs[s][k]['count']):
                print('-->', lid, '| c =', self.motifs[s][lid]['count'])
