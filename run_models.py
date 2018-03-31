import numpy as np
from parameters import *
import model_elimination
import sys

task = 'DMRS90'

for j in range(0,100,1):
    save_fn = 'elim_high_noise_' + task + '_' + str(j) + '.pkl'
    updates = {'trial_type': task, 'save_fn': save_fn}
    update_parameters(updates)


    # Keep the try-except clauses to ensure proper GPU memory release
    try:
        # GPU designated by first argument (must be integer 0-3)
        try:
            print('Selecting GPU ',  sys.argv[1])
            assert(int(sys.argv[1]) in [0,1,2,3])
        except AssertionError:
            quit('Error: Select a valid GPU number.')

        # Run model
        model_elimination.main(sys.argv[1])
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt')
