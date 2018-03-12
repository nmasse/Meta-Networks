import numpy as np
from parameters import *
import model
import sys

task = 'DMC'

for j in range(0,1000,2):
    save_fn = task + '_' + str(j) + '.pkl'
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
        model.main(sys.argv[1])
    except KeyboardInterrupt:
        quit('Quit by KeyboardInterrupt')
