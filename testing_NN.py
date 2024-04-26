from cosmopower import cosmopower_NN
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os


#parameters and data filepaths
PARAM = "Data/BB_params_dict.npz"
DATA = "Data/BB_log10S_dict.npz"


TESTD = "Data/BB_log10Stest_dict.npz"
TESTP = "Data/BB_paramstest_dict.npz"

filename_saved = 'cp_NN_BB'


#load dicts
training_Data = np.load(DATA)
testing_params = np.load(TESTP)
testing_features = 10.**(np.load(TESTD)['features'])

#instantiation of cosmopower_NN

ell_range = training_Data['modes']
training_features = training_Data['features']


cp_nn = cosmopower_NN(restore=True,
                      restore_filename=filename_saved)
                      
predicted_testing_spectra = cp_nn.ten_to_predictions_np(testing_params)

pred = predicted_testing_spectra[5]
true = testing_features[5]

plt.figure(figsize=(10, 6))
plt.plot(ell_range, true, 'blue', label='Original')
plt.plot(ell_range, pred, 'red', label='NN', linestyle='--')
plt.xlabel('$l$', fontsize='x-large')
plt.ylabel('BB')
plt.legend(fontsize=15)
plt.show()
plt.savefig('testing.png')