from cosmopower import cosmopower_NN
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec



#parameters and data filepaths
PARAM = "Data/BB_params_dict.npz"
DATA = "Data/BB_log10S_dict.npz"

TESTD = "Data/BB_log10Stest_dict.npz"
TESTP = "Data/BB_paramstest_dict.npz"

filename_saved = 'cp_NN_BB'


#load dicts
training_params = np.load(PARAM)
training_Data = np.load(DATA)

testing_params = np.load(TESTP)
testing_features = 10.**(np.load(TESTD)['features'])

#instantiation of cosmopower_NN

ell_range = training_Data['modes']
training_features = training_Data['features']

#parameters of the model
model_params = ['ns', 
                'ombh2', 
                'As', 
                'Alens', 
                'ombch2', 
                'H0', 
                'r', 
                'tau']

cp_nn = cosmopower_NN(parameters=model_params,
                     modes= ell_range,
                     n_hidden= [512, 512, 512, 512],
                     verbose=True
                     )

#training
cp_nn.train(training_parameters=training_params,
              training_features=training_features,
              filename_saved_model=filename_saved,
              validation_split=0.1,
              learning_rates=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
              batch_sizes=[1024, 1024, 1024, 1024, 1024],
              gradient_accumulation_steps= [1, 1, 1, 1, 1],
              patience_values=[100, 100, 100, 100, 100],
              max_epochs = [1000, 1000, 1000, 1000, 1000]
              )


cp_nn = cosmopower_NN(restore=True,
                      restore_filename=filename_saved)
                      
predicted_testing_spectra = cp_nn.ten_to_predictions_np(TESTP)

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(50, 10))

for i in range(3):
  pred = predicted_testing_spectra[i]
  true = testing_features[i]
  ax[i].semilog(ell_range, true, 'blue', label = 'Original')
  ax[i].semilog(ell_range, pred, 'red', label = 'NN', linestyle='--')
  ax[i].set_xlabel('$l$', fontsize='x-large')
  ax[i].set_ylabel('BB')
  ax[i].legend(fontsize=15)
plt.show()
fig.savefig('testing.png')










                      
                      

