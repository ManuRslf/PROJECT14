import numpy as np
import BB

PARAMS_PATH = "Data/D_Params.npy"


#loading the data of random parameters
params = np.load(PARAMS_PATH)


#generating spectrum for each paraneters
for i in range(len(params)):
  BB.training_DATAf(params[i], BB.FILEPATH)
  
print("EVERY SPECTRUM IS GENERATED!\n")






