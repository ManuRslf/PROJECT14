import matplotlib
import os
from matplotlib import pyplot as plt
import numpy as np
import camb


FILEPATH = "Data/TData_BB.npy"


def push_Data(filepath, data):
  #If the file exists
  if os.path.exists(filepath):
    previous_DATA = np.load(filepath)
    new_DATA = np.vstack((previous_DATA, data))
    np.save(filepath, new_DATA)
  else:
    np.save(filepath, data)



def training_DATAf(params, filepath):
  #This function sets up with one massive neutrino and helium set using BBN consistency
  pars = camb.set_params(H0=params[0], ombh2=params[6], omch2=params[7], mnu=0.06, omk=0, tau=params[3], As=params[5], Alens=params[1], ns=params[4], r=params[2])
  #calculate results for these parameters
  results = camb.get_results(pars)

  #get dictionary of CAMB power spectra
  BB = results.get_lensed_scalar_cls(lmax=2400, CMB_unit='muK')[:, 2]
  
  push_Data(filepath, BB)


    











