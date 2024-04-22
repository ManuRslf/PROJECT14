import numpy as np


PATH_PARAMS = "Data/D_Params.npy"
PATH_DATA = "Data/TData_BB.npy"

#size of the testing params
t_size = 80000

#load array generated
parameters_array = np.load(PATH_PARAMS)
data_array = np.load(PATH_DATA)

params_name = ['H0', 'Alens', 'r', 'tau', 'ns', 'As', 'ombh2', 'ombch2']
n_name = len(params_name)

#logspectra
log10_bb = np.log10(data_array[:, 2:])

#transform to dict
linear_parameters_dict = {params_name[i] : parameters_array[:t_size, i] for i in range(n_name)}
lineartesting_parameters_dict = {params_name[i] : parameters_array[t_size:, i] for i in range(n_name)}

l_max = len(data_array[0])

linear_data_BB = { 'modes' : [i for i in range(2, l_max)], 'features' : log10_bb[:t_size]}
lineartesting_data_BB = { 'modes' : [i for i in range(2, l_max)], 'features' : log10_bb[t_size:]}



#save
np.savez("Data/BB_params_dict.npz", **linear_parameters_dict)
np.savez("Data/BB_log10S_dict.npz", **linear_data_BB)

np.savez("Data/BB_paramstest_dict.npz", **lineartesting_parameters_dict)
np.savez("Data/BB_log10Stest_dict.npz", **lineartesting_data_BB)


