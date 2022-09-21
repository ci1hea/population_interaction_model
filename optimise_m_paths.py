import numpy as np
import pandas as pd
import os
import geopandas as gpd
from scipy import stats
import scipy.optimize

import time
import powerlaw
import pickle5 as pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import attractivity_modelling
import fractal_working
import theta_function
import optimise_for_theta


#Pickling functions
def save_obj(obj, name ):
    with open('resources/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)# pickle.HIGHEST_PROTOCOL) - doesn't work for this code

#Unpickling the data
def load_obj(name):
    with open('resources/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

# Input data
lsoa_data = load_obj("newdata_lsoa_data")
sheff_shape, income_params, edu_counts, edu_ratios = lsoa_data['sheff_lsoa_shape'], lsoa_data['income_params'], lsoa_data['edu_counts'], lsoa_data['edu_ratios']

# k
comp_ratio = np.load("resources/newdata_companyhouse.npy")

# distances
paths_matrix = load_obj("newdata_ave_paths")
# removes all 0s not on the diag
paths_matrix[paths_matrix==0] = 1
paths_matrix[np.diag_indices_from(paths_matrix)] = 0

# bus freq paths
stoproute = pd.read_csv('resources/stoproute_withareacodes.csv')
lsoa_list = pd.read_csv("resources/E47000002_KS101EW.csv")['lsoa11cd']
route_freqs = pd.read_csv('resources/Bus_routes_frequency.csv', usecols= ["line","average"]).astype(str)
m_paths = optimise_for_theta.bus_adjacency(stoproute, lsoa_list, route_freqs)


attractivity_avg = optimise_for_theta.median_attractivity(edu_ratios, income_params)

#population amplification
pop = np.asarray(edu_counts).reshape((len(edu_counts), 1))

pop = np.matmul(pop, pop.transpose())

#connectivity matrix
attractivity_product = np.matmul(attractivity_avg, attractivity_avg.transpose())
attractivity_product = np.multiply(attractivity_product, comp_ratio)

#ensure 0 on diagonal?
connectivity = np.divide(attractivity_product, np.power(paths_matrix, m_paths))
connectivity[np.where(np.isinf(connectivity))[0], np.where(np.isinf(connectivity))[1]] = 0
connectivity[np.diag_indices_from(connectivity)] = 0

from scipy.optimize import minimize

# objective function - sum of connectivity
def m_opt(m):

    m = np.reshape(m,(853,853))
    f = np.divide(attractivity_product, np.power(paths_matrix, m)) # needs removing of inf/nan
    f[np.where(np.isinf(f))[0], np.where(np.isinf(f))[1]] = 0
    f[np.diag_indices_from(f)] = 0
    f1 = - np.sum(np.sum(f))

    return f1

ineq_cons = {'type': 'ineq',
              'fun' : lambda m: np.array([1.05 * np.sum(np.sum(m_paths)) - np.sum(np.sum(m)),   #A1*sum M - sum m
                                          np.sum(np.sum(m)) - 0.95 * np.sum(np.sum(m_paths))])} #sum m - A2*sum M
              #'jac' : lambda x: np.array([[-1.0],
                                          #[1.0]])}
eq_cons = {'type': 'eq',
            'fun' : lambda m: np.array(np.median(m))}
            #'jac' : lambda x: np.array(0)}  #??

x0 = m_paths.copy()
res = minimize(m_opt, x0, method='SLSQP', #jac=opt_der,
            constraints=[eq_cons, ineq_cons], options={'ftol': 1e-9, 'disp': True})
            #bounds=bounds)
# may vary

print(res.x)
