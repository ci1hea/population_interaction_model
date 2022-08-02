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


#Pickling functions
def save_obj(obj, name ):
    with open('resources/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)# pickle.HIGHEST_PROTOCOL) - doesn't work for this code

#Unpickling the data
def load_obj(name):
    with open('resources/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)




def median_attractivity(edu_ratios, income_params): #,fit = None):

    """
    Average individual attractivity / lsoa (taken as a sample of 1000 ppl)
    Sample is directinal - matrix not symmetrical
    """

    attractivity = np.zeros((len(income_params)))
    size = 10000

    for i in range(len(income_params)):
        attractivity[i] = attractivity_modelling.attractivity_median_sampler(i, edu_ratios, income_params, size)

    attractivity = attractivity.reshape((len(attractivity),1))

    # attractivity1 = np.zeros((len(income_params)))
    # attractivity2 = np.zeros((len(income_params)))
    # for i in range(len(income_params)): #Loop across  OAs
    #     attractivity1[i] = attractivity_modelling.attractivity_sampler(i, edu_ratios, income_params)
    #     attractivity2[i] = attractivity_modelling.attractivity_sampler(i, edu_ratios, income_params)
    #
    # if fit != None:
    #     all_attractivity = np.concatenate((attractivity1, attractivity1) , axis=0)
    #     attractivity_powerlaw = powerlaw.Fit(all_attractivity, verbose=False)
    #     alpha = attractivity_powerlaw.alpha
    #     xmin = attractivity_powerlaw.xmin
    #     return attractivity1, attractivity2, alpha, xmin
    # else:
    #     return attractivity1, attractivity2

    return attractivity


## Unchanged
def euclidean_dists_fun(sheff_shape):
    """
    Dummy distances function
    """
    euclidean_dists = []
    point1s = []
    centroids = sheff_shape.centroid
    for i in range(len(sheff_shape)):
        euclidean_dists.append(centroids.distance(centroids[i]).values)
        point1s.append((centroids.x[i], centroids.y[i]))
    all_coords = pd.DataFrame(point1s, columns = ['x-coord', 'y-coord'])

    #generating path matrix
    paths_matrix = np.column_stack(euclidean_dists)

    #median path distances
    paths = np.concatenate(euclidean_dists, axis=0)
    paths = paths[paths != 0]
    med_paths = sorted(paths)
    med_paths = int(med_paths[int(len(med_paths)/2)])

    return euclidean_dists, all_coords, paths_matrix, med_paths


## Unchanged
def fractal_dimension(coords_data):
    """
    Graph may require some intuition to fit the linear regression through certain points
    """
    rangex = coords_data['x-coord'].values.max() - coords_data['x-coord'].values.min()
    rangey = coords_data['y-coord'].values.max() - coords_data['y-coord'].values.min()
    L = int(max(rangex, rangey)) # max of x or y distance

    r = np.array([ L/(2.0**i) for i in range(5,0,-1) ]) #Create array of box lengths
    N = [ fractal_working.count_boxes( coords_data, ri, L ) for ri in r ] #Non empty boxes for each array of box lenghts

    popt, pcov = scipy.optimize.curve_fit( fractal_working.f_temp, np.log( 1./r ), np.log( N ) )
    A, Df = popt #A lacunarity, Df fractal dimension


    # fig, ax = plt.subplots(1,1)
    # ax.plot(1./r, N, 'b.-')
    # ax.plot( 1./r, np.exp(A)*1./r**Df, 'g', alpha=1.0 )
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_aspect(1)
    # ax.set_xlabel('Box Size')
    # ax.set_ylabel('Number of boxes')

    # #Playing around with data points to use
    # Y = np.log( N )
    # X = np.log( 1./r )
    # T = np.vstack((Y,X,np.ones_like(X))).T

    # df = pd.DataFrame( T, columns=['N(r)','Df','A'] )
    # Y = df['N(r)']
    # X = df[['Df','A']]
    # result = OLS( Y, X ).fit()
    # result.summary()
    return Df


#Monte Carlo function with theta as parameter -------------------------
def opt_theta_funct(m_paths, lsoa_data, paths_matrix, comp_ratio, commute_matrix):

    startt = time.time()
    time_log = []


    sheff_shape, income_params, edu_counts, edu_ratios = lsoa_data['sheff_lsoa_shape'], lsoa_data['income_params'], lsoa_data['edu_counts'], lsoa_data['edu_ratios']

    #Constants
    base_m = 1


    #dummy distances
    euclidean_dists, centroids, centroid_paths_matrix, med_paths = euclidean_dists_fun(sheff_shape)
    ## eps = 1200 #med_paths 1200 is the median diameter of the lsoa polygons


    #fractal dimension
    Df = fractal_dimension(centroids)

    #create data structures
    UrbanY = []
    #edges = np.zeros((len(sheff_shape), len(sheff_shape), n))

    # for i in range(n):
    #
    #
    #     #Sample attractivities
    #     attractivity1, attractivity2, alpha, xmin = sample_attractivities(edu_ratios, income_params, 1)
    alpha = 1.45653 #mean fixed alpha from 1000 runs
    #
    #     theta = np.exp(np.log(xmin**2) - (base_m*np.log(eps)))
    dc = base_m * (alpha - 1)
    #
    #
    #     #connectivity matrix
    #     attractivity1 = attractivity1.reshape((len(attractivity1),1))
    #     attractivity2 = attractivity2.reshape((len(attractivity2),1))
    #


    ## avg attractivity
    attractivity_avg = median_attractivity(edu_ratios, income_params)# 1)  ## no alpha and xmin returned

    #population amplification
    pop = np.asarray(edu_counts).reshape((len(edu_counts), 1))


    # if is_shuffled is None:
    pop = np.matmul(pop, pop.transpose())
    # else:
    #     attractivity1[east_inds] = attractivity1[income_inds]
    #     attractivity2[east_inds] = attractivity2[income_inds]
    #
    #     pop[east_inds] = pop[income_inds]
    #     pop = np.matmul(pop, pop.transpose())

    ##need commute_matrix - nvm passed as variable

    #connectivity matrix
    attractivity_product = np.matmul(attractivity_avg, attractivity_avg.transpose())
    attractivity_product = np.multiply(attractivity_product, comp_ratio)

       #ensure 0 on diagonal?
    connectivity = np.divide(attractivity_product, np.power(paths_matrix, m_paths))
    connectivity[np.where(np.isinf(connectivity))[0], np.where(np.isinf(connectivity))[1]] = 0
    connectivity[np.diag_indices_from(connectivity)] = 0

    low_bound = 0
    high_bound = 2
    step = 0.001

    theta_opt, thetas, prod_Fs, adjacency = theta_function.loop_theta(connectivity, pop, commute_matrix, low_bound, high_bound, step)

    #edges[:,:] = adjacency  #not sure what to do with this just removed the i from the intial loop


    if Df <= dc:
        eta = ((-5/6) * Df) + dc
    else:
        eta = (Df/6)

        #activity
        # paths_matrix_n = (paths_matrix - paths_matrix.min()) / (paths_matrix.max() - paths_matrix.min()) +1
    activity_lsoa = np.power(paths_matrix, eta)
    activity_msoa = theta_function.convert_to_msoa(activity_lsoa)

    #activity[np.where(np.isinf(activity))[0], np.where(np.isinf(activity))[1]] = 0

    UrbanY.append( 0.5 * np.sum(np.multiply(adjacency, activity_msoa)) )
        # UrbanY.append( 0.5 * np.sum(adjacency))


    #Creating network data
    # edge_freq = np.count_nonzero(edges, axis = 2) / n
    # edge_width = np.sum(edges, axis = 2) / n

    endt = time.time()
    print("Time for this n run through is: "+str(endt-startt))


    time_log.append(endt-startt)
    total_time = sum(time_log)
    print("Total run time is: " + str(total_time))


    return UrbanY, theta_opt, thetas, prod_Fs, adjacency


#Running Monte Carlo ----------------------------------------

# import multiprocessing
if __name__ == '__main__':   #- no need to put in multiprocessing

        #imports
    lsoa_data = load_obj("newdata_lsoa_data")

        #import scipy.io as sio
        # mldata = sio.loadmat(r'G:\My Drive\PIN_Productivity_Project\Scripts\optimisedpaths.mat')#import new paths

    m_paths = np.ones(np.shape(np.load("resources/newdata_m_paths_bus.npy"))) # np.load("resources/newdata_m_paths_bus.npy") #m values for from buses

        # np.random.shuffle(m_paths) #shuffled bus service

    comp_ratio = np.load("resources/newdata_companyhouse.npy") #m values for from buses

        #n = 1000 #number of monte carlo repeats
        # ms = [1]

    ##converting commuter matrix - data used directionally
    commute = pd.read_csv("resources/SCR_Commute_msoa_to_msoa.csv")
    comm_matrix = (
        commute
        .pivot_table(index="O_Code", columns="D_Code")#, values="Commuters", aggfunc=len)
        .fillna(0)
        .astype(int)
    )
    commute_matrix = comm_matrix.to_numpy()
    commute_matrix[np.diag_indices_from(commute_matrix)] = 0

    # commute_matrix = commute_array + commute_array.transpose()
    # commute_matrix[np.diag_indices_from(commute_matrix)] = 0

        # -----------------------------------------
        # Normal paths
        # -----------------------------------------


    t1 = time.time()
    ##no_scripts = multiprocessing.cpu_count()



    paths_matrix = load_obj("newdata_ave_paths")
    # args_normal = []
    #
    #     # for i in range(len(ms)):
    # args_normal.append((m_paths, lsoa_data, paths_matrix, comp_ratio, commute_matrix)
    #
    # with multiprocessing.Pool(processes=no_scripts) as pool:
        # output = pool.starmap(monte_carlo_runs, args_normal)

    UrbanY, theta_opt, thetas, prod_Fs, adjacency = opt_theta_funct(m_paths, lsoa_data, paths_matrix, comp_ratio, commute_matrix)
# for i in range(len(output)):
#     UrbanYs.append(output[i][0])
#     edge_freqs.append(output[i][1])
#     edge_widths.append(output[i][2])
#
    normal = {
        "UrbanY": UrbanY,
        "theta_opt": theta_opt,
        "theta_all": thetas,
        "prod_F_all": prod_Fs,
        "adjacency": adjacency
        }

    save_obj(normal, "normal_layout_optimise_for_theta5")
    #pickle.dump(normal,"normal_layout_optimise_for_theta.pkl")
    # with open("normal_layout_optimise_for_theta","wb") as f:
    #     pickle.dump(normal, f)

    print(time.time()-t1)
