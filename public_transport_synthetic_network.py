"""
Synthetic Sheffield - USING NEWDATA - NEEDS
1) a newdata_lsoa pickle in this folder (done)
2) an newdata_ave-paths (waiting for at least 1 iteration of newdata_all_paths
3) G:\My Drive\PIN_Productivity_Project\Scripts\optimisedpaths.mat (unchanged?)
"""

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

#my scripts
import attractivity_modelling
import fractal_working


#Pickling functions
def save_obj(obj, name ):
    with open('resources/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
#save_obj(lsoa_dist, "lsoa_data")
def load_obj(name ):
    with open('resources/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)






#Monte carlo functions -------------------------------------------------------
def sample_attractivities(edu_ratios, income_params, fit = None):

    attractivity1 = np.zeros((len(income_params)))
    attractivity2 = np.zeros((len(income_params)))
    for i in range(len(income_params)): #Loop across  OAs
        attractivity1[i] = attractivity_modelling.attractivity_sampler(i, edu_ratios, income_params)
        attractivity2[i] = attractivity_modelling.attractivity_sampler(i, edu_ratios, income_params)

    if fit != None:
        all_attractivity = np.concatenate((attractivity1, attractivity1) , axis=0)
        attractivity_powerlaw = powerlaw.Fit(all_attractivity, verbose=False)
        alpha = attractivity_powerlaw.alpha
        xmin = attractivity_powerlaw.xmin
        return attractivity1, attractivity2, alpha, xmin
    else:
        return attractivity1, attractivity2


def euclidean_dists_fun(sheff_shape):
    """
    Dummy distances function.
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


#Shuffling data
def paths_shuffle(shape, income_params):
    """
    Returns the mean of each oa's income distribution'
    """
    means = []
    for oa in range(len(shape)):
        means.append(stats.beta.mean(income_params[oa, 0], income_params[oa, 1], loc = income_params[oa, 2], scale = income_params[oa, 3]))
    return means


#Bus adjacency
def bus_adjacency(stoproute,lsoa_list,route_freqs):
    # Create matrix that combines location data and route frequencies
    combine = pd.merge(stoproute,route_freqs,on='1')
    combine = combine.drop_duplicates(['1','naptan_sto'])
    combine = combine.rename(columns={'geo_code':'lsoa11cd'})

    # Create adjacency matrix LSOA x route
    bstopfreq = combine[['lsoa11cd','naptan_sto','1','average']]
    adj = pd.pivot(bstopfreq,index=["lsoa11cd","naptan_sto"], columns="1", values="average").fillna(0)
    adj = adj.astype(float)
    adj = adj.groupby(level="lsoa11cd").mean()
    bus2route = pd.merge(lsoa_list, adj, how='left',on='lsoa11cd').set_index('lsoa11cd')

    #Adjacency matrix LSOA x LSOA
    bus2route = np.array(bus2route)
    bus2routeT = bus2route.transpose()
    lsoa2lsoa = np.dot(bus2route,bus2routeT)**0.5
    lsoa2lsoa[np.diag_indices_from(lsoa2lsoa)] = 0

    lsoa2lsoa = pd.DataFrame(lsoa2lsoa)

    #m values created
    m_paths = lsoa2lsoa.copy()
    m_paths[m_bus>0]=np.log10(m_bus[m_bus>0])
    m_paths=1-(m_bus/np.max(np.max(m_bus)))
    return m_paths

#Monte Carlo function---------------------------------------------------------
def monte_carlo_runs( n, lsoa_data, paths_matrix, comp_ratio, is_shuffled=None):
    """

    Parameters
    ----------
    m_paths : numpy array
        DESCRIPTION.
    n : TYPE
        DESCRIPTION.
    lsoa_data : TYPE
        DESCRIPTION.
    paths_matrix : TYPE
        DESCRIPTION.
    is_shuffled : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    UrbanY : List of n values
        Output values to be interpreted.
    edge_freq : List of n numpy arrays
        How often an edge ocurrs between an lsoa pair.
    edge_width : List of n numpy arrays
        Ave. strength of an lsoa edge

    """
    startt = time.time()
    time_log = []


    sheff_shape, income_params, edu_counts, edu_ratios = lsoa_data['sheff_lsoa_shape'], lsoa_data['income_params'], lsoa_data['edu_counts'], lsoa_data['edu_ratios']

    #Constants
    base_m = 1 #mean of m=0.46 from bus network


    #dummy distances
    euclidean_dists, centroids, centroid_paths_matrix, med_paths = euclidean_dists_fun(sheff_shape)
    eps = 1200 #med_paths 1200 is the median diameter of the lsoa polygons

    #Creating m
    stoproute = pd.read_csv('stoproute_withareacodes.csv')
    lsoa_list = pd.read_csv("E47000002_KS101EW.csv")['lsoa11cd']
    route_freqs = pd.read_excel('Bus_routes_frequency final.xlsx','raw data',usecols= ["1","average"]).astype(str)
    m_paths = bus_adjacency(stoproute,lsoa_list,route_freqs)


    if is_shuffled is None:
        pass
    else:
        east_inds = centroids["x-coord"].argsort().values  #low to high sort

        income_inds = np.argsort(paths_shuffle(sheff_shape, income_params)) #returns indices that would sort the array, low to high



    #fractal dimension
    Df = fractal_dimension(centroids)

    #create data structures
    UrbanY = []
    edges = np.zeros((len(sheff_shape), len(sheff_shape), n))

    for i in range(n):


        #Sample attractivities
        attractivity1, attractivity2, alpha, xmin = sample_attractivities(edu_ratios, income_params, 1)
        #alpha = 1.45653 #mean fixed alpha from 1000 runs

        theta = np.exp(np.log(xmin**2) - (base_m*np.log(eps)))
        dc = base_m * (alpha - 1)


        #connectivity matrix
        attractivity1 = attractivity1.reshape((len(attractivity1),1))
        attractivity2 = attractivity2.reshape((len(attractivity2),1))

        #population amplification
        pop = np.asarray(edu_counts).reshape((len(edu_counts), 1))


        if is_shuffled is None:
            pop = np.matmul(pop, pop.transpose())
        else:
            attractivity1[east_inds] = attractivity1[income_inds]
            attractivity2[east_inds] = attractivity2[income_inds]

            pop[east_inds] = pop[income_inds]
            pop = np.matmul(pop, pop.transpose())


        attractivity_product = np.matmul(attractivity1, attractivity2.transpose())
        attractivity_product = np.multiply(attractivity_product, comp_ratio)

        #ensure 0 on diagonal?
        connectivity = np.divide(attractivity_product, np.power(paths_matrix, m_paths))
        connectivity[np.where(np.isinf(connectivity))[0], np.where(np.isinf(connectivity))[1]] = 0
        connectivity[np.diag_indices_from(connectivity)] = 0

        #adjacency matrix
        adjacency = np.zeros_like(connectivity)
        adjacency[np.where(connectivity>theta)] = 1
        adjacency = np.multiply(adjacency, pop) #population amplification factor
        edges[:,:,i] = adjacency


        if Df <= dc:
            eta = ((-5/6) * Df) + dc
        else:
            eta = (Df/6)

        #activity
        # paths_matrix_n = (paths_matrix - paths_matrix.min()) / (paths_matrix.max() - paths_matrix.min()) +1
        activity = np.power(paths_matrix, eta)
        activity[np.where(np.isinf(activity))[0], np.where(np.isinf(activity))[1]] = 0

        UrbanY.append( 0.5 * np.sum(np.multiply(adjacency, activity)) )
        # UrbanY.append( 0.5 * np.sum(adjacency))



    #Creating network data
    edge_freq = np.count_nonzero(edges, axis = 2) / n
    edge_width = np.sum(edges, axis = 2) / n

    endt = time.time()
    print("Time for this n run through is: "+str(endt-startt))


    time_log.append(endt-startt)
    total_time = sum(time_log)
    print("Total run time is: " + str(total_time))
    return UrbanY, edge_freq, edge_width





#------------------------------------------------------------------------------
#
#                   Multiprocessing - Running Monte Carlo
#
#------------------------------------------------------------------------------
import multiprocessing
if __name__ == '__main__':

    #imports
    lsoa_data = load_obj("newdata_lsoa_data")
    import scipy.io as sio
    # mldata = sio.loadmat(r'G:\My Drive\PIN_Productivity_Project\Scripts\optimisedpaths.mat')#import new paths
    m_paths = np.load("resources/newdata_m_paths_bus.npy") #m values for from buses
    # np.random.shuffle(m_paths) #shuffled bus service
    comp_ratio = np.load("resources/newdata_companyhouse.npy") #m values for from buses
    n = 1000 #number of monte carlo repeats
    # ms = [1]

    # -----------------------------------------
    # Normal paths
    # -----------------------------------------


    t1 = time.time()
    no_scripts = multiprocessing.cpu_count()



    paths_matrix = load_obj("newdata_ave_paths")
    args_normal = []

    # for i in range(len(ms)):
    args_normal.append((m_paths, n, lsoa_data, paths_matrix, comp_ratio))

    with multiprocessing.Pool(processes=no_scripts) as pool:
        output = pool.starmap(monte_carlo_runs, args_normal)

    UrbanYs, edge_freqs, edge_widths = [], [], []
    for i in range(len(output)):
        UrbanYs.append(output[i][0])
        edge_freqs.append(output[i][1])
        edge_widths.append(output[i][2])

    normal = {
        "UrbanYs": UrbanYs,
        "edge_freqs": edge_freqs,
        "edge_widths": edge_widths
        }

    save_obj(normal, "normal_layout_"+str(n)+"run_bus")

    print(time.time()-t1)
