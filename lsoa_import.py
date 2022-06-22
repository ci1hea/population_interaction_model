"""
Create LSOA pickles and see if they look like the previous ones
This is split from the all_paths pickle script because that script called osmnx, which when installed, broke this script. If you're smart with environments, you can probably merge into one script.
update on above - a single environment was found, (described in 'environment setup.txt' but all_paths takes FOREVER so still worth splitting out.
new pickle saved as new_lsoa_data.pkl
"""

import numpy as np
import pandas as pd
import os
import geopandas as gpd
from scipy import stats
import scipy.optimize

from time import time
import powerlaw
import pickle5 as pickle
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import minmax_scale


#Pickling functions
def save_obj(obj, name):
    print("About to save it")
    with open('resources/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print("Saved it")
#save_obj(lsoa_dist, "lsoa_data")
def load_obj(name):
    with open('resources/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


#--------------Addressing data ------------------------------------------------
mca_lsoa_shape_path = 'resources/E47000002.shp'
mca_lsoa_pop_path = 'resources/E47000002_KS101EW.csv'
mca_lsoa_income_path = 'resources/SCR_small_area_individual_income.shp'
mca_lsoa_education_path = 'resources/E47000002_KS501EW.csv'


#--------------Loading data ---------------------------------------------------
def import_lsoa(lsoa_shape_path, lsoa_pop_path, lsoa_income_path, lsoa_education_path):
    print("About to do it")


    print("New import is starting")
    """ from sheff_import function in synthetic_network.py
    Function to generate the lsoa_data dictionary used throughout.
    """
    sheff_lsoa = {}
    sheff_lsoa['sheff_lsoa_shape'] = gpd.read_file(lsoa_shape_path)
    sheff_lsoa['sheff_lsoa_pop'] = gpd.read_file(lsoa_pop_path)
    sheff_lsoa['sheff_lsoa_pop']['KS101EW0001'] = pd.to_numeric(sheff_lsoa['sheff_lsoa_pop'][r'Variable: All usual residents; measures: Value']) #Count: All categories:sex
    #need to rename raw data column to KS101EW0001 so that the income_params and edu_count calculations pick it up correctly
    # check that above adds a KS101EW0001 column

    #adding an extra column for GeographyCode based on lsoa11cd
    sheff_lsoa['sheff_lsoa_pop']['GeographyCode'] = sheff_lsoa['sheff_lsoa_pop']['lsoa11cd']
    print("Population import: ", sheff_lsoa['sheff_lsoa_pop'])
    #to check above

    #so, I erred on adding columns to match orig names in case code elsewhere/later that I'm not using for my purposes cares about these headings - can be simplified but costs not a lot

    sheff_lsoa['sheff_lsoa_income'] = gpd.read_file(lsoa_income_path)
    # print("Income import: ", sheff_lsoa['sheff_lsoa_income'])
    sheff_lsoa['sheff_lsoa_income']['lsoa11cd'] = sheff_lsoa['sheff_lsoa_income']['geo_code']
    #adding an extra column for lsoa11cd based on GeographyCode
    # print("Income import: ", sheff_lsoa['sheff_lsoa_income'])

    #WE NEED TO SORT THIS BY
    sheff_lsoa['sheff_lsoa_income'] = sheff_lsoa['sheff_lsoa_income'].sort_values(by='geo_code')
    # print("Income import after sort: ", sheff_lsoa['sheff_lsoa_income'])
    sheff_lsoa['sheff_lsoa_income'] = sheff_lsoa['sheff_lsoa_income'].reset_index(drop=True)
    # print("Income import after reindexing: ", sheff_lsoa['sheff_lsoa_income'])

    #LSOA Income data includes extra LSOA that are not in Sheffield City region, these should be removed.
    ids = sheff_lsoa['sheff_lsoa_income']['lsoa11cd'].isin(sheff_lsoa['sheff_lsoa_pop']['GeographyCode'].values)

    ids = np.where(ids==True)
    sheff_lsoa['sheff_lsoa_income'] = sheff_lsoa['sheff_lsoa_income'].iloc[ids]

    sheff_lsoa['sheff_lsoa_education'] = gpd.read_file(lsoa_education_path)


    print("New import is complete")
    return sheff_lsoa


    """
    to sort each dataset on geo_id after importing

    sense check = matrix size?
    """



def attractivity(shape, population, income, education):
    """next add income_params, edu_count and edu_ratios
    from Attractivity function in attractivity_modelling.py
    Counts used are those we have data for in terms of income. The same counts are then used in education sampling.
    """

    bounds = [0, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 150000] #income bounds
    bounds = minmax_scale(bounds)

    counts=[]
    edu_counts = []
    edu_ratios = []
    b_params = np.zeros((len(shape), 4))


    """
    when we have a new file, check that anything here isn't manual - is it getting the correct number of rows, lsoas etc
    """
    for i in range(len(shape)):

        #Income
        count = population['KS101EW0001'].values[i]
    #check that this label remains
        count = count * income.iloc[i]['Rank_1':'Rank_9'].values

    #old code had no caps for rank headers
        count = count.astype(int)

        x = []
        for j in range(len(count)):
            if j == 0:
                x.append(np.random.uniform(low = 0.001, high = 0.002, size = count[j]))
            else:
                x.append(np.random.uniform(low = bounds[j], high = bounds[j + 1], size = count[j]))

        x = np.concatenate(x, axis = 0)
        counts.append(sum(count)) #Create list of counts to use later

        b_params[i, 0], b_params[i, 1], b_params[i, 2], b_params[i, 3] = stats.beta.fit(x, floc = 0, fscale = 1) #Fitting beta functions


        #Education ---------------------------------------------------------------
        edu_counts.append(population['KS101EW0001'].values[i].astype(int))

        edu = education.iloc[i][[6,7,8,9,10,11]].values.astype(int)#original was 2-7
        levels = [edu[0], edu[1] + edu[2], edu[3] + edu[4], edu[5]]
        edu = sum(education.iloc[i][[6,7,8,9,10,11]].values.astype(int))#original was 2-7
        edu_ratios.append(np.divide(levels,edu))

    return b_params, edu_counts, edu_ratios



sheff_lsoa = import_lsoa(mca_lsoa_shape_path, mca_lsoa_pop_path, mca_lsoa_income_path, mca_lsoa_education_path)
sheff_lsoa['income_params'], sheff_lsoa['edu_counts'], sheff_lsoa['edu_ratios'] = attractivity(sheff_lsoa['sheff_lsoa_shape'], sheff_lsoa['sheff_lsoa_pop'], sheff_lsoa['sheff_lsoa_income'], sheff_lsoa['sheff_lsoa_education'])
# print(sheff_lsoa)

#sheff_lsoa = sheff_import
save_obj(sheff_lsoa,'newdata_lsoa_data')
print("Done it")
