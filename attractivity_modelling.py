#Importing packages
import numpy as np

from scipy import stats
from sklearn.preprocessing import minmax_scale

#Deleted attractivity function because it was already used in lsoa_import.py

def attractivity_sampler(oa, edu_ratios, income_params):
    """
    Parameters
    ----------
    oa : Integer of oa

    Returns
    -------
    attractivity

    """
    edu = np.random.choice(4, size = 1, p=edu_ratios[oa]) #where p values are effectively the ratio of people with a given education level
    income = stats.beta.rvs(income_params[oa, 0], income_params[oa, 1], loc = income_params[oa, 2], scale = income_params[oa, 3], size=1)

    attractivity = np.power(income, -edu)

    return attractivity
# returns one random attractivity sample


def attractivity_median_sampler(oa, edu_ratios, income_params, size):
    """
    Parameters
    ----------
    oa : Integer of oa

    Returns
    -------
    attractivity

    """
    edu = np.random.choice(4, size = size, p=edu_ratios[oa]) #where p values are effectively the ratio of people with a given education level
    income = stats.beta.rvs(income_params[oa, 0], income_params[oa, 1], loc = income_params[oa, 2], scale = income_params[oa, 3], size=size)

    attractivity = np.power(income, -edu)

    return np.median(attractivity)


#returns median values/oa - need to loop through to construct the avg attractivity matrix, shaped (853,)
#loop is directional - not a symmetrical matrix
