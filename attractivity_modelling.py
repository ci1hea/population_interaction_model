#Importing packages
import numpy as np

from scipy import stats
from sklearn.preprocessing import minmax_scale


def attractivity(shape, population, income, education):
    
    """
    Attractivity modelling 
    Counts used are those we have data for in terms of income. The same counts are then used in education sampling. 
    """
    bounds = [0, 5000, 10000, 15000, 20000, 30000, 40000, 50000, 60000, 150000] #income bounds
    bounds = minmax_scale(bounds)
    
    
    counts=[]
    edu_counts = []
    edu_ratios = []
    b_params = np.zeros((len(shape), 4))

    
    
    for i in range(len(shape)):
        
        #Income
        count = population['KS101EW0001'].values[i]
        count = count * income.iloc[i]['rank_1':'rank_9'].values
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
        
        edu = education.iloc[i][[2,3,4,5,6,7]].values.astype(int)
        levels = [edu[0], edu[1]+edu[2], edu[3] + edu[4], edu[5]]
        edu = sum(education.iloc[i][[2,3,4,5,6,7]].values.astype(int))
        edu_ratios.append(np.divide(levels,edu))
        
    
    return b_params, edu_counts, edu_ratios
  
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