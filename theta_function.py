import numpy as np
import pandas as pd

#Imports

#list of msoas and adjacenct lsoas - for the whole country
pclookup = pd.read_csv("resources/PCD_OA_LSOA_MSOA_LAD_FEB20_UK_LU.csv", encoding = "ISO-8859-1", low_memory=False)
pclookup = pclookup[['lsoa11cd','msoa11cd']].copy().drop_duplicates()

#importing the company data to get the lsoa-msoa for Sheffield City Region
comp_data = pd.read_csv("resources/newdata_companyhouse.csv")


def convert_to_msoa(data):

    data_df = pd.DataFrame(data)

    data_df['lsoa11cd'] = comp_data['lsoa11cd']
    data_row = pclookup.merge(data_df, left_on='lsoa11cd', right_on='lsoa11cd', how='right')
    data_row = data_row.groupby('msoa11cd', as_index=False).sum()
    data_trans = data_row.drop('msoa11cd', axis=1).T

    data_trans['lsoa11cd'] = comp_data['lsoa11cd']
    data_col = pclookup.merge(data_trans, left_on='lsoa11cd', right_on='lsoa11cd', how='right')
    data_col = data_col.groupby('msoa11cd', as_index=False).sum()
    data_col = data_col.drop('msoa11cd',axis=1).T

    data_msoa = data_col.to_numpy()
    data_msoa[np.where(np.isinf(data_msoa))[0], np.where(np.isinf(data_msoa))[1]] = 0


    return data_msoa


def loop_theta(connectivity, pop, commute_matrix, low_bound, high_bound, step):
    prod_max = 0
    theta = np.arange(low_bound, high_bound, step)
    prod_F = np.zeros(len(theta))

    #normalising the commuter data/row
    commute_matrix_norm = commute_matrix/commute_matrix.sum(axis=1)[:,None]
    commute_matrix_norm[np.isnan(commute_matrix_norm)]=0


    for i in range(len(theta)):

         #adjacency matrix
         adjacency = np.zeros_like(connectivity)
         adjacency[np.where(connectivity>theta[i])] = 1

         #assuming population amplificator is defined in main code
         adjacency = np.multiply(adjacency,pop)

         #convert data to msoa2msoa - assume lsoas and msoas already sorted
         adjacency_msoa = convert_to_msoa(adjacency)

         #normalising the data/row
         A = adjacency_msoa/adjacency_msoa.sum(axis=1)[:,None]
         A[np.isnan(A)] = 0

         #Frobenius product
         prod_F[i] = np.sum(np.multiply(A, commute_matrix_norm))

         if prod_F[i] > prod_max:
             prod_max = prod_F[i]
             A_sim_norm = A
             A_sim = adjacency_msoa
             theta_fin = theta[i]


    return theta_fin, theta, prod_F, A_sim
