# A sample run on the BlogCatalog Dataset

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

import RandNE
from eval import Precision_Np, AUC

if __name__ == '__main__':

    print('---loading dataset---')
    data = pd.read_csv('BlogCatalog.csv')      
    data = np.array(data) - 1                       # change index from 0
    N = np.max(np.max(data)) + 1
    A = csr_matrix((np.ones(data.shape[0]), (data[:,0],data[:,1])), shape = (N,N))
    A += A.T
    
    # Common parameters
    d = 128
    Ortho = 0
    seed = 0
    
    print('---calculating embedding---')
    # embedding for adjacency matrix for reconstruction
    q = 2
    weights = [1,0.1,0.001]
    U_list = RandNE.Projection(A, q, d, Ortho, seed)
    U = RandNE.Combine(U_list, weights)
    print('---evaluating---')
    #prec = Precision_Np(A, csr_matrix((N,N)), U, U, 1e6)
    #print(prec)
    auc = AUC(A, csr_matrix((N,N)), U, U, 1e6)
    print(auc)
    
    # embedding for transition matrix for classification
    q = 3
    weights = [1,1e2,1e4,1e5]
    A_tran = normalize(A, norm = 'l1', axis = 1)
    U_list = RandNE.Projection(A_tran,q,d,Ortho,seed)
    U = RandNE.Combine(U_list,weights)
    # normalizing
    U = normalize(A, norm = 'l2', axis = 1)
    # Some Classification method, such as SVM in http://leitang.net/social_dimension.html
    





