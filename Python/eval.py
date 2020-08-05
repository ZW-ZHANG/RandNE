
import numpy as np

def Precision_Np(Matrix_test, Matrix_train, U, V, Np):
    '''
    Input:
        Matrix_test is n x n testing matrix, may overlap with Matrix_train
        Matrix_train is n x n training matrix
        U/V are content/context embedding vectors
        Np: returns Precision@Np for pairwise similarity 
    Output:
        Precision@Np
    '''
    
    N = U.shape[0]
    if N >= 30000:
        raise Warning('Network too large, may consume heavy memory and too slow. Sampling suggested.')
    Sim = U.dot(V.T)
    temp_row, temp_col = np.nonzero(Sim)
    temp_value = Sim[temp_row,temp_col]
    temp_choose = np.logical_and(np.array(Matrix_train[temp_row,temp_col])[0] == 0, temp_row != temp_col)
    temp_row, temp_col, temp_value = temp_row[temp_choose], temp_col[temp_choose], temp_value[temp_choose]
    temp_index = np.argsort(temp_value)[::-1]    # need to be accelerated
    if len(temp_index) < Np: 
        raise ValueError('Np too large!')  
    temp_index = temp_index[: int(Np)]
    temp_row, temp_col = temp_row[temp_index], temp_col[temp_index]
    result = np.array(Matrix_test[temp_row,temp_col])[0] > 0
    result = np.divide(np.cumsum(result > 0), np.array(range(len(result))) + 1)
    return result
    
def AUC(Matrix_test, Matrix_train, U, V, sample_num = 1e6, seed = 0):
    '''
    Input:
        Matrix_test is n x n testing matrix, may overlap with Matrix_train
        Matrix_train is n x n training matrix
        U/V are content/context embedding vectors
        sample_num: number of samples
    Output:
        AUC
    '''
    N = U.shape[0]
    sample_num = int(sample_num)
    np.random.seed(seed)
    pos_row, pos_col = np.nonzero(Matrix_test)
    # get rid of overlapping edges
    pos_choose = np.array(Matrix_train[pos_row,pos_col])[0] == 0
    pos_row = pos_row[pos_choose]
    pos_col = pos_col[pos_choose]
    # sampling
    pos_choose = np.random.randint(len(pos_row),size = sample_num)
    pos_row = pos_row[pos_choose]
    pos_col = pos_col[pos_choose]
    pos_value = np.sum(np.multiply(U[pos_row,:], V[pos_col,:]), axis = 1)
    
    neg_num = int(2 * sample_num) 
    while True:
        neg_row = np.random.randint(N,size = neg_num)
        neg_col = np.random.randint(N,size = neg_num)
        neg_choose = np.logical_and(np.array(Matrix_train[neg_row, neg_col])[0] == 0,np.array(Matrix_test[neg_row,neg_col])[0] == 0)
        neg_row = neg_row[neg_choose]
        neg_col = neg_col[neg_choose]
        if len(neg_row) >= sample_num:
            neg_row = neg_row[:sample_num]
            neg_col = neg_col[:sample_num]
            break
        neg_num = int(neg_num + sample_num)  # add more sampling
        raise Warning('Can be accelerated')
        
    neg_value = np.sum(np.multiply(U[neg_row,:], V[neg_col,:]), axis = 1)
    
    return np.mean(pos_value > neg_value)
