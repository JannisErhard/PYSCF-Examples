import numpy as np

def Add_Block_Matrices(Paa, Pbb):
    Pab = np.zeros(Paa.shape)
    Pba = np.zeros(Paa.shape)
    PE = np.concatenate((np.concatenate((Paa, Pab), axis=1), np.concatenate((Pba, Pbb), axis=1)), axis=0) 
    return PE

def expand_matrix(P):
    Paa = P
    Pbb = P
    Pab = np.zeros(P.shape)
    Pba = np.zeros(P.shape)
    PE = np.concatenate((np.concatenate((Paa, Pab), axis=1), np.concatenate((Pba, Pbb), axis=1)), axis=0) 
    return PE
