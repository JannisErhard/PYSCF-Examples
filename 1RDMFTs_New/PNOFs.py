import numpy as np

def PNOF2(n_a, n_b, M, N_a, N_b):
    # make Delta and Pi specific to PNOF2
    Delta = np.zeros((M,M))
    Pi = np.zeros((M,M))

    S_F = 0.0
    for i in range(0,N_a):
        S_F += max(1.0-n_a[i],0)
    print(S_F, (1.0-S_F)/S_F)
    S_F = 0.0
    for i in range(N_a,M):
        S_F += n_a[i]
    print(S_F, (1.0-S_F)/S_F)


    
    for i in range(0,M):
        for j in range(0,M):
            if i != j:
                if i < N_a and j < N_b:
                    Delta[i,j] = (max(1.0-n_a[i],0))*(max(1.0-n_b[j],0))
                    Pi[i,j] = np.sqrt(n_a[i]*n_b[j])+np.sqrt((max(1.0-n_a[i],0))*(max(1.0-n_b[j],0)))+n_a[i]*n_b[j]-Delta[i,j]
                if i < N_a and j >= N_b:
                    Delta[i,j] = (max(1.0-n_a[i],0))*(n_b[j])*(1.0-S_F)/S_F
                    Pi[i,j] = np.sqrt(n_a[i]*n_b[j])-np.sqrt((max(1.0-n_a[i],0))*(n_b[j]))+n_a[i]*n_b[j]-Delta[i,j]
                if i >= N_a and j < N_b:
                    Delta[i,j] = (max(1.0-n_b[j],0))*(n_a[i])*(1.0-S_F)/S_F
                    Pi[i,j] = np.sqrt(n_a[i]*n_b[j])-np.sqrt((n_a[i])*(max(1.0-n_b[j],0)))+n_a[i]*n_b[j]-Delta[i,j]
                if i >= N_a and j >= N_b:
                    Delta[i,j] = n_a[i]*n_b[j]
                    Pi[i,j] = n_a[i]*n_b[j]-Delta[i,j]
            else:
                Delta[i,i] = n_a[i]**2
                Pi[i,i] = n_a[i]


    
    TWORDM = (np.zeros((M,M,M,M)), np.zeros((M,M,M,M)), np.zeros((M,M,M,M)))
    # alpha alpha alpha alpha 
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==k and j==l:
                        TWORDM[0][i,j,k,l] += n_a[i]*n_a[j] - Delta[i,j]
                    if i==l and k==j:
                        TWORDM[0][i,j,k,l] += -n_a[i]*n_a[j] + Delta[i,j]
    # alpha beta and beta alpha
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==k and j==l:
                        TWORDM[1][i,j,k,l] += n_a[i]*n_b[j] - Delta[i,j] 
                    if i==j and k==l:
                        TWORDM[1][i,j,k,l] +=  Pi[i,k]
    # beta beta beta beta
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==k and j==l:
                        TWORDM[2][i,j,k,l] += n_b[i]*n_b[j] - Delta[i,j]
                    if i==l and k==j:
                        TWORDM[2][i,j,k,l] += -n_b[i]*n_b[j] + Delta[i,j]
#                    print(TWORDM[2][i,j,k,l])

    return TWORDM
    
def PNOF4(n_a, n_b, M, N_a, N_b):
    # make Delta and Pi specific to PNOF2
    Delta = np.zeros((M,M))
    Pi = np.zeros((M,M))

    S_F = 0.0
    for i in range(0,N_a):
        S_F += max(1.0-n_a[i],0)
    print(S_F, (1.0-S_F)/S_F)
    S_F = 0.0
    for i in range(N_a,M):
        S_F += n_a[i]
    print(S_F, (1.0-S_F)/S_F)


    
    for i in range(0,M):
        for j in range(0,M):
            if i != j:
                if i < N_a and j < N_b:
                    Delta[i,j] += (max(1.0-n_a[i],0))*(max(1.0-n_b[j],0))
                    Pi[i,j] -= np.sqrt((max(1.0-n_a[i],0))*(max(1.0-n_b[j],0)))
                if i < N_a and j >= N_b:
                    Delta[i,j] += (max(1.0-n_a[i],0))*(n_b[j])*(1.0-S_F)/S_F
                    Pi[i,j] -= np.sqrt((max(1.0-n_a[i],0))*n_b[j])*np.sqrt(n_a[i]-n_b[j]+(max(1.0-n_a[i],0)*n_b[j]/S_F)) 
                if i >= N_a and j < N_b:
                    Delta[i,j] += (max(1.0-n_b[j],0))*(n_a[i])*(1.0-S_F)/S_F
                    Pi[i,j] -= np.sqrt((max(1.0-n_a[j],0))*n_b[i])*np.sqrt(n_a[j]-n_b[i]+(max(1.0-n_a[j],0)*n_b[i]/S_F))
                if i >= N_a and j >= N_b:
                    Delta[i,j] += n_a[i]*n_b[j]
                    Pi[i,j] += np.sqrt(n_a[i]*n_b[j])
            else:
                Delta[i,i] = n_a[i]**2
                Pi[i,i] = n_a[i]


    
    TWORDM = (np.zeros((M,M,M,M)), np.zeros((M,M,M,M)), np.zeros((M,M,M,M)))
    # alpha alpha alpha alpha 
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==k and j==l:
                        TWORDM[0][i,j,k,l] += n_a[i]*n_a[j] - Delta[i,j]
                    if i==l and k==j:
                        TWORDM[0][i,j,k,l] += -n_a[i]*n_a[j] + Delta[i,j]
    # alpha beta and beta alpha
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==k and j==l:
                        TWORDM[1][i,j,k,l] += n_a[i]*n_b[j] - Delta[i,j] 
                    if i==j and k==l:
                        TWORDM[1][i,j,k,l] +=  Pi[i,k]
    # beta beta beta beta
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==k and j==l:
                        TWORDM[2][i,j,k,l] += n_b[i]*n_b[j] - Delta[i,j]
                    if i==l and k==j:
                        TWORDM[2][i,j,k,l] += -n_b[i]*n_b[j] + Delta[i,j]
#                    print(TWORDM[2][i,j,k,l])

    return TWORDM
    
def HF_U2RDM_phys(n_a, n_b, M):
    ''' Compute 2 RDMFTs in Natural Spin Orbital basis for Mueller approximation
    Parameters 
    ----------
    n_a, n_b  : np.ndarray
        occupation numbers of a spin unrestricted 1dm, they lie in [0,1]
    M : integer 
        basis set size

    Returns
    -------
    TWORDM : tupel of 3 np.ndarrays
        2RDMs alpha,alpha-block alpha,beta-block and beta,beta block  
    '''
    TWORDM = (np.zeros((M,M,M,M)), np.zeros((M,M,M,M)), np.zeros((M,M,M,M)))
    # alpha alpha alpha alpha 
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==k and j==l:
                        TWORDM[0][i,j,k,l] = n_a[i]*n_a[j]
                    if i==l and k==j:
                        TWORDM[0][i,j,k,l] -= n_a[i]*n_a[j]
    # alpha beta and beta alpha
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==k and j==l:
                        TWORDM[1][i,j,k,l] = n_a[i]*n_b[j]
    # beta beta beta beta
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==k and j==l:
                        TWORDM[2][i,j,k,l] = n_b[i]*n_b[j]
                    if i==l and k==j:
                        TWORDM[2][i,j,k,l] -= n_b[i]*n_b[j]
#                    print(TWORDM[2][i,j,k,l])

    return TWORDM



            
    

def HF_U2RDM(n_a, n_b, M):
    ''' Compute 2 RDMFTs in Natural Spin Orbital basis for Mueller approximation
    Parameters 
    ----------
    n_a, n_b  : np.ndarray
        occupation numbers of a spin unrestricted 1dm, they lie in [0,1]
    M : integer 
        basis set size

    Returns
    -------
    TWORDM : tupel of 3 np.ndarrays
        2RDMs alpha,alpha-block alpha,beta-block and beta,beta block  
    '''
    TWORDM = (np.zeros((M,M,M,M)), np.zeros((M,M,M,M)), np.zeros((M,M,M,M)))
    # alpha alpha alpha alpha 
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==j and k==l:
                        TWORDM[0][i,j,k,l] = n_a[i]*n_a[k]
                    if i==l and k==j:
                        TWORDM[0][i,j,k,l] -= n_a[i]*n_a[k]
    # alpha beta and beta alpha
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==j and k==l:
                        TWORDM[1][i,j,k,l] = n_a[i]*n_b[k]
    # beta beta beta beta
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==j and k==l:
                        TWORDM[2][i,j,k,l] = n_b[i]*n_b[k]
                    if i==l and k==j:
                        TWORDM[2][i,j,k,l] -= n_b[i]*n_b[k]

    return TWORDM

def MU_U2RDM(n_a, n_b, M):
    ''' Compute 2 RDMFTs in Natural Spin Orbital basis for Mueller approximation
    Parameters
    ----------
    n_a, n_b  : np.ndarray
        occupation numbers of a spin unrestricted 1dm, they lie in [0,1]
    M : integer 
        basis set size

    Returns
    -------
    TWORDM : tupel of 3 np.ndarrays
        2RDMs alpha,alpha-block alpha,beta-block and beta,beta block  
    '''
    TWORDM = (np.zeros((M,M,M,M)), np.zeros((M,M,M,M)), np.zeros((M,M,M,M)))
    # alpha alpha alpha alpha 
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==j and k==l:
                        TWORDM[0][i,j,k,l] = n_a[i]*n_a[k]
                    if i==l and k==j:
                        TWORDM[0][i,j,k,l] -= np.sqrt(n_a[i]*n_a[k])
    # alpha beta and beta alpha
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==j and k==l:
                        TWORDM[1][i,j,k,l] = n_a[i]*n_b[k]
    # beta beta 
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==j and k==l:
                        TWORDM[2][i,j,k,l] = n_b[i]*n_b[k]
                    if i==l and k==j:
                        TWORDM[2][i,j,k,l] -= np.sqrt(n_b[i]*n_b[k])

    return TWORDM

def GU_U2RDM(n_a, n_b, M):
    ''' Compute 2 RDMFTs in Natural Orbital basis for Goedecker Umrigar approximation
    Parameters
    ----------
    n_a, n_b  : np.ndarray
        occupation numbers of a spin unrestricted 1dm, they lie in [0,1]
    M : integer 
        basis set size
    Returns
    -------
    TWORDM : tupel of 3 np.ndarrays
        2RDMs alpha,alpha-block alpha,beta-block and beta,beta block  
    '''    
    TWORDM = (np.zeros((M,M,M,M)), np.zeros((M,M,M,M)), np.zeros((M,M,M,M)))
    # alpha alpha alpha alpha 
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==j and k==l and j!=k:
                        TWORDM[0][i,j,k,l] = n_a[i]*n_a[k]
                    if i==l and k==j and l!=k :
                        TWORDM[0][i,j,k,l] -= np.sqrt(n_a[i]*n_a[k])
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==j and k==l:
                        TWORDM[1][i,j,k,l] = n_a[i]*n_b[k]

    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==j and k==l and j!=k:
                        TWORDM[2][i,j,k,l] = n_b[i]*n_b[k]
                    if i==l and k==j and l!=k :
                        TWORDM[2][i,j,k,l] -= np.sqrt(n_b[i]*n_b[k])


    return TWORDM

def BBC1_U2RDM(n_a, n_b, M, N_a, N_b):
    ''' Compute 2 RDMFTs in Natural Orbital basis for BBC1 approximation
    Parameters
    ----------
    n_a, n_b  : np.ndarray
        occupation numbers of a spin unrestricted 1dm, they lie in [0,1]
    N_a, N_b : integer
        number of "strongly" occupied orbitals in alpha and beta channel
    M : integer 
        basis set size
    N : integer 
        number of strongly occupied (close to 1) orbitals
    Returns
    -------
    TWORDM : tupel of 3 np.ndarrays
        2RDMs alpha,alpha-block alpha,beta-block and beta,beta block  
    ''' 
    TWORDM = (np.zeros((M,M,M,M)), np.zeros((M,M,M,M)), np.zeros((M,M,M,M)))
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==j and k==l:
                        TWORDM[0][i,j,k,l] = n_a[i]*n_a[k]
                    if i==l and k==j :
                        TWORDM[0][i,j,k,l] -= np.sqrt(n_a[i]*n_a[k])
                    if i==l and k==j and (i >= N_a and k >= N_a) and l!=k :
                        TWORDM[0][i,j,k,l] += 2*np.sqrt(n_a[i]*n_a[k])
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==j and k==l:
                        TWORDM[1][i,j,k,l] = n_a[i]*n_b[k]

    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==j and k==l:
                        TWORDM[2][i,j,k,l] = n_b[i]*n_b[k]
                    if i==l and k==j :
                        TWORDM[2][i,j,k,l] -= np.sqrt(n_b[i]*n_b[k])
                    if i==l and k==j and (i >= N_b and k >= N_b) and l!=k :
                        TWORDM[2][i,j,k,l] += 2*np.sqrt(n_b[i]*n_b[k])
                    
    return TWORDM

def BBC2_U2RDM(n_a, n_b, M, N_a, N_b):
    ''' Compute 2 RDMFTs in Natural Orbital basis for BBC2 approximation
    Parameters
    ----------
    n_a, n_b  : np.ndarray
        occupation numbers of a spin unrestricted 1dm, they lie in [0,1]
    N_a, N_b : integer
        number of "strongly" occupied orbitals in alpha and beta channel
    M : integer 
        basis set size
    N : integer 
        number of strongly occupied (close to 1) orbitals
    Returns
    -------
    TWORDM : tupel of 3 np.ndarrays
        2RDMs alpha,alpha-block alpha,beta-block and beta,beta block  
    '''   
    TWORDM = (np.zeros((M,M,M,M)), np.zeros((M,M,M,M)), np.zeros((M,M,M,M)))
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==j and k==l:
                        TWORDM[0][i,j,k,l] = n_a[i]*n_a[k]
                    if i==l and k==j :
                        TWORDM[0][i,j,k,l] -= np.sqrt(n_a[i]*n_a[k])
                    if i==l and k==j and (i >= N_a and k >= N_a) and l!=k :
                        TWORDM[0][i,j,k,l] += 2*np.sqrt(n_a[i]*n_a[k])
                    if i==l and k==j and (i < N_a and k < N_a) and l!=k :
                        TWORDM[0][i,j,k,l] += np.sqrt(n_a[i]*n_a[k])-.5*n_a[i]*n_a[k]

    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==j and k==l:
                        TWORDM[1][i,j,k,l] = n_a[i]*n_b[k]

    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==j and k==l:
                        TWORDM[2][i,j,k,l] = n_b[i]*n_b[k]
                    if i==l and k==j :
                        TWORDM[2][i,j,k,l] -= np.sqrt(n_b[i]*n_b[k])
                    if i==l and k==j and (i >= N_b and k >= N_b) and l!=k :
                        TWORDM[2][i,j,k,l] += 2*np.sqrt(n_b[i]*n_b[k])
                    if i==l and k==j and (i < N_b and k < N_b) and l!=k :
                        TWORDM[2][i,j,k,l] += np.sqrt(n_b[i]*n_b[k])-.5*n_b[i]*n_b[k]

                    
    return TWORDM

def BBC3_U2RDM(n_a, n_b, M, N_a, N_b,t_a,t_b):
    ''' Compute 2 RDMFTs in Natural Orbital basis for BBC3 approximation
    Parameters
    ----------
    n_a, n_b  : np.ndarray
        occupation numbers of a spin unrestricted 1dm, they lie in [0,1]
    N_a, N_b : integer
        number of "strongly" occupied orbitals in alpha and beta channel
    M : integer 
        basis set size
    N : integer 
        number of strongly occupied (close to 1) orbitals
    t_a, t_b : integer 
        the index of the antibonding orbital
    Returns
    -------
    TWORDM : tupel of 3 np.ndarrays
        2RDMs alpha,alpha-block alpha,beta-block and beta,beta block  
    '''   
    TWORDM = (np.zeros((M,M,M,M)), np.zeros((M,M,M,M)), np.zeros((M,M,M,M)))
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==j and k==l:
                        TWORDM[0][i,j,k,l] = n_a[i]*n_a[k]
                    if i==l and k==j :
                        TWORDM[0][i,j,k,l] -= np.sqrt(n_a[i]*n_a[k])
                    if i==l and k==j and (i >= N_a and k >= N_a) and l!=k :
                        TWORDM[0][i,j,k,l] += 2*np.sqrt(n_a[i]*n_a[k])
                    if i==l and k==j and (i < N_a and k < N_a) and l!=k :
                        TWORDM[0][i,j,k,l] += np.sqrt(n_a[i]*n_a[k])-.5*n_a[i]*n_a[k]
                    if i==j and i==k and i==l and i != N_a-1 and i != t_a:
                        TWORDM[2][i,j,k,l] = -n_a[i]**2+n_a[i]

    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==j and k==l:
                        TWORDM[1][i,j,k,l] = n_a[i]*n_b[k]

    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==j and k==l:
                        TWORDM[2][i,j,k,l] = n_b[i]*n_b[k]
                    if i==l and k==j :
                        TWORDM[2][i,j,k,l] -= np.sqrt(n_b[i]*n_b[k])
                    if i==l and k==j and (i >= N_b and k >= N_b) and l!=k :
                        TWORDM[2][i,j,k,l] += 2*np.sqrt(n_b[i]*n_b[k])
                    if i==l and k==j and (i < N_b and k < N_b) and l!=k :
                        TWORDM[2][i,j,k,l] += np.sqrt(n_b[i]*n_b[k])-.5*n_b[i]*n_b[k]
                    if i==j and i==k and i==l and i != N_b-1 : #and i != t_b: #to extend this it needs 
                        TWORDM[2][i,j,k,l] = -n_b[i]**2+n_b[i]


                    
    return TWORDM

def MU_2RDM(n, M):
    ''' Compute 2 RDMFTs in Natural Orbital basis for Mueller approximation
    Parameters
    ----------
    n : np.ndarray
        occupation numbers of a spin restricted 1dm, they lie in [0,2]
    M : integer 
        basis set size

    Returns
    -------
    TWORDM : np.ndarray
        2RDM
    '''
    TWORDM = np.zeros((M,M,M,M))
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==j and k==l:
                        TWORDM[i,j,k,l] = 4*n[i]/2*n[k]/2
                    if i==l and k==j:
                        TWORDM[i,j,k,l] -= 2*np.sqrt(n[i]/2*n[k]/2)
    return TWORDM

def GU_2RDM(n, M):
    ''' Compute 2 RDMFTs in Natural Orbital basis for Goedecker Umrigar approximation
    Parameters
    ----------
    n_a, n_b  : np.ndarray
        occupation numbers of a spin unrestricted 1dm, they lie in [0,1]
    M : integer 
        basis set size
    Returns
    -------
    TWORDM : tupel of 3 np.ndarrays
        2RDMs alpha,alpha-block alpha,beta-block and beta,beta block  
    ''' 
    TWORDM = np.zeros((M,M,M,M))
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==j and k==l and j!=k:
                        TWORDM[i,j,k,l] = 4*n[i]/2*n[k]/2
                    if i==l and k==j and l!=k :
                        TWORDM[i,j,k,l] -= 2*np.sqrt(n[i]/2*n[k]/2)
                    if i==j and k==l and j==k:
                         TWORDM[i,j,k,l] = 2*n[i]/2*n[k]/2 

    return TWORDM

def BBC1_2RDM(n, M, N):
    ''' Compute 2 RDMFTs in Natural Orbital basis for Goedecker Umrigar approximation
    Parameters
    ----------
    n : np.ndarray
        occupation numbers of a spin restricted 1dm, they lie in [0,2]
    M : integer 
        basis set size
    N : integer 
        number of strongly, double-occupied (close to 2) spatial-orbitals
    Returns
    -------
    TWORDM : np.ndarray
        2RDM
    '''    
    TWORDM = np.zeros((M,M,M,M))
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==j and k==l:
                        TWORDM[i,j,k,l] = 4*n[i]/2*n[k]/2
                    if i==l and k==j :
                        TWORDM[i,j,k,l] -= 2*np.sqrt(n[i]/2*n[k]/2)
                    if i==l and k==j and (i >= N and k >= N) and l!=k :
                        TWORDM[i,j,k,l] += 4*np.sqrt(n[i]/2*n[k]/2)
                    
    return TWORDM

def HF_2RDM(n, M):
    ''' Compute 2 RDMFTs in Natural Orbital basis for Mueller approximation
    Parameters
    ----------
    n : np.ndarray
        occupation numbers of a spin restricted 1dm, they lie in [0,2]
    M : integer 
        basis set size

    Returns
    -------
    TWORDM : np.ndarray
        2RDM
    '''
    TWORDM = np.zeros((M,M,M,M))
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==j and k==l:
                        TWORDM[i,j,k,l] = 4*n[i]/2*n[k]/2
                    if i==l and k==j:
                        TWORDM[i,j,k,l] -= 2*n[i]/2*n[k]/2
    return TWORDM

def R_PNOF2(n, M, N):
    ''' Compute 2 RDMFTs in Natural Orbital basis for Mueller approximation
    Parameters
    ----------
    n : np.ndarray
        occupation numbers of a spin restricted 1dm, they lie in [0,2]
    M : integer 
        basis set size

    Returns
    -------
    TWORDM : np.ndarray
        2RDM
    '''
    TWORDM = np.zeros((M,M,M,M))
    S=0
    for i in range(0,N):
        S+=max((2.0-n[i]),0)
    for i in range(0,M):
        for j in range(0,M):
            for k in range(0,M):
                for l in range(0,M):
                    if i==j and k==l and i < N and k < N and i != k:
                        TWORDM[i,j,k,l] += n[i]*n[k]-np.sqrt(max((2.0-n[i]),0)*max((2.0-n[k]),0))
                    if i==j and k==l and i < N and k >= N:
                        TWORDM[i,j,k,l] += n[i]*n[k]-(2-S)/(S)*max((2.0-n[i]),0)*n[k]
                    if i==j and k==l and k < N and i >= N:
                        TWORDM[i,j,k,l] += n[i]*n[k]-(2-S)/(S)*max((2.0-n[k]),0)*n[i]
                    if i==l and k==j and i < N and k < N:
                        TWORDM[i,j,k,l] -= 1/2*np.sqrt(n[i]*n[k])-1/2*np.sqrt(max((2.0-n[i]),0)*max((2.0-n[k]),0))
                    if i==l and k==j and i < N and k >= N:
                        TWORDM[i,j,k,l] -= 1/2*np.sqrt(n[i]*n[k])+1/2*np.sqrt(max((2.0-n[i]),0)*n[k])
                    if i==l and k==j and k < N and i >= N:
                        TWORDM[i,j,k,l] -= 1/2*np.sqrt(n[i]*n[k])+1/2*np.sqrt(n[i]*max((2.0-n[k]),0))
    return TWORDM

def BBC2_2RDM(n, M, N):
    ''' Compute 2 RDMFTs in Natural Orbital basis for BBC2 approximation
    Parameters
    ----------
    n : np.ndarray
        occupation numbers of a spin restricted 1dm, they lie in [0,2]
    M : integer 
        basis set size
    N : integer 
        number of strongly, double-occupied (close to 2) spatial-orbitals
    Returns
    -------
    TWORDM : np.ndarray
        2RDM
    '''    
    TWORDM = BBC2_U2RDM(n*0.5, n*0.5, M, N, N)
                    
    return TWORDM[0]+2*TWORDM[1]+TWORDM[2]

def BBC3_2RDM(n, M, N, t):
    ''' Compute 2 RDMFTs in Natural Orbital basis for BBC2 approximation
    Parameters
    ----------
    n : np.ndarray
        occupation numbers of a spin restricted 1dm, they lie in [0,2]
    M : integer 
        basis set size
    N : integer 
        number of strongly, double-occupied (close to 2) spatial-orbitals
    Returns
    -------
    TWORDM : np.ndarray
        2RDM
    '''    
    TWORDM = BBC3_U2RDM(n*0.5, n*0.5, M, N, N, t, t)
                    
    return TWORDM[0]+2*TWORDM[1]+TWORDM[2]
