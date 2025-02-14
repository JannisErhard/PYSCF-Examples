import numpy as np

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