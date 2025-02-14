import numpy as np

def spectral_clean(dm1):
    '''
    Parameters 
    ----------
    dm1 : list of numpy.ndarray
    A list containing two (N, N) density matrices:
    - dm1[0]: Spin-up density matrix (N x N)
    - dm1[1]: Spin-down density matrix (N x N)
    Each matrix represents the electron density in a given basis.

    Returns
    -------    
    occ_a : numpy.ndarray
    Eigenvalues (occupations) of the spin-up density matrix, sorted in descending order.
    Any negative eigenvalues are set to zero.
    occ_b : numpy.ndarray
    Eigenvalues (occupations) of the spin-down density matrix, sorted in descending order.
    Any negative eigenvalues are set to zero.
    C_NAO_a : numpy.ndarray
    Eigenvectors (Natural Atomic Orbitals) corresponding to occ_a, ordered accordingly.
    C_NAO_b : numpy.ndarray
    Eigenvectors (Natural Atomic Orbitals) corresponding to occ_b, ordered accordingly.
    '''
    
    occ_a, C_NAO_a = np.linalg.eigh(dm1[0])
    occ_b, C_NAO_b = np.linalg.eigh(dm1[1])
    
    C_NAO_a = C_NAO_a[:,::-1]
    occ_a = occ_a[::-1]
    for i, n  in enumerate(occ_a):
        if n < 0:
            occ_a[i] = 0
    
    C_NAO_b = C_NAO_b[:,::-1]
    occ_b = occ_b[::-1]
    for i, n  in enumerate(occ_b):
        if n < 0:
            occ_b[i] = 0
        
    return occ_a, occ_b, C_NAO_a, C_NAO_b

def R_twoDM_Eigvals(dm2):
    return np.linalg.eigvalsh(np.reshape(dm2, (dm2.shape[0]**2, dm2.shape[0]**2)))

def twoDM_Eigvals(dm2):
    N = dm2[0].shape[0]
    FULL_2dm = np.zeros((dm2[0].shape[0]*2, dm2[0].shape[0]*2, dm2[0].shape[0]*2, dm2[0].shape[0]*2))
    FULL_2dm[0:N,0:N,0:N,0:N]  = np.transpose(dm2[0], (0, 2, 1, 3))
    FULL_2dm[0:N,N:2*N,0:N,N:2*N] = np.transpose(dm2[1], (0, 2, 1, 3))
    FULL_2dm[N:2*N,0:N,N:2*N,0:N] = np.transpose(dm2[1], (0, 2, 1, 3))
    FULL_2dm[N:2*N,N:2*N,N:2*N,N:2*N] = np.transpose(dm2[2], (0, 2, 1, 3))
    return np.linalg.eigvalsh(np.reshape(FULL_2dm, ((2*N)**2, (2*N)**2)))

def R_spectral_clean(dm1, M):
    
    occ_a, C_NAO_a = np.linalg.eigh(dm1)
    
    C_NAO_a = C_NAO_a[::-1]
    occ_a = occ_a[::-1]

    for i, n  in enumerate(occ_a):
        if n < 0:
            occ_a[i] = 0
    return occ_a, C_NAO_a

def DM2_DiagSum(dm2):
    N = dm2[0].shape[0]
    FULL_2dm = np.zeros((dm2[0].shape[0]*2, dm2[0].shape[0]*2, dm2[0].shape[0]*2, dm2[0].shape[0]*2))
    FULL_2dm[0:N,0:N,0:N,0:N]  = np.transpose(dm2[0], (0, 2, 1, 3))
    FULL_2dm[0:N,N:2*N,0:N,N:2*N] = np.transpose(dm2[1], (0, 2, 1, 3))
    FULL_2dm[N:2*N,0:N,N:2*N,0:N] = np.transpose(dm2[1], (0, 2, 1, 3))
    FULL_2dm[N:2*N,N:2*N,N:2*N,N:2*N] = np.transpose(dm2[2], (0, 2, 1, 3))
    return np.trace(np.reshape(FULL_2dm, (4*N**2, 4*N**2)))

def BST_FIDX(FIDX, C_NAO_a, C_NAO_b):
    FIDX_NAO_aa = FIDX[0].copy()
    FIDX_NAO_ab = FIDX[1].copy()
    FIDX_NAO_bb = FIDX[2].copy()
    for i in range(4):
        FIDX_NAO_aa = np.tensordot(FIDX_NAO_aa, C_NAO_a, axes=1).transpose(3, 0, 1, 2)
        FIDX_NAO_bb = np.tensordot(FIDX_NAO_bb, C_NAO_b, axes=1).transpose(3, 0, 1, 2)
    
    FIDX_NAO_ab = np.tensordot(FIDX_NAO_ab, C_NAO_a, axes=1).transpose(3, 0, 1, 2)
    FIDX_NAO_ab = np.tensordot(FIDX_NAO_ab, C_NAO_a, axes=1).transpose(3, 0, 1, 2)
    FIDX_NAO_ab = np.tensordot(FIDX_NAO_ab, C_NAO_b, axes=1).transpose(3, 0, 1, 2)
    FIDX_NAO_ab = np.tensordot(FIDX_NAO_ab, C_NAO_b, axes=1).transpose(3, 0, 1, 2)

    return (FIDX_NAO_aa, FIDX_NAO_ab, FIDX_NAO_bb)

def R_Vee(eri,dm2):
    ''' Compute electron-electron repulsion for 2 body integrals and 2RDM
    ----------
    eri : np.ndarray
        2 body integrals
    dm2 : np.ndarray 
        second order reduced density matrix

    Returns
    -------
    _ : np.float
        it is the energy component V_ee
    '''
    return .5*np.einsum('pqrs,pqrs', eri, dm2)

def Vee(eri,dm2):
    ''' Compute electron-electron repulsion for unrestricted 2 body integrals and unrestricted 2RDM
    ----------
    eri : np.ndarray
        2 body integrals
    dm2 : np.ndarray 
        second order reduced density matrix

    Returns
    -------
    _ : np.float
        it is the energy component V_ee
    '''
    return .5*np.einsum('pqrs,pqrs', eri[0], dm2[0])+.5*np.einsum('pqrs,pqrs', eri[2], dm2[2])+np.einsum('pqrs,pqrs', eri[1], dm2[1])