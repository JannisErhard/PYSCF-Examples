import numpy as np

def spectral_clean(dm1, M):
    
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