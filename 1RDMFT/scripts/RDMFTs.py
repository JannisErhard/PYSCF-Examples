import numpy as np
#from numba import prange, jit, njit
import sys
sys.path.insert(0, './build/')
import RDMFT

# for jit and prange
#@jit(nopython=True, parallel=True)
def ONERDMFT_hartree_energy_parallel(Fouridx, C, n):
    energy = 0
    M = C.shape[0]
    K = Fouridx.shape[0]
    for a in  prange(0,M):
        for b in range(0,M):
            for mu in range(0,M):
                for nu in range(0,M):
                    for kappa in range(0,M):
                        for lamda in range(0,M):
                            energy += n[a]*n[b]*C[mu,a]*C[nu,a]*C[kappa,b]*C[lamda,b]*Fouridx[mu%K,nu%K,kappa%K,lamda%K]

    return 1/2.*energy

# for spinorbitals
#@jit(nopython=True, parallel=True)
def ONERDMFT_Umrigar_hartree_energy_parallel(Fouridx, C, n):
    energy = 0
    M = C.shape[0]
    K = Fouridx.shape[0]
    for a in  prange(0,M):
        for b in range(a+1,M): #[b for b in range(0,M) if b!=a ]:
            for mu in range(0,M):
                for nu in range(0,M):
                    for kappa in range(0,M):
                        for lamda in range(0,M):
                            energy += n[a]*n[b]*C[mu,a]*C[nu,a]*C[kappa,b]*C[lamda,b]*Fouridx[mu%K,nu%K,kappa%K,lamda%K]

    return energy
#@jit(nopython=True, parallel=True)
def ONERDMFT_Umrigar_exchange_correlation_energy_parallel(Fouridx, C, n):
    energy = 0
    M = C.shape[0]
    K = Fouridx.shape[0]
    for a in  prange(0,M//2):
        for b in [b for b in range(0,M//2) if b!=a ]:
            for mu in range(0,M):
                for nu in range(0,M):
                    for kappa in range(0,M):
                        for lamda in range(0,M):
                            energy += np.sqrt(n[a]*n[b])*C[mu,a]*C[nu,b]*C[kappa,a]*C[lamda,b]*Fouridx[mu%K,nu%K,kappa%K,lamda%K]
    for a in  prange(M//2,M):
        for b in [b for b in range(M//2,M) if b!=a ]:
            for mu in range(0,M):
                for nu in range(0,M):
                    for kappa in range(0,M):
                        for lamda in range(0,M):
                            energy += np.sqrt(n[a]*n[b])*C[mu,a]*C[nu,b]*C[kappa,a]*C[lamda,b]*Fouridx[mu%K,nu%K,kappa%K,lamda%K]

    

    return -1/2.*energy

#@jit(parallel=True)
def ONERDMFT_exchange_energy_parallel(Fouridx, C, n):
    energy = 0
    M = C.shape[0]
    K = Fouridx.shape[0]
    for a in  prange(0,M//2):
        for b in range(0,M//2):
            for mu in range(0,M):
                for nu in range(0,M):
                    for kappa in range(0,M):
                        for lamda in range(0,M):
                            energy += n[a]*n[b]*C[mu,a]*C[nu,b]*C[kappa,a]*C[lamda,b]*Fouridx[mu%K,nu%K,kappa%K,lamda%K]
    for a in  prange(M//2,M):
        for b in range(M//2,M):
            for mu in range(0,M):
                for nu in range(0,M):
                    for kappa in range(0,M):
                        for lamda in range(0,M):
                            energy += n[a]*n[b]*C[mu,a]*C[nu,b]*C[kappa,a]*C[lamda,b]*Fouridx[mu%K,nu%K,kappa%K,lamda%K]


    return -1/2.*energy

#@jit(parallel=True)
def ONERDMFT_Mueller_exchange_correlation_energy_parallel(Fouridx, C, n):
    energy = 0
    M = C.shape[0]
    K = Fouridx.shape[0]
    for a in  prange(0,M//2):
        for b in range(0,M//2):
            for mu in range(0,M):
                for nu in range(0,M):
                    for kappa in range(0,M):
                        for lamda in range(0,M):
                            energy += np.sqrt(n[a]*n[b])*C[mu,a]*C[nu,b]*C[kappa,a]*C[lamda,b]*Fouridx[mu%K,nu%K,kappa%K,lamda%K]
    for a in  prange(M//2,M):
        for b in range(M//2,M):
            for mu in range(0,M):
                for nu in range(0,M):
                    for kappa in range(0,M):
                        for lamda in range(0,M):
                            energy += np.sqrt(n[a]*n[b])*C[mu,a]*C[nu,b]*C[kappa,a]*C[lamda,b]*Fouridx[mu%K,nu%K,kappa%K,lamda%K]


    return -1/2.*energy

#@jit(parallel=True)
def ONERDMFT_BBC1(Fouridx, C, n, Na, Nb):
    energy = 0
    M = C.shape[0]
    K = Fouridx.shape[0]
    for a in  prange(Na,M//2):
        for b in [b for b in range(Na,M//2) if b!=a ]:
            for mu in range(0,M):
                for nu in range(0,M):
                    for kappa in range(0,M):
                        for lamda in range(0,M):
                            energy += np.sqrt(n[a]*n[b])\
                            *C[mu,a]*C[nu,b]*C[kappa,a]*C[lamda,b]\
                            *Fouridx[mu%K,nu%K,kappa%K,lamda%K]
    for a in  prange(M//2+Nb,M):
        for b in [b for b in range(M//2+Nb,M) if b!=a ]:
            for mu in range(0,M):
                for nu in range(0,M):
                    for kappa in range(0,M):
                        for lamda in range(0,M):
                            energy += np.sqrt(n[a]*n[b])\
                            *C[mu,a]*C[nu,b]*C[kappa,a]*C[lamda,b]\
                            *Fouridx[mu%K,nu%K,kappa%K,lamda%K]


    return energy

#@jit(parallel=True)
def ONERDMFT_BBC2(Fouridx, C, n, Na, Nb):
    energy = 0
    M = C.shape[0]
    K = Fouridx.shape[0]
    for a in  prange(0,Na):
        for b in [b for b in range(0,Na) if b!=a ]:
            for mu in range(0,M):
                for nu in range(0,M):
                    for kappa in range(0,M):
                        for lamda in range(0,M):
                            energy += (np.sqrt(n[a]*n[b])-(n[a]*n[b]))\
                            *C[mu,a]*C[nu,b]*C[kappa,a]*C[lamda,b]\
                            *Fouridx[mu%K,nu%K,kappa%K,lamda%K]
    for a in  prange(M//2,M//2+Nb):
        for b in [b for b in range(M//2,M//2+Nb) if b!=a ]:
            for mu in range(0,M):
                for nu in range(0,M):
                    for kappa in range(0,M):
                        for lamda in range(0,M):
                            energy += (np.sqrt(n[a]*n[b])-(n[a]*n[b]))\
                            *C[mu,a]*C[nu,b]*C[kappa,a]*C[lamda,b]\
                            *Fouridx[mu%K,nu%K,kappa%K,lamda%K]


    return  .5*energy

#@jit(parallel=True)
# this is only the correct BBC3 if there are no Bonds involved, without being able to predict bonds I dont know what to implement, read text leading up to Eq 3.10 in 10.1063/1.1906203 to understand this
def ONERDMFT_BBC3(Fouridx, C, n, Na, Nb):
    energy = 0
    M = C.shape[0]
    K = Fouridx.shape[0]
    for a in  prange(0,M):
        for mu in range(0,M):
            for nu in range(0,M):
                for kappa in range(0,M):
                    for lamda in range(0,M):
                        energy += (np.sqrt(n[a]*n[a])-(n[a]*n[a]))*C[mu,a]*C[nu,a]*C[kappa,a]*C[lamda,a]*Fouridx[mu%K,nu%K,kappa%K,lamda%K]


    return energy

def energy_components_umrigar(eri, FCInaturalCTTE, FCIoccuE,h1,E_HF,E_nn,PYTHONIC):
    if PYTHONIC:
        E_U = ONERDMFT_Umrigar_hartree_energy_parallel(eri, FCInaturalCTTE, FCIoccuE)
        GU_E_xc = ONERDMFT_Umrigar_exchange_correlation_energy_parallel(eri, FCInaturalCTTE, FCIoccuE)
    else:
        E_U = RDMFT.wrap_gu_hartree(FCIoccuE,FCInaturalCTTE,eri,eri.shape[0])
        GU_E_xc = RDMFT.wrap_gu_xc(FCIoccuE,FCInaturalCTTE,eri,eri.shape[0])
    Vee = E_U + GU_E_xc
    E_tot = h1 + Vee + E_nn
    E_c = E_tot - E_HF
    return E_tot, Vee, E_c 

def energy_components_mueller(eri, FCInaturalCTTE, FCIoccuE,h1,E_HF,E_nn,PYTHONIC):
    if PYTHONIC:
        E_H = ONERDMFT_hartree_energy_parallel(eri, FCInaturalCTTE, FCIoccuE)
        Mu_E_xc = ONERDMFT_Mueller_exchange_correlation_energy_parallel(eri, FCInaturalCTTE, FCIoccuE)
    else:
        E_H = RDMFT.wrap_hartree(FCIoccuE,FCInaturalCTTE,eri,eri.shape[0])
        Mu_E_xc = RDMFT.wrap_mu_xc(FCIoccuE,FCInaturalCTTE,eri,eri.shape[0])
    Vee = E_H + Mu_E_xc
    E_tot = h1 + Vee + E_nn
    E_c = E_tot - E_HF
    return E_tot, Vee, E_c

def energy_components_bbc1(eri, FCInaturalCTTE, FCIoccuE,h1,E_HF,E_nn,nelec,PYTHONIC):
    n_a, n_b = nelec[0], nelec[1]
    if PYTHONIC:
        E_H = ONERDMFT_hartree_energy_parallel(eri, FCInaturalCTTE, FCIoccuE)
        Mu_E_xc = ONERDMFT_Mueller_exchange_correlation_energy_parallel(eri, FCInaturalCTTE, FCIoccuE)
        BBC1 = ONERDMFT_BBC1(eri, FCInaturalCTTE, FCIoccuE,n_a,n_b)
    else:
        E_H = RDMFT.wrap_hartree(FCIoccuE,FCInaturalCTTE,eri,eri.shape[0])
        Mu_E_xc = RDMFT.wrap_mu_xc(FCIoccuE,FCInaturalCTTE,eri,eri.shape[0])
        BBC1 = RDMFT.wrap_bbc_1(n_a,n_b,FCIoccuE,FCInaturalCTTE,eri,eri.shape[0])
    Vee = E_H + Mu_E_xc + BBC1
    E_tot = h1 + Vee + E_nn
    E_c = E_tot - E_HF
    return E_tot, Vee, E_c

def energy_components_bbc2(eri, FCInaturalCTTE, FCIoccuE,h1,E_HF,E_nn,nelec,PYTHONIC):
    n_a, n_b = nelec[0], nelec[1]
    if PYTHONIC:
        E_H = ONERDMFT_hartree_energy_parallel(eri, FCInaturalCTTE, FCIoccuE)
        Mu_E_xc = ONERDMFT_Mueller_exchange_correlation_energy_parallel(eri, FCInaturalCTTE, FCIoccuE)
        BBC1 = ONERDMFT_BBC1(eri, FCInaturalCTTE, FCIoccuE,n_a,n_b)
        BBC2 = ONERDMFT_BBC2(eri, FCInaturalCTTE, FCIoccuE,n_a,n_b)

    else:
        E_H = RDMFT.wrap_hartree(FCIoccuE,FCInaturalCTTE,eri,eri.shape[0])
        Mu_E_xc = RDMFT.wrap_mu_xc(FCIoccuE,FCInaturalCTTE,eri,eri.shape[0])
        BBC1 = RDMFT.wrap_bbc_1(n_a,n_b,FCIoccuE,FCInaturalCTTE,eri,eri.shape[0])
        BBC2 = RDMFT.wrap_bbc_2(n_a,n_b,FCIoccuE,FCInaturalCTTE,eri,eri.shape[0])

    Vee = E_H + Mu_E_xc + BBC1 + BBC2
    E_tot = h1 + Vee + E_nn
    E_c = E_tot - E_HF
    return E_tot, Vee, E_c

def energy_components_bbc3(eri, FCInaturalCTTE, FCIoccuE,h1,E_HF,E_nn,nelec,PYTHONIC):
    n_a, n_b = nelec[0], nelec[1]
    if PYTHONIC:
        E_H = ONERDMFT_hartree_energy_parallel(eri, FCInaturalCTTE, FCIoccuE)
        Mu_E_xc = ONERDMFT_Mueller_exchange_correlation_energy_parallel(eri, FCInaturalCTTE, FCIoccuE)
        BBC1 = ONERDMFT_BBC1(eri, FCInaturalCTTE, FCIoccuE,n_a,n_b)
        BBC2 = ONERDMFT_BBC2(eri, FCInaturalCTTE, FCIoccuE,n_a,n_b)
        BBC3 = ONERDMFT_BBC3(eri, FCInaturalCTTE, FCIoccuE,n_a,n_b)

    else:
        E_H = RDMFT.wrap_hartree(FCIoccuE,FCInaturalCTTE,eri,eri.shape[0])
        Mu_E_xc = RDMFT.wrap_mu_xc(FCIoccuE,FCInaturalCTTE,eri,eri.shape[0])
        BBC1 = RDMFT.wrap_bbc_1(n_a,n_b,FCIoccuE,FCInaturalCTTE,eri,eri.shape[0])
        BBC2 = RDMFT.wrap_bbc_2(n_a,n_b,FCIoccuE,FCInaturalCTTE,eri,eri.shape[0])
        BBC3 = RDMFT.wrap_bbc_3(n_a,n_b,FCIoccuE,FCInaturalCTTE,eri,eri.shape[0])

    Vee = E_H + Mu_E_xc + BBC1 + BBC2 + BBC3
    E_tot = h1 + Vee + E_nn
    E_c = E_tot - E_HF
    return E_tot, Vee, E_c
