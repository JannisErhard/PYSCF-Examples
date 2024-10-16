import numpy as np
from pyscf import gto, scf, fci
from numba import prange, jit, njit
from tabulate import tabulate
from RDMFTs import energy_components_umrigar, energy_components_mueller, energy_components_bbc1
from utils import expand_matrix, Add_Block_Matrices 

tab_top = "H-H Vee_{GU} Vee_{Mu} Vee_{BBC1} Vee_{FCI}"
R = np.linspace(0.5,6,30)
stats, tab = [], []
for dist in R:
    mol = gto.Mole()
    mol.unit = 'B' 
    mol.atom = f"""
        H    0.    0.    0.
        H    0.    0.    {dist}

    """
    # this basis has 2 functions for Helium
    mol.basis = "ccpvdz"
    #mol.basis = "sto-6g"
    mol.spin =  0
    mol.verbose=0
    mol.build()
    
    # the 2 electron integrals \langle \mu \nu | \kappa \lambda \rangle have M^4 entries
    eri = mol.intor('int2e')
    S = mol.intor('int1e_ovlp')
        
    ## Run Hartree-Fock.
    mf = scf.RHF(mol)
    mf.kernel()

    # Harvesting Fock Properties
    E_nn = mf.energy_nuc()
    C = mf.mo_coeff
    h = mf.get_hcore()    
    N = mol.nelec[0]
    P=np.matmul(C[:,0:N],C[:,0:N].T)

    # Translate Fock Properties into Fock Basis Set 
    #gamma = np.matmul(np.matmul(C.T,np.matmul(np.matmul(S,P),S)), C)
    #occu, naturalC = np.linalg.eigh(gamma)
    
    #  get natural orb
    gamma = np.matmul(np.matmul(C.T,np.matmul(np.matmul(S,P),S)), C)
    occu_aa, naturalC_aa = np.linalg.eigh(gamma)
    occu = np.append(occu_aa, occu_aa)

    # translate into AO basis and expand
    CE = expand_matrix(C)
    naturalCE = expand_matrix(naturalC_aa)
    naturalCTT = np.matmul(CE,naturalCE)
    for i, n  in enumerate(occu):
        if n < 0:
            occu[i] = 0 
            
# Run FCI            
    fs = fci.FCI(mol, mf.mo_coeff)
    e, ci = fs.kernel(verbose=0)

# Preparing Data for the Natural Orbital Functionals
    FCIgamma_a, FCIgamma_b = fci.direct_spin1.make_rdm1s(ci, mf.mo_coeff.shape[0], mol.nelec)
    FCIoccu_a, FCInaturalC_a = np.linalg.eigh(FCIgamma_a)
    FCIoccu_b, FCInaturalC_b = np.linalg.eigh(FCIgamma_b)
    FCInaturalC_a = FCInaturalC_a[:,::-1]
    FCInaturalC_b = FCInaturalC_b[:,::-1]
    FCIoccu_a = FCIoccu_a[::-1]
    FCIoccu_b = FCIoccu_b[::-1]
    FCInaturalCTT_a, FCInaturalCTT_b = np.matmul(C,FCInaturalC_a), np.matmul(C,FCInaturalC_b)
    FCInaturalCTTE = Add_Block_Matrices(FCInaturalCTT_a, FCInaturalCTT_b)
    FCIoccuE = np.append(FCIoccu_a,FCIoccu_b)

# Sorting Out FCI results
    E_HF = mf.e_tot
    FCI_tot = e
    FCI_c = e-E_HF
    h1 = np.trace(np.matmul(h,np.matmul(C, np.matmul(FCIgamma_a,C.T)))) + np.trace(np.matmul(h,np.matmul(C, np.matmul(FCIgamma_b,C.T))))
    FCI_Vee = FCI_tot - h1

# calling 1RDMFT energy functions
    GU_tot,GU_Vee,GU_E_c = energy_components_umrigar(eri, FCInaturalCTTE, FCIoccuE,h1,E_HF,E_nn)
    Mu_tot,Mu_Vee,Mu_E_c = energy_components_mueller(eri, FCInaturalCTTE, FCIoccuE,h1,E_HF,E_nn)
    BBC1_tot,BBC1_Vee,BBC1_E_c = energy_components_bbc1(eri, FCInaturalCTTE, FCIoccuE,h1,E_HF,E_nn,mol.nelec)


    stats.append([dist, GU_E_c, Mu_E_c, BBC1_E_c, FCI_c, GU_tot, Mu_tot, BBC1_tot, FCI_tot, GU_Vee, Mu_Vee, BBC1_Vee,FCI_Vee]) 
    tab.append([dist, GU_Vee, Mu_Vee, BBC1_Vee,FCI_Vee]) 

print(tabulate(tab,headers=tab_top.split()))


