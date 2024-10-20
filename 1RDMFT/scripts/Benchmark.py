import numpy as np
from pyscf import gto, scf, fci
from tabulate import tabulate
from RDMFTs import energy_components_umrigar, energy_components_mueller, energy_components_bbc1, energy_components_bbc2, energy_components_bbc3
from utils import expand_matrix, Add_Block_Matrices 

PSE = {'H': 1,'He': 2,'Li': 3,'Be': 4,'B': 5,'C': 6,'N': 7,'O': 8,'F': 9,'Ne': 10,\
       'Na': 11,'Mg': 12,'Al': 13,'Si': 14,'P': 15,'S': 16,'Cl': 17,'Ar': 18,'K': 19,\
       'Ca': 20,'Sc': 21,'Ti': 22,'V': 23,'Cr': 24,'Mn': 25,'Fe': 26,'Co': 27,'Ni': 28,\
       'Cu': 29,'Zn': 30,'Ga': 31,'Ge': 32,'As': 33,'Se': 34,'Br': 35,'Kr': 36}
multiplicities = [2, 1, 2, 1, 2, 3, 4, 3, 2, 1, 2, 1, 2, 3, 4, 3, 2, 1, 2, 1, 2, 3, 4, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 3, 2, 1]
spins = [ i-1 for i in multiplicities ]
top = "El. Ec_{GU} Ec_{Mu} Ec_{FCI} Etot_{GU} Etot_{Mu} Etot_{FCI} Vee_{GU} Vee_{Mu} Vee_{FCI}"

top = "el. Ec_{GU} Ec_{Mu}  Ec_{BBC1} Ec_{FCI} Etot_{GU} Etot_{Mu} Etot_{BBC1} Etot_{FCI} Vee_{GU} Vee_{Mu} Vee_{BBC1} Vee_{FCI}"
stats = []
for el in list(PSE.keys())[0:10]:
    mol = gto.Mole()
    mol.atom = f"""
        {el}    0.    0.    0.
    """
    # this basis has 2 functions for Helium
    mol.basis = "ccpv5z"
    #mol.basis = "sto-6g"
    mol.spin =  spins [PSE[el]-1 ] 
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
    for i, n  in enumerate(FCIoccuE):
        if n < 0:
            FCIoccuE[i] = 0 
    
# Sorting Out FCI results
    E_HF = mf.e_tot
    FCI_tot = e
    FCI_c = e-E_HF
    h1 = np.trace(np.matmul(h,np.matmul(C, np.matmul(FCIgamma_a,C.T)))) + np.trace(np.matmul(h,np.matmul(C, np.matmul(FCIgamma_b,C.T))))
    FCI_Vee = FCI_tot - h1

# calling 1RDMFT energy functions
    PYTHONIC=False
    GU_tot,GU_Vee,GU_E_c = energy_components_umrigar(eri, FCInaturalCTTE, FCIoccuE,h1,E_HF,E_nn,PYTHONIC)
    Mu_tot,Mu_Vee,Mu_E_c = energy_components_mueller(eri, FCInaturalCTTE, FCIoccuE,h1,E_HF,E_nn,PYTHONIC)
    BBC1_tot, BBC1_Vee, BBC1_E_c = energy_components_bbc1(eri, FCInaturalCTTE, FCIoccuE,h1,E_HF,E_nn,mol.nelec,PYTHONIC)
    BBC2_tot, BBC2_Vee, BBC2_E_c = energy_components_bbc2(eri, FCInaturalCTTE, FCIoccuE,h1,E_HF,E_nn,mol.nelec,PYTHONIC)
    BBC3_tot, BBC3_Vee, BBC3_E_c = energy_components_bbc3(eri, FCInaturalCTTE, FCIoccuE,h1,E_HF,E_nn,mol.nelec,PYTHONIC)

    #print(f"{el:2s}  {BBC1_E_c:1.6f} {BBC1_E_c_2:1.6f}  {BBC1_tot:3.6f} {BBC1_tot_2:3.6f} {BBC1_Vee:3.6f} {BBC1_Vee_2:3.6f}  ")

    print(f"{el:2s}\
            {GU_E_c:1.6f} {Mu_E_c:1.6f} {BBC1_E_c:1.6f} {BBC2_E_c:1.6f} {BBC3_E_c:1.6f} {FCI_c:1.6f}\
            {GU_tot:3.6f} {Mu_tot:3.6f} {BBC1_tot:3.6f} {BBC2_tot:3.6f} {BBC3_tot:3.6f} {FCI_tot:3.6f}\
            {GU_Vee:3.6f} {Mu_Vee:3.6f} {BBC1_Vee:3.6f} {BBC2_Vee:3.6f} {BBC3_Vee:3.6f} {FCI_Vee:3.6f}", flush=True)
    stats.append([el, GU_E_c, Mu_E_c, BBC1_E_c, FCI_c, GU_tot, Mu_tot, BBC1_tot, FCI_tot, GU_Vee, Mu_Vee, BBC1_Vee,FCI_Vee]) 
