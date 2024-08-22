import numpy as np
from pyscf import gto, scf, ao2mo

def hartree_energy(Fouridx, C):
    energy = 0 
    M = Fouridx.shape[0]
    N = 1
    for i in  range(0,N):
        for j in range(0,N):
            for mu in range(0,M):
                for nu in range(0,M):
                    for kappa in range(0,M):
                        for lamda in range(0,M):
                            #print(i,j,mu,nu,kappa,lamda)
                            energy += C[i,mu]*C[j,nu]*C[i,kappa]*C[j,lamda]*Fouridx[mu,nu,kappa,lamda]

    return energy


mol = gto.Mole()
mol.atom = """
    Be    0.    0.    0.
"""
# this basis has 2 functions for Helium
mol.basis = "sto-6g"
#mol.basis = "6-31g"
#mol.basis = "ccpvdz"
mol.build()

# the 2 electron integrals \langle \mu \nu | \kappa \lambda \rangle have M^4 in the case of  case 16 distinct elements
eri = mol.intor('int2e')
print(f"number of elements in 2 electron intergrals/ 4 index matrix {eri.size}, of the {mol.basis}-basis")
print("*"*24)
print(eri.shape)
print("*"*24)
print("2e integrals")
for i,column in enumerate(eri): # these arent really row, column and so on but they do corespond to the indices \mu \nu \kappa \lambda
    for j,row in enumerate(column):
        for k, hyper in enumerate(row):
            for l, element in enumerate(hyper):
                print(f"element <{i}{j}|{k}{l} > is {element}" )
print("*"*24)
print("Overlap Integrals")
S = mol.intor('int1e_ovlp')
print(S)



## Run Hartree-Fock.
mf = scf.RHF(mol)
mf.kernel()

print("*"*24)
print("MO-Coefficent matrix")
print(mf.mo_coeff)
print("*"*24)

print("den ?")
print(np.matmul(mf.mo_coeff.T,mf.mo_coeff))
print("*"*24)


# get j, k and gamma (1RDM) matrix from hf, 
J = mf.get_j() 
K = mf.get_k() 
h = mf.get_hcore()
gamma = mf.make_rdm1()

# this should give the occupation numbers but it doesnt and I dont know why
occu, naturalC = np.linalg.eigh(gamma)
print(naturalC)

# calculate the energy components to see what they are from the matrices the mf object offers you 
print(f"h_0 = {np.trace(np.matmul(h,gamma))} U = {1/2*np.trace(np.matmul(J,gamma))} E_x =  {1/4.*np.trace(np.matmul(K,gamma))}")
print(np.trace(np.matmul(h,gamma))+1/2.*np.trace(np.matmul(J,gamma))-1/4.*np.trace(np.matmul(K,gamma)))

# this should also work
print(hartree_energy(eri, mf.mo_coeff))

#
## Find electron-repulsion integrals (eri).
#eri = ao2mo.kernel(mol, mf.mo_coeff)
#eri = np.asarray(ao2mo.restore(1, eri, mol.nao))

