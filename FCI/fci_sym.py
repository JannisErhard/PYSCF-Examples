#!/usr/bin/env python
#
# Author: Jannis Erhard <jannis.erhard@fau.de>
#

'''
Compare result and perfomance, Symmetrized and Unsymmetrized FCI
'''



import numpy
from pyscf import gto, scf, fci
from pyscf import ao2mo, symm, mcscf
import time
import psutil

process = psutil.Process()

# what is tested
Run_With_Sym, Run_Without_Sym = True, False

# setup of calculations
basis = '631g'
element = 'B'
r = 4.0

# Assign FCI wavefunction symmetry

if Run_With_Sym:
    start_time = time.time() # timing
    mol = gto.M(atom=f'{element} 0 0 0;  {element} 0 0 {r}', basis=basis, symmetry=True, charge=0)
    m = scf.RHF(mol).run()
    norb = m.mo_coeff.shape[1]
    nelec = mol.nelec
    
    fs = fci.FCI(mol, m.mo_coeff)
    fs.wfnsym = 'A1g'
    e, c = fs.kernel(verbose=0)
    print('Energy of %s state %.12f' % (fs.wfnsym, e))
    print("--- %s seconds ---" % (time.time() - start_time))

# Do not assign FCI wavefunction symmetry

if Run_Without_Sym:
    start_time = time.time()
    mol = gto.M(atom='{element} 0 0 0;  {element} 0 0 {r}', basis=basis, symmetry=False, charge=0)
    m = scf.RHF(mol).run()
    norb = m.mo_coeff.shape[1]
    nelec = mol.nelec
    fs = fci.FCI(mol, m.mo_coeff)
    e, c = fs.kernel(verbose=0)
    print('Energy of unsymmetrized result %.12f' % (e))
    print("--- %s seconds ---" % (time.time() - start_time))

print(f"Programs memory usage {process.memory_info().rss/ 1024 ** 2} MiB")  # in bytes 

# Result True, False:
#converged SCF energy = -48.9306158661173
#Energy of A1g state -49.128747751762
#--- 452.160982131958 seconds ---
#Programs memory usage 809.0859375 MiB

