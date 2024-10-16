! this is the dignature file of the parallelized 1RDMFTs

SUBROUTINE WRAP_HARTREE(M,n,eri)
IMPLICIT NONE
INTEGER,  INTENT(IN)  :: M
REAL(8),  INTENT(IN)  :: eri(M,M,M,M), C(M,M), n(M)
REAL(8),  INTENT(OUT)  :: energy

CALL hartree_energy(M,n,eri,res)

END SUBROUTINE
