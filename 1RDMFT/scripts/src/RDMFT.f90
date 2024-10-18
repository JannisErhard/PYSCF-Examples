! this is the dignature file of the parallelized 1RDMFTs

SUBROUTINE WRAP_HARTREE(n,C,eri,res,M)
IMPLICIT NONE
INTEGER,  INTENT(IN)  :: M
REAL(8),  INTENT(IN)  :: n(2*M), C(2*M,2*M), eri(M,M,M,M) 
REAL(8),  INTENT(OUT)  :: res

CALL hartree_energy(M,n,C,eri,res)

END SUBROUTINE

SUBROUTINE WRAP_GU_HARTREE(n,C,eri,res,M)
IMPLICIT NONE
INTEGER,  INTENT(IN)  :: M
REAL(8),  INTENT(IN)  :: n(2*M), C(2*M,2*M), eri(M,M,M,M) 
REAL(8),  INTENT(OUT)  :: res

CALL umrigar_hartree_energy(M,n,C,eri,res)

END SUBROUTINE