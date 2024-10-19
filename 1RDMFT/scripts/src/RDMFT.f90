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

SUBROUTINE WRAP_GU_XC(n,C,eri,res,M)
IMPLICIT NONE
INTEGER,  INTENT(IN)  :: M
REAL(8),  INTENT(IN)  :: n(2*M), C(2*M,2*M), eri(M,M,M,M) 
REAL(8),  INTENT(OUT)  :: res

CALL umrigar_exchange_correlation_energy(M,n,C,eri,res)

END SUBROUTINE

SUBROUTINE WRAP_MU_XC(n,C,eri,res,M)
IMPLICIT NONE
INTEGER,  INTENT(IN)  :: M
REAL(8),  INTENT(IN)  :: n(2*M), C(2*M,2*M), eri(M,M,M,M) 
REAL(8),  INTENT(OUT)  :: res

CALL mueller_exchange_correlation_energy(M,n,C,eri,res)

END SUBROUTINE

SUBROUTINE WRAP_BBC_1(na,nb,n,C,eri,res,M)
IMPLICIT NONE
INTEGER,  INTENT(IN)  :: M,na,nb
REAL(8),  INTENT(IN)  :: n(2*M), C(2*M,2*M), eri(M,M,M,M)
REAL(8),  INTENT(OUT)  :: res

CALL Buijse_Baerends_Correction_One(M,n,C,eri,na,nb,res)

END SUBROUTINE

SUBROUTINE WRAP_BBC_2(na,nb,n,C,eri,res,M)
IMPLICIT NONE
INTEGER,  INTENT(IN)  :: M,na,nb
REAL(8),  INTENT(IN)  :: n(2*M), C(2*M,2*M), eri(M,M,M,M)
REAL(8),  INTENT(OUT)  :: res

CALL Buijse_Baerends_Correction_Two(M,n,C,eri,na,nb,res)

END SUBROUTINE

