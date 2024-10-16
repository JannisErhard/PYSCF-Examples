MODULE parallel 
use omp_lib
integer :: thread_num
END MODULE parallel

subroutine hartree_energy(M,n,C,eri,res)
use parallel
IMPLICIT NONE

INTEGER, INTENT(IN) :: M
REAL(8), INTENT(IN) :: n(2*M),C(2*M,2*M),eri(M,M,M,M)
REAL(8) :: energy
INTEGER :: a,b,mu,nu,kappa,lambda
REAL(8), INTENT(OUT) :: res

energy = 0d0
!$omp parallel default(firstprivate) &
!$omp private(a,b,mu,nu,kappa,lambda) &
!$omp shared(M,n,C,eri,energy)
!$omp do schedule(static,1) reduction(+:energy)
  DO a=1,2*M
    DO b=1,2*M
      DO mu=1,2*M
        DO nu=1,2*M
          DO kappa=1,2*M
            DO lambda=1,2*M
              energy = energy+n(a)*n(b)&
                      *C(mu,a)*C(nu,a)*C(kappa,b)*C(lambda,b)&
                      *eri(MODULO(mu-1,M)+1,modulo(nu-1,M)+1,modulo(kappa-1,M)+1,modulo(lambda-1,M)+1)
            ENDDO
          ENDDO
        ENDDO
      ENDDO
    ENDDO
  ENDDO
!$omp end do
!$omp end parallel
  res = .5d0*energy
end subroutine
