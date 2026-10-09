! hmatrix-bridges: flush the Fortran runtime's standard output and error units from C++.
!
! Used by the ButterflyPACK adapter's exit handler (GOTCHA(fortran-stop)): when it replaces
! the exit status of a Fortran STOP it ends the process with _Exit, which skips the Fortran
! runtime's own cleanup, so ButterflyPACK's buffered messages ("randomized scheme not
! converged ...") must be flushed first. Must be compiled by the Fortran compiler that built
! ButterflyPACK (one runtime per program).
subroutine hmb_bpack_flush_fortran_output() bind(c, name="hmb_bpack_flush_fortran_output")
  use, intrinsic :: iso_fortran_env, only: output_unit, error_unit
  implicit none
  flush (output_unit)
  flush (error_unit)
end subroutine hmb_bpack_flush_fortran_output
