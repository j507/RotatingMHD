#include <rotatingMHD/navier_stokes_projection.h>

namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::
assemble_poisson_prestep()
{
  /* System matrix setup */
  // System matrix is constant and assembled in the
  // NavierStokesProjection constructor.

  /* Right hand side setup */
  assemble_poisson_prestep_rhs();
}

template <int dim>
void
NavierStokesProjection<dim>::
solve_poisson_prestep()
{
  // In this method we create temporal non ghosted copies
  // of the pertinent vectors to be able to perform the solve()
  // operation.
  LinearAlgebra::MPI::Vector distributed_old_old_pressure(pressure_rhs);
  distributed_old_old_pressure = pressure.old_old_solution;

  projection_step_preconditioner.initialize(pressure_laplace_matrix);

  SolverControl solver_control(parameters.n_maximum_iterations,
                               std::max(parameters.relative_tolerance * pressure_rhs.l2_norm(),
                                        absolute_tolerance));

  #ifdef USE_PETSC_LA
    LinearAlgebra::SolverCG solver(solver_control,
                                   MPI_COMM_WORLD);
  #else
    LinearAlgebra::SolverCG solver(solver_control);
  #endif

  try
  {
    solver.solve(pressure_laplace_matrix,
                 distributed_old_old_pressure,
                 poisson_prestep_rhs,
                 projection_step_preconditioner);
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception in the solve method of the projection step: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::abort();
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
                << std::endl;
    std::cerr << "Unknown exception in the solve method of the projection step!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::abort();
  }

  pressure.constraints.distribute(distributed_old_old_pressure);

  pressure.old_old_solution = distributed_old_old_pressure;
}

} // namespace RMHD

// explicit instantiations
template void RMHD::NavierStokesProjection<2>::assemble_poisson_prestep();
template void RMHD::NavierStokesProjection<3>::assemble_poisson_prestep();

template void RMHD::NavierStokesProjection<2>::solve_poisson_prestep();
template void RMHD::NavierStokesProjection<3>::solve_poisson_prestep();