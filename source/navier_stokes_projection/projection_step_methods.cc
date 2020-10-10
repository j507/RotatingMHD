#include <rotatingMHD/navier_stokes_projection.h>

#include <deal.II/lac/solver_cg.h>
#include <deal.II/numerics/vector_tools.h>

namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::assemble_projection_step()
{
  if (parameters.verbose)
    *pcout << "    Assemble projection step...";

  /* System matrix setup */
  // System matrix is constant and assembled in the
  // NavierStokesProjection constructor.

  /* Right hand side setup */
  assemble_projection_step_rhs();

  /* Zeros out the DoFs on the boundary where Dirichlet
     boundary conditions on the pressure or Neumann boundary
     conditions on the stress tensor are given
   */
  pressure.constraints.set_zero(pressure_rhs);

  if (parameters.verbose)
    *pcout << "   done." << std::endl;

}

template <int dim>
void NavierStokesProjection<dim>::solve_projection_step
(const bool reinit_prec)
{
  if (parameters.verbose)
    *pcout << "    Solve projection step..." << std::endl;

  TimerOutput::Scope  t(*computing_timer, "Pressure projection solve");

  // In this method we create temporal non ghosted copies
  // of the pertinent vectors to be able to perform the solve()
  // operation.
  LinearAlgebra::MPI::Vector distributed_phi(pressure_rhs);
  distributed_phi = phi;

  if (reinit_prec)
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
                 distributed_phi,
                 pressure_rhs,
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

  pressure.constraints.distribute(distributed_phi);

  if (flag_normalize_pressure)
    VectorTools::subtract_mean_value(distributed_phi);

  phi = distributed_phi;

  if (parameters.verbose)
    *pcout << "   done." << std::endl;

  if (parameters.verbose)
    *pcout << "    Number of GMRES iterations: " << solver_control.last_step()
           << ", "
           << "final residual: " << solver_control.last_value() << "."
           << std::endl;
}

}

// explicit instantiations
template void RMHD::NavierStokesProjection<2>::assemble_projection_step();
template void RMHD::NavierStokesProjection<3>::assemble_projection_step();

template void RMHD::NavierStokesProjection<2>::solve_projection_step(const bool);
template void RMHD::NavierStokesProjection<3>::solve_projection_step(const bool);
