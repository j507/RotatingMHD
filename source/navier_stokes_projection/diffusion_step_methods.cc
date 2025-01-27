#include <rotatingMHD/navier_stokes_projection.h>
#include <rotatingMHD/utility.h>

namespace RMHD
{

template <int dim>
void NavierStokesProjection<dim>::
assemble_diffusion_step()
{
  /* System matrix setup */

  /* This if scope makes sure that if the time step did not change
     between solve calls, the following matrix summation is only done once */
  if (time_stepping.coefficients_changed() == true ||
      flag_matrices_were_updated)
  {
    TimerOutput::Scope  t(*computing_timer, "Navier Stokes: Mass and stiffness matrix addition");
    velocity_mass_plus_laplace_matrix = 0.;

    velocity_mass_plus_laplace_matrix.add
    (time_stepping.get_alpha()[0] / time_stepping.get_next_step_size(),
     velocity_mass_matrix);

    velocity_mass_plus_laplace_matrix.add
    (time_stepping.get_gamma()[0] * parameters.C2,
     velocity_laplace_matrix);
  }

  /* In case of a semi-implicit scheme, the advection matrix has to be
  assembled and added to the system matrix */
  if (parameters.convective_term_time_discretization ==
      RunTimeParameters::ConvectiveTermTimeDiscretization::semi_implicit)
  {
    assemble_velocity_advection_matrix();
    velocity_system_matrix.copy_from(velocity_mass_plus_laplace_matrix);
    velocity_system_matrix.add(1. , velocity_advection_matrix);
  }
  /* Right hand side setup */
  assemble_diffusion_step_rhs();
}

template <int dim>
void NavierStokesProjection<dim>::
solve_diffusion_step(const bool reinit_prec)
{
  if (parameters.verbose)
    *pcout << "  Navier Stokes: Solving the diffusion step...";

  TimerOutput::Scope  t(*computing_timer, "Navier Stokes: Diffusion step - Solve");

  // In this method we create temporal non ghosted copies
  // of the pertinent vectors to be able to perform the solve()
  // operation.
  LinearAlgebra::MPI::Vector distributed_velocity(velocity->distributed_vector);
  distributed_velocity = velocity->solution;

  /* The following pointer holds the address to the correct matrix
  depending on if the semi-implicit scheme is chosen or not */
  const LinearAlgebra::MPI::SparseMatrix  * system_matrix;
  if (parameters.convective_term_time_discretization ==
      RunTimeParameters::ConvectiveTermTimeDiscretization::semi_implicit)
    system_matrix = &velocity_system_matrix;
  else
    system_matrix = &velocity_mass_plus_laplace_matrix;


  const typename RunTimeParameters::LinearSolverParameters &solver_parameters
    = parameters.diffusion_step_solver_parameters;
  if (reinit_prec)
  {
    build_preconditioner(diffusion_step_preconditioner,
                         *system_matrix,
                         solver_parameters.preconditioner_parameters_ptr,
                         (velocity->fe_degree() > 1? true: false));
  }

  AssertThrow(diffusion_step_preconditioner != nullptr,
              ExcMessage("The pointer to the diffusion step's preconditioner has not being initialized."));

  SolverControl solver_control(
    parameters.diffusion_step_solver_parameters.n_maximum_iterations,
    std::max(solver_parameters.relative_tolerance * diffusion_step_rhs.l2_norm(),
             solver_parameters.absolute_tolerance));

  #ifdef USE_PETSC_LA
    LinearAlgebra::SolverGMRES solver(solver_control,
                                      MPI_COMM_WORLD);
  #else
    LinearAlgebra::SolverGMRES solver(solver_control);
  #endif

  try
  {
    solver.solve(*system_matrix,
                 distributed_velocity,
                 diffusion_step_rhs,
                 *diffusion_step_preconditioner);
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception in the solve method of the diffusion step: " << std::endl
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
    std::cerr << "Unknown exception in the solve method of the diffusion step!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::abort();
  }

  velocity->get_constraints().distribute(distributed_velocity);

  velocity->solution = distributed_velocity;

  if (parameters.verbose)
    *pcout << " done!" << std::endl
           << "    Number of GMRES iterations: "
           << solver_control.last_step()
           << ", Final residual: " << solver_control.last_value() << "."
           << std::endl;
}
}
// explicit instantiations
template void RMHD::NavierStokesProjection<2>::assemble_diffusion_step();
template void RMHD::NavierStokesProjection<3>::assemble_diffusion_step();

template void RMHD::NavierStokesProjection<2>::solve_diffusion_step(const bool);
template void RMHD::NavierStokesProjection<3>::solve_diffusion_step(const bool);
