#include <rotatingMHD/convection_diffusion_solver.h>
#include <rotatingMHD/utility.h>

namespace RMHD
{

template <int dim>
void ConvectionDiffusionSolver<dim>::solve()
{
  AssertThrow(flag_setup_problem == false,
              ExcMessage("ConvectionDiffusionSolver<dim>::setup was not called."));

  assemble_linear_system();

  flag_update_preconditioner |= (time_stepping.get_step_number() %
                                 parameters.preconditioner_update_frequency == 0);
  flag_update_preconditioner |= (time_stepping.get_step_number() == 1);

  solve_linear_system();
}

template <int dim>
void ConvectionDiffusionSolver<dim>::assemble_linear_system()
{
  computing_timer->enter_subsection("Convection diffusion: Matrix assembly");

  // System matrix setup
  if (flag_assemble_matrices)
    assemble_constant_matrices();
  if (time_stepping.coefficients_changed() == true || flag_assemble_matrices)
  {
    mass_plus_stiffness_matrix = 0.;

    mass_plus_stiffness_matrix.add(
      time_stepping.get_alpha()[0] / time_stepping.get_next_step_size(),
      mass_matrix);

    mass_plus_stiffness_matrix.add(
      time_stepping.get_gamma()[0] * parameters.equation_coefficient,
      stiffness_matrix);

    flag_update_preconditioner = true;
  }
  flag_assemble_matrices = false;

  if ((parameters.time_discretization ==
       RunTimeParameters::ConvectiveTermTimeDiscretization::semi_implicit) &&
      (velocity != nullptr || velocity_function_ptr != nullptr))
  {
    assemble_advection_matrix();
    system_matrix.copy_from(mass_plus_stiffness_matrix);
    system_matrix.add(1.0, advection_matrix);
  }
  computing_timer->leave_subsection();

  // Right hand side setup
  assemble_rhs();
  rhs_norm = rhs.l2_norm();
}

template <int dim>
void ConvectionDiffusionSolver<dim>::solve_linear_system()
{
  if (parameters.verbose)
  *pcout << "  Heat Equation: Solving linear system...";

  TimerOutput::Scope  t(*computing_timer, "Convection diffusion: Solve");

  // In this method we create temporal non ghosted copies
  // of the pertinent vectors to be able to perform the solve()
  // operation.
  LinearAlgebra::MPI::Vector distributed_temperature(rhs);
  distributed_temperature = temperature->solution;

  // The following pointer holds the address to the correct matrix
  // depending on if the semi-implicit scheme is chosen or not
  const LinearAlgebra::MPI::SparseMatrix  *system_matrix_ptr;
  if ((parameters.time_discretization ==
       RunTimeParameters::ConvectiveTermTimeDiscretization::semi_implicit) &&
      (velocity != nullptr || velocity_function_ptr != nullptr))
    system_matrix_ptr = &system_matrix;
  else
    system_matrix_ptr = &mass_plus_stiffness_matrix;


  const typename RunTimeParameters::LinearSolverParameters &solver_parameters
    = parameters.linear_solver;

  if (flag_update_preconditioner)
  {
    build_preconditioner(preconditioner,
                         *system_matrix_ptr,
                         solver_parameters.preconditioner_parameters_ptr,
                         (temperature->fe_degree() > 1? true: false));
  }

  AssertThrow(preconditioner != nullptr,
              ExcMessage("The pointer to the heat equation solver's preconditioner has not being initialized."));

  SolverControl solver_control(solver_parameters.n_maximum_iterations,
                               std::max(solver_parameters.relative_tolerance * rhs_norm,
                                        solver_parameters.absolute_tolerance));

  #ifdef USE_PETSC_LA
    LinearAlgebra::SolverGMRES solver(solver_control,
                                      mpi_communicator);
  #else
    LinearAlgebra::SolverGMRES solver(solver_control);
  #endif

  try
  {
    solver.solve(*system_matrix_ptr,
                 distributed_temperature,
                 rhs,
                 *preconditioner);
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception in the solve method of heat equation: " << std::endl
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
    std::cerr << "Unknown exception in the solve method of the heat equation!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::abort();
  }

  temperature->get_constraints().distribute(distributed_temperature);

  temperature->solution = distributed_temperature;

  if (parameters.verbose)
    *pcout << " done!" << std::endl
           << "    Number of GMRES iterations: "
           << solver_control.last_step()
           << ", Final residual: " << solver_control.last_value() << "."
           << std::endl << std::endl;
}

} // namespace RMHD

// explicit instantiations
template void RMHD::ConvectionDiffusionSolver<2>::solve();
template void RMHD::ConvectionDiffusionSolver<3>::solve();

template void RMHD::ConvectionDiffusionSolver<2>::assemble_linear_system();
template void RMHD::ConvectionDiffusionSolver<3>::assemble_linear_system();

template void RMHD::ConvectionDiffusionSolver<2>::solve_linear_system();
template void RMHD::ConvectionDiffusionSolver<3>::solve_linear_system();
