#include <rotatingMHD/convection_diffusion_solver.h>

#include <deal.II/dofs/dof_tools.h>
#ifdef USE_PETSC_LA
  #include <deal.II/lac/dynamic_sparsity_pattern.h>
  #include <deal.II/lac/sparsity_tools.h>
#else
  #include <deal.II/lac/trilinos_sparsity_pattern.h>
#endif
#include <deal.II/numerics/vector_tools.h>

namespace RMHD
{
template <int dim>
void ConvectionDiffusionSolver<dim>::setup()
{
  TimerOutput::Scope  t(*computing_timer, "Convection diffusion: Setup");

  setup_matrices();

  setup_vectors();

  flag_setup_problem = false;
  flag_assemble_matrices = true;
}



template <int dim>
void ConvectionDiffusionSolver<dim>::setup_matrices()
{
  if (parameters.verbose)
    *pcout << "  Heat Equation: Setting up matrices...";

  mass_matrix.clear();
  stiffness_matrix.clear();
  advection_matrix.clear();
  system_matrix.clear();

  {
    #ifdef USE_PETSC_LA
      DynamicSparsityPattern
      sparsity_pattern(temperature->get_locally_relevant_dofs());

      DoFTools::make_sparsity_pattern(
        temperature->get_dof_handler(),
        sparsity_pattern,
        temperature->get_constraints(),
        false,
        Utilities::MPI::this_mpi_process(mpi_communicator));

      SparsityTools::distribute_sparsity_pattern
      (sparsity_pattern,
       temperature->get_locally_owned_dofs(),
       mpi_communicator,
       temperature->get_locally_relevant_dofs());

      mass_matrix.reinit
      (temperature->get_locally_owned_dofs(),
       temperature->get_locally_owned_dofs(),
       sparsity_pattern,
       mpi_communicator);
      stiffness_matrix.reinit
      (temperature->get_locally_owned_dofs(),
       temperature->get_locally_owned_dofs(),
       sparsity_pattern,
       mpi_communicator);
      mass_plus_stiffness_matrix.reinit
      (temperature->get_locally_owned_dofs(),
       temperature->get_locally_owned_dofs(),
       sparsity_pattern,
       mpi_communicator);
      advection_matrix.reinit
      (temperature->get_locally_owned_dofs(),
       temperature->get_locally_owned_dofs(),
       sparsity_pattern,
       mpi_communicator);
      system_matrix.reinit
      (temperature->get_locally_owned_dofs(),
       temperature->get_locally_owned_dofs(),
       sparsity_pattern,
       mpi_communicator);

    #else
      TrilinosWrappers::SparsityPattern
      sparsity_pattern(temperature->get_locally_owned_dofs(),
                       temperature->get_locally_owned_dofs(),
                       temperature->get_locally_relevant_dofs(),
                       mpi_communicator);

      DoFTools::make_sparsity_pattern(
        temperature->get_dof_handler(),
        sparsity_pattern,
        temperature->get_constraints(),
        false,
        Utilities::MPI::this_mpi_process(mpi_communicator));

      sparsity_pattern.compress();

      mass_matrix.reinit(sparsity_pattern);
      stiffness_matrix.reinit(sparsity_pattern);
      mass_plus_stiffness_matrix.reinit(sparsity_pattern);
      advection_matrix.reinit(sparsity_pattern);
      system_matrix.reinit(sparsity_pattern);

    #endif
  }

  if (parameters.verbose)
    *pcout << " done!" << std::endl;
}



template <int dim>
void ConvectionDiffusionSolver<dim>::
setup_vectors()
{
  if (parameters.verbose)
    *pcout << "  Heat Equation: Setting up vectors...";

  // Initializing the temperature related vectors
  rhs.reinit(temperature->distributed_vector);

  if (parameters.verbose)
    *pcout << " done!" << std::endl;
}

} // namespace RMHD

// explicit instantiations
template void RMHD::ConvectionDiffusionSolver<2>::setup();
template void RMHD::ConvectionDiffusionSolver<3>::setup();

template void RMHD::ConvectionDiffusionSolver<2>::setup_matrices();
template void RMHD::ConvectionDiffusionSolver<3>::setup_matrices();

template void RMHD::ConvectionDiffusionSolver<2>::setup_vectors();
template void RMHD::ConvectionDiffusionSolver<3>::setup_vectors();
