#include <rotatingMHD/assembly_data.h>
#include <rotatingMHD/solver_class.h>
#include <rotatingMHD/utility.h>

#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/filtered_iterator.h>
#ifdef USE_PETSC_LA
  #include <deal.II/lac/dynamic_sparsity_pattern.h>
  #include <deal.II/lac/sparsity_tools.h>
#else
  #include <deal.II/lac/trilinos_sparsity_pattern.h>
#endif
#include <deal.II/numerics/vector_tools.h>


namespace RMHD
{



namespace Solvers
{



using namespace dealii;



template<int dim>
SolverBase<dim>::SolverBase(
  TimeDiscretization::VSIMEXMethod          &time_stepping,
  const std::shared_ptr<Mapping<dim>>       external_mapping,
  const std::shared_ptr<ConditionalOStream> external_pcout,
  const std::shared_ptr<TimerOutput>        external_timer)
:
mpi_communicator(MPI_COMM_WORLD),
time_stepping(time_stepping),
flag_matrices_were_updated(true)
{
  // Initiating the internal Mapping instance.
  if (external_mapping.get() != nullptr)
    mapping = external_mapping;
  else
    mapping = std::make_shared<MappingQ<dim>>(1);

  // Initiating the internal ConditionalOStream instance.
  if (external_pcout.get() != nullptr)
    pcout = external_pcout;
  else
    pcout = std::make_shared<ConditionalOStream>(
      std::cout,
      Utilities::MPI::this_mpi_process(mpi_communicator) == 0);

  // Initiating the internal TimerOutput instance.
  if (external_timer.get() != nullptr)
    computing_timer = external_timer;
  else
    computing_timer = std::make_shared<TimerOutput>(
      *pcout,
      TimerOutput::summary,
      TimerOutput::wall_times);
}



template<int dim>
void SolverBase<dim>::clear()
{
  flag_matrices_were_updated = true;
}




template<int dim>
ProjectionSolverBase<dim>::ProjectionSolverBase(
  const RunTimeParameters::ProjectionSolverParametersBase &parameters,
  TimeDiscretization::VSIMEXMethod                        &time_stepping,
  std::shared_ptr<Entities::FE_VectorField<dim>>          &vector_field,
  std::shared_ptr<Entities::FE_ScalarField<dim>>          &lagrange_multiplier,
  const std::shared_ptr<Mapping<dim>>                     external_mapping,
  const std::shared_ptr<ConditionalOStream>               external_pcout,
  const std::shared_ptr<TimerOutput>                      external_timer)
:
SolverBase<dim>(time_stepping,
                external_mapping,
                external_pcout,
                external_timer),
projection_solver_parameters(parameters),
vector_field(vector_field),
lagrange_multiplier(lagrange_multiplier),
norm_diffusion_step_rhs(std::numeric_limits<double>::lowest()),
norm_projection_step_rhs(std::numeric_limits<double>::lowest()),
flag_setup_auxiliary_scalar(true),
flag_mean_value_constrain(false)
{
  // Explicitly set the supply term's pointer to null
  ptr_supply_term = nullptr;
}



template<int dim>
ProjectionSolverBase<dim>::ProjectionSolverBase(
  const RunTimeParameters::ProjectionSolverParametersBase &parameters,
  TimeDiscretization::VSIMEXMethod                        &time_stepping,
  std::shared_ptr<Entities::FE_VectorField<dim>>          &vector_field,
  const std::shared_ptr<Mapping<dim>>                     external_mapping,
  const std::shared_ptr<ConditionalOStream>               external_pcout,
  const std::shared_ptr<TimerOutput>                      external_timer)
:
SolverBase<dim>(time_stepping,
                external_mapping,
                external_pcout,
                external_timer),
projection_solver_parameters(parameters),
vector_field(vector_field),
norm_diffusion_step_rhs(std::numeric_limits<double>::lowest()),
norm_projection_step_rhs(std::numeric_limits<double>::lowest()),
flag_setup_auxiliary_scalar(true),
flag_mean_value_constrain(false)
{
  // Explicitly set the supply term's pointer to null
  ptr_supply_term = nullptr;
}



template <int dim>
void ProjectionSolverBase<dim>::solve()
{
  if (vector_field->solution.size() != diffusion_step_rhs.size())
  {
    setup();

    diffusion_step(true);

    projection_step(true);

    correction_step(true);

    this->flag_matrices_were_updated = false;
  }
  else
  {
    diffusion_step(this->time_stepping.get_step_number() %
                   projection_solver_parameters.preconditioner_update_frequency == 0 ||
                   this->time_stepping.get_step_number() == 1);

    projection_step(false);

    correction_step(false);
  }

  auxiliary_scalar->update_solution_vectors();

  previous_alpha_zeros[1] = previous_alpha_zeros[0];
  previous_alpha_zeros[0] = this->time_stepping.get_alpha()[0];
  previous_step_sizes[1]  = previous_step_sizes[0];
  previous_step_sizes[0]  = this->time_stepping.get_next_step_size();
}



template <int dim>
void ProjectionSolverBase<dim>::clear()
{
  // Pointers
  ptr_supply_term = nullptr;

  // Auxiliary scalar
  auxiliary_scalar->clear();

  // Preconditioners
  diffusion_step_preconditioner.reset();
  projection_step_preconditioner.reset();
  zeroth_step_preconditioner.reset();

  // Matrices
  diffusion_step_system_matrix.clear();
  diffusion_step_mass_matrix.clear();
  diffusion_step_stiffness_matrix.clear();
  diffusion_step_mass_plus_stiffness_matrix.clear();
  diffusion_step_advection_matrix.clear();
  projection_step_system_matrix.clear();
  zeroth_step_system_matrix.clear();

  // Vectors
  diffusion_step_rhs.clear();
  projection_step_rhs.clear();
  zeroth_step_rhs.clear();

  // Norms
  norm_projection_step_rhs  = std::numeric_limits<double>::lowest();
  norm_diffusion_step_rhs   = std::numeric_limits<double>::lowest();

  // Flags
  flag_setup_auxiliary_scalar       = true;
  flag_mean_value_constrain         = false;

  SolverBase<dim>::clear();
}


template <int dim>
void ProjectionSolverBase<dim>::set_supply_term(
  TensorFunction<1, dim> &supply_term)
{
  ptr_supply_term = &supply_term;
}



template <int dim>
void ProjectionSolverBase<dim>::setup()
{
  if (flag_setup_auxiliary_scalar)
    setup_auxiliary_scalar();

  this->setup_matrices();

  this->setup_vectors();

  assemble_constant_matrices();

  if (auxiliary_scalar->get_dirichlet_boundary_conditions().empty())
    flag_mean_value_constrain = true;

  this->flag_matrices_were_updated = true;

  if (this->time_stepping.get_step_number() == 0)
    zeroth_step();
}



template <int dim>
void ProjectionSolverBase<dim>::setup_matrices()
{
  setup_matrices_vector_field();

  setup_matrices_scalar_fields();
}



template <int dim>
void ProjectionSolverBase<dim>::setup_matrices_vector_field()
{
  // Clear the matrices
  diffusion_step_system_matrix.clear();
  diffusion_step_mass_matrix.clear();
  diffusion_step_stiffness_matrix.clear();
  diffusion_step_mass_plus_stiffness_matrix.clear();
  diffusion_step_advection_matrix.clear();

  #ifdef USE_PETSC_LA
    // Initiate the sparsity pattern
    DynamicSparsityPattern
    sparsity_pattern(vector_field->get_locally_relevant_dofs());

    // Create the sparsity pattern
    DoFTools::make_sparsity_pattern(vector_field->get_dof_handler(),
                                    sparsity_pattern,
                                    vector_field->get_constraints(),
                                    false,
                                    Utilities::MPI::this_mpi_process(this->mpi_communicator));

    // Distribute the sparsity pattern
    SparsityTools::distribute_sparsity_pattern
    (sparsity_pattern,
     vector_field->get_locally_owned_dofs(),
     this->mpi_communicator,
     vector_field->get_locally_relevant_dofs());

    // Re-initate the matrices
    diffusion_step_system_matrix.reinit
    (vector_field->get_locally_owned_dofs(),
     vector_field->get_locally_owned_dofs(),
     sparsity_pattern,
     this->mpi_communicator);
    diffusion_step_mass_matrix.reinit
    (vector_field->get_locally_owned_dofs(),
     vector_field->get_locally_owned_dofs(),
     sparsity_pattern,
     this->mpi_communicator);
    diffusion_step_stiffness_matrix.reinit
    (vector_field->get_locally_owned_dofs(),
     vector_field->get_locally_owned_dofs(),
     sparsity_pattern,
     this->mpi_communicator);
    diffusion_step_mass_plus_stiffness_matrix.reinit
    (vector_field->get_locally_owned_dofs(),
     vector_field->get_locally_owned_dofs(),
     sparsity_pattern,
     this->mpi_communicator);
    diffusion_step_advection_matrix.reinit
    (vector_field->get_locally_owned_dofs(),
     vector_field->get_locally_owned_dofs(),
     sparsity_pattern,
     this->mpi_communicator);

  #else
    // Initiate sparsity pattern
    TrilinosWrappers::SparsityPattern
    sparsity_pattern(vector_field->get_locally_owned_dofs(),
                     vector_field->get_locally_owned_dofs(),
                     vector_field->get_locally_relevant_dofs(),
                     this->mpi_communicator);

    // Make sparsity pattern
    DoFTools::make_sparsity_pattern(vector_field->get_dof_handler(),
                                    sparsity_pattern,
                                    vector_field->get_constraints(),
                                    false,
                                    Utilities::MPI::this_mpi_process(this->mpi_communicator));

    // Compress sparsity pattern
    sparsity_pattern.compress();

    // Re-initiate matrices
    diffusion_step_system_matrix.reinit(sparsity_pattern);
    diffusion_step_mass_matrix.reinit(sparsity_pattern);
    diffusion_step_stiffness_matrix.reinit(sparsity_pattern);
    diffusion_step_mass_plus_stiffness_matrix.reinit(sparsity_pattern);
    diffusion_step_advection_matrix.reinit(sparsity_pattern);
  #endif
}



template <int dim>
void ProjectionSolverBase<dim>::setup_matrices_scalar_fields()
{
  // Clear matrices
  projection_step_system_matrix.clear();
  zeroth_step_system_matrix.clear();

  #ifdef USE_PETSC_LA
    // Initiate the sparsity patterns
    DynamicSparsityPattern
    lagrange_multiplier_sparsity_pattern(lagrange_multiplier->get_locally_relevant_dofs());
    DynamicSparsityPattern
    auxiliary_scalar_sparsity_pattern(auxiliary_scalar->get_locally_relevant_dofs());

    // Make the sparsity patterns
    DoFTools::make_sparsity_pattern(lagrange_multiplier->get_dof_handler(),
                                    lagrange_multiplier_sparsity_pattern,
                                    lagrange_multiplier->get_constraints(),
                                    false,
                                    Utilities::MPI::this_mpi_process(mpi_communicator));
    DoFTools::make_sparsity_pattern(auxiliary_scalar->get_dof_handler(),
                                    auxiliary_scalar_sparsity_pattern,
                                    auxiliary_scalar->get_constraints(),
                                    false,
                                    Utilities::MPI::this_mpi_process(mpi_communicator));

    // Distribute the sparsity patterns
    SparsityTools::distribute_sparsity_pattern
    (lagrange_multiplier_sparsity_pattern,
     lagrange_multiplier->get_locally_owned_dofs(),
     this->mpi_communicator,
     lagrange_multiplier->get_locally_relevant_dofs());
    SparsityTools::distribute_sparsity_pattern
    (auxiliary_scalar_sparsity_pattern,
     auxiliary_scalar->get_locally_owned_dofs(),
     this->mpi_communicator,
     auxiliary_scalar->locally_relevant_dofs);

    // Re-initiate the matrices
    zeroth_step_system_matrix.reinit
    (lagrange_multiplier->get_locally_owned_dofs(),
     lagrange_multiplier->get_locally_owned_dofs(),
     lagrange_multiplier_sparsity_pattern,
     this->mpi_communicator);
    projection_step_system_matrix.reinit
    (auxiliary_scalar->get_locally_owned_dofs(),
     auxiliary_scalar->get_locally_owned_dofs(),
     auxiliary_scalar_sparsity_pattern,
     this->mpi_communicator);

  #else
    TrilinosWrappers::SparsityPattern
    lagrange_multiplier_sparsity_pattern(
      lagrange_multiplier->get_locally_owned_dofs(),
      lagrange_multiplier->get_locally_owned_dofs(),
      lagrange_multiplier->get_locally_relevant_dofs(),
      this->mpi_communicator);

    TrilinosWrappers::SparsityPattern
    auxiliary_scalar_sparsity_pattern(
      auxiliary_scalar->get_locally_owned_dofs(),
      auxiliary_scalar->get_locally_owned_dofs(),
      auxiliary_scalar->get_locally_relevant_dofs(),
      this->mpi_communicator);

    DoFTools::make_sparsity_pattern(
      lagrange_multiplier->get_dof_handler(),
      lagrange_multiplier_sparsity_pattern,
      lagrange_multiplier->get_constraints(),
      false,
      Utilities::MPI::this_mpi_process(this->mpi_communicator));

    DoFTools::make_sparsity_pattern(
      auxiliary_scalar->get_dof_handler(),
      auxiliary_scalar_sparsity_pattern,
      auxiliary_scalar->get_constraints(),
      false,
      Utilities::MPI::this_mpi_process(this->mpi_communicator));

    lagrange_multiplier_sparsity_pattern.compress();
    auxiliary_scalar_sparsity_pattern.compress();

    zeroth_step_system_matrix.reinit(lagrange_multiplier_sparsity_pattern);
    projection_step_system_matrix.reinit(auxiliary_scalar_sparsity_pattern);

  #endif
}



template <int dim>
void ProjectionSolverBase<dim>::setup_vectors()
{
  diffusion_step_rhs.reinit(vector_field->distributed_vector);
  projection_step_rhs.reinit(auxiliary_scalar->distributed_vector);
  zeroth_step_rhs.reinit(lagrange_multiplier->distributed_vector);
}



template <int dim>
void ProjectionSolverBase<dim>::zeroth_step()
{
  if (ptr_supply_term != nullptr)
    ptr_supply_term->set_time(this->time_stepping.get_start_time());

  assemble_zeroth_step();

  solve_zeroth_step();
}



template <int dim>
std::pair<int, double> ProjectionSolverBase<dim>::solve_zeroth_step()
{
  LinearAlgebra::MPI::Vector distributed_lagrange_multiplier(lagrange_multiplier->distributed_vector);
  distributed_lagrange_multiplier = lagrange_multiplier->old_solution;

  const typename RunTimeParameters::LinearSolverParameters &solver_parameters
    = projection_solver_parameters.zeroth_step_solver_parameters;

  build_preconditioner(zeroth_step_preconditioner,
                       zeroth_step_system_matrix,
                       solver_parameters.preconditioner_parameters_ptr,
                       (lagrange_multiplier->fe_degree() > 1? true: false));

  AssertThrow(zeroth_step_preconditioner != nullptr,
              ExcMessage("The pointer to the Poisson pre-step's preconditioner has not being initialized."));

  SolverControl solver_control(
    solver_parameters.n_maximum_iterations,
    std::max(solver_parameters.relative_tolerance * zeroth_step_rhs.l2_norm(),
             solver_parameters.absolute_tolerance));

  #ifdef USE_PETSC_LA
    LinearAlgebra::SolverCG solver(solver_control,
                                   this->mpi_communicator);
  #else
    LinearAlgebra::SolverCG solver(solver_control);
  #endif

  try
  {
    solver.solve(zeroth_step_system_matrix,
                 distributed_lagrange_multiplier,
                 zeroth_step_rhs,
                 *zeroth_step_preconditioner);
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception in the solve method of the zeroth step: " << std::endl
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
    std::cerr << "Unknown exception in the solve method of the  zeroth step!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::abort();
  }

  lagrange_multiplier->get_constraints().distribute(distributed_lagrange_multiplier);

  lagrange_multiplier->old_solution = distributed_lagrange_multiplier;

  if (flag_mean_value_constrain)
  {
    const LinearAlgebra::MPI::Vector::value_type mean_value
      = VectorTools::compute_mean_value(lagrange_multiplier->get_dof_handler(),
                                        QGauss<dim>(lagrange_multiplier->fe_degree() + 1),
                                        lagrange_multiplier->old_solution,
                                        0);

    distributed_lagrange_multiplier.add(-mean_value);
    lagrange_multiplier->old_solution = distributed_lagrange_multiplier;
  }

  return {solver_control.last_step(), solver_control.last_value()};
}


template <int dim>
void ProjectionSolverBase<dim>::diffusion_step(const bool reinit_preconditioner)
{
  assemble_diffusion_step();

  solve_diffusion_step(reinit_preconditioner);
}



template <int dim>
std::pair<int, double> ProjectionSolverBase<dim>::solve_diffusion_step(const bool reinit_preconditioner)
{
  LinearAlgebra::MPI::Vector distributed_vector_field(vector_field->distributed_vector);
  distributed_vector_field = vector_field->solution;

  /* The following pointer holds the address to the correct matrix
  depending on if the semi-implicit scheme is chosen or not */
  const LinearAlgebra::MPI::SparseMatrix  * system_matrix;

  if (projection_solver_parameters.convective_term_time_discretization ==
      RunTimeParameters::ConvectiveTermTimeDiscretization::semi_implicit)
    system_matrix = &diffusion_step_system_matrix;
  else
    system_matrix = &diffusion_step_mass_plus_stiffness_matrix;


  const typename RunTimeParameters::LinearSolverParameters &solver_parameters
    = projection_solver_parameters.diffusion_step_solver_parameters;
  if (reinit_preconditioner)
  {
    build_preconditioner(diffusion_step_preconditioner,
                         *system_matrix,
                         solver_parameters.preconditioner_parameters_ptr,
                         (vector_field->fe_degree() > 1? true: false));
  }

  AssertThrow(diffusion_step_preconditioner != nullptr,
              ExcMessage("The pointer to the diffusion step's preconditioner has not being initialized."));

  SolverControl solver_control(
    projection_solver_parameters.diffusion_step_solver_parameters.n_maximum_iterations,
    std::max(solver_parameters.relative_tolerance * diffusion_step_rhs.l2_norm(),
             solver_parameters.absolute_tolerance));

  #ifdef USE_PETSC_LA
    LinearAlgebra::SolverGMRES solver(solver_control,
                                      this->mpi_communicator);
  #else
    LinearAlgebra::SolverGMRES solver(solver_control);
  #endif

  try
  {
    solver.solve(*system_matrix,
                 distributed_vector_field,
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

  vector_field->get_constraints().distribute(distributed_vector_field);

  vector_field->solution = distributed_vector_field;

  return {solver_control.last_step(), solver_control.last_value()};
}



template <int dim>
void ProjectionSolverBase<dim>::projection_step(const bool reinit_preconditioner)
{
  assemble_projection_step();

  solve_projection_step(reinit_preconditioner);
}



template <int dim>
std::pair<int, double>  ProjectionSolverBase<dim>::solve_projection_step(const bool reinit_preconditioner)
{
  LinearAlgebra::MPI::Vector distributed_auxiliary_scalar(auxiliary_scalar->distributed_vector);
  distributed_auxiliary_scalar = auxiliary_scalar->solution;

  const typename RunTimeParameters::LinearSolverParameters &solver_parameters
    = projection_solver_parameters.projection_step_solver_parameters;
  if (reinit_preconditioner)
  {
    build_preconditioner(projection_step_preconditioner,
                         projection_step_system_matrix,
                         solver_parameters.preconditioner_parameters_ptr,
                         (auxiliary_scalar->fe_degree() > 1? true: false));
  }

  AssertThrow(projection_step_preconditioner != nullptr,
              ExcMessage("The pointer to the projection step's preconditioner has not being initialized."));

  SolverControl solver_control(
    solver_parameters.n_maximum_iterations,
    std::max(solver_parameters.relative_tolerance * projection_step_rhs.l2_norm(),
             solver_parameters.absolute_tolerance));

  #ifdef USE_PETSC_LA
    LinearAlgebra::SolverCG solver(solver_control,
                                   this->mpi_communicator);
  #else
    LinearAlgebra::SolverCG solver(solver_control);
  #endif

  try
  {
    solver.solve(projection_step_system_matrix,
                 distributed_auxiliary_scalar,
                 projection_step_rhs,
                 *projection_step_preconditioner);
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

  auxiliary_scalar->get_constraints().distribute(distributed_auxiliary_scalar);

  auxiliary_scalar->solution = distributed_auxiliary_scalar;

  if (flag_mean_value_constrain)
  {
    const LinearAlgebra::MPI::Vector::value_type mean_value
      = VectorTools::compute_mean_value(auxiliary_scalar->get_dof_handler(),
                                        QGauss<dim>(auxiliary_scalar->fe_degree() + 1),
                                        auxiliary_scalar->solution,
                                        0);

    distributed_auxiliary_scalar.add(-mean_value);
    auxiliary_scalar->solution = distributed_auxiliary_scalar;
  }

  return {solver_control.last_step(), solver_control.last_value()};
}



template <int dim>
void ProjectionSolverBase<dim>::assemble_constant_matrices()
{
  assemble_constant_matrices_vector_field();

  assemble_constant_matrices_scalar_fields();
}



template <int dim>
void ProjectionSolverBase<dim>::assemble_constant_matrices_vector_field()
{
  // Reset data
  diffusion_step_mass_matrix      = 0.;
  diffusion_step_stiffness_matrix = 0.;

  // Initiate the quadrature formula for exact numerical integration
  const QGauss<dim>   quadrature_formula(vector_field->fe_degree() + 1);

  // Set up the lambda function for the local assembly operation
  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
           AssemblyData::ProjectionSolverBase::VectorFieldConstantMatrices::Scratch<dim> &scratch,
           AssemblyData::ProjectionSolverBase::VectorFieldConstantMatrices::Copy    &data)
    {
      this->assemble_local_matrices_vector_field(cell,
                                                 scratch,
                                                 data);
    };

  // Set up the lambda function for the copy local to global operation
  auto copier =
    [this](const AssemblyData::ProjectionSolverBase::VectorFieldConstantMatrices::Copy &data)
    {
      this->copy_local_to_global_matrices_vector_field(data);
    };

  // Assemble using the WorkStream approach
  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  WorkStream::run
  (CellFilter(IteratorFilters::LocallyOwnedCell(),
              vector_field->get_dof_handler().begin_active()),
   CellFilter(IteratorFilters::LocallyOwnedCell(),
              vector_field->get_dof_handler().end()),
   worker,
   copier,
   AssemblyData::ProjectionSolverBase::VectorFieldConstantMatrices::Scratch<dim>(
     *this->mapping,
     quadrature_formula,
     vector_field->get_finite_element(),
     update_values|update_gradients|update_JxW_values),
   AssemblyData::ProjectionSolverBase::VectorFieldConstantMatrices::Copy(
     vector_field->get_finite_element().dofs_per_cell));

  // Compress global data
  diffusion_step_mass_matrix.compress(VectorOperation::add);
  diffusion_step_stiffness_matrix.compress(VectorOperation::add);
}



template <int dim>
void ProjectionSolverBase<dim>::assemble_local_matrices_vector_field(
 const typename DoFHandler<dim>::active_cell_iterator  &cell,
 AssemblyData::ProjectionSolverBase::VectorFieldConstantMatrices::Scratch<dim> &scratch,
 AssemblyData::ProjectionSolverBase::VectorFieldConstantMatrices::Copy &data)
{
  // Reset local data
  data.local_mass_matrix = 0.;
  data.local_stiffness_matrix = 0.;

  // Velocity's cell data
  scratch.fe_values.reinit(cell);

  const FEValuesExtractors::Vector  vector_extractor(0);

  // Local to global indices mapping
  cell->get_dof_indices(data.local_dof_indices);

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.phi[i]      = scratch.fe_values[vector_extractor].value(i, q);
      scratch.grad_phi[i] = scratch.fe_values[vector_extractor].gradient(i, q);
    }

    // Loop over local degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      // Compute values of the lower triangular part (Symmetry)
      for (unsigned int j = 0; j <= i; ++j)
      {
        // Local matrices
        data.local_mass_matrix(i, j) += scratch.phi[i] *
                                        scratch.phi[j] *
                                        scratch.fe_values.JxW(q);
        data.local_stiffness_matrix(i, j) +=  scalar_product(
                                                scratch.grad_phi[i],
                                                scratch.grad_phi[j]) *
                                              scratch.fe_values.JxW(q);
      } // Loop over local degrees of freedom
  } // Loop over quadrature points

  // Copy lower triangular part values into the upper triangular part
  for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    for (unsigned int j = i + 1; j < scratch.dofs_per_cell; ++j)
    {
      data.local_mass_matrix(i, j)      = data.local_mass_matrix(j, i);
      data.local_stiffness_matrix(i, j) = data.local_stiffness_matrix(j, i);
    }
}



template <int dim>
void ProjectionSolverBase<dim>::copy_local_to_global_matrices_vector_field(
 const AssemblyData::ProjectionSolverBase::VectorFieldConstantMatrices::Copy &data)
{
  vector_field->get_constraints().distribute_local_to_global(
                                      data.local_mass_matrix,
                                      data.local_dof_indices,
                                      diffusion_step_mass_matrix);
  vector_field->get_constraints().distribute_local_to_global(
                                      data.local_stiffness_matrix,
                                      data.local_dof_indices,
                                      diffusion_step_stiffness_matrix);
}

template <int dim>
void ProjectionSolverBase<dim>::assemble_constant_matrices_scalar_fields()
{
  // Reset data
  zeroth_step_system_matrix     = 0.;
  projection_step_system_matrix = 0.;

  // Initiate the quadrature formula for exact numerical integration
  const QGauss<dim>   quadrature_formula(lagrange_multiplier->fe_degree() + 1);

  // Set up the lambda function for the local assembly operation
  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
           AssemblyData::ProjectionSolverBase::ScalarFieldsConstantMatrices::Scratch<dim> &scratch,
           AssemblyData::ProjectionSolverBase::ScalarFieldsConstantMatrices::Copy &data)
    {
      this->assemble_local_matrices_scalar_fields(cell,
                                                  scratch,
                                                  data);
    };

  // Set up the lamba function for the copy local to global operation
  auto copier =
    [this](const AssemblyData::ProjectionSolverBase::ScalarFieldsConstantMatrices::Copy &data)
    {
      this->copy_local_to_global_matrices_scalar_fields(data);
    };

  // Assemble using the WorkStream approach
  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  WorkStream::run
  (CellFilter(IteratorFilters::LocallyOwnedCell(),
              lagrange_multiplier->get_dof_handler().begin_active()),
   CellFilter(IteratorFilters::LocallyOwnedCell(),
              lagrange_multiplier->get_dof_handler().end()),
   worker,
   copier,
   AssemblyData::ProjectionSolverBase::ScalarFieldsConstantMatrices::Scratch<dim>(
    *this->mapping,
    quadrature_formula,
    lagrange_multiplier->get_finite_element(),
    update_values|update_gradients|update_JxW_values),
   AssemblyData::ProjectionSolverBase::ScalarFieldsConstantMatrices::Copy(
    lagrange_multiplier->get_finite_element().dofs_per_cell));

  // Compress global data
  zeroth_step_system_matrix.compress(VectorOperation::add);
  projection_step_system_matrix.compress(VectorOperation::add);
}



template <int dim>
void ProjectionSolverBase<dim>::assemble_local_matrices_scalar_fields(
 const typename DoFHandler<dim>::active_cell_iterator  &cell,
 AssemblyData::ProjectionSolverBase::ScalarFieldsConstantMatrices::Scratch<dim> &scratch,
 AssemblyData::ProjectionSolverBase::ScalarFieldsConstantMatrices::Copy &data)
{
  // Reset local data
  data.local_mass_matrix      = 0.;
  data.local_stiffness_matrix = 0.;

  // Pressure's cell data
  scratch.fe_values.reinit(cell);

  // Local to global indices mapping
  cell->get_dof_indices(data.local_dof_indices);

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.phi[i]      = scratch.fe_values.shape_value(i, q);
      scratch.grad_phi[i] = scratch.fe_values.shape_grad(i, q);
    }

    // Loop over local degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      // Compute values of the lower triangular part (Symmetry)
      for (unsigned int j = 0; j <= i; ++j)
      {
        // Local matrices
        data.local_mass_matrix(i, j) += scratch.phi[i] *
                                        scratch.phi[j] *
                                        scratch.fe_values.JxW(q);
        data.local_stiffness_matrix(i, j) +=  scratch.grad_phi[i] *
                                              scratch.grad_phi[j] *
                                              scratch.fe_values.JxW(q);
      } // Loop over local degrees of freedom
  } // Loop over quadrature points

  // Copy lower triangular part values into the upper triangular part
  for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    for (unsigned int j = i + 1; j < scratch.dofs_per_cell; ++j)
    {
      data.local_mass_matrix(i, j)      = data.local_mass_matrix(j, i);
      data.local_stiffness_matrix(i, j) = data.local_stiffness_matrix(j, i);
    }
}



template <int dim>
void ProjectionSolverBase<dim>::copy_local_to_global_matrices_scalar_fields(
 const AssemblyData::ProjectionSolverBase::ScalarFieldsConstantMatrices::Copy &data)
{
  lagrange_multiplier->get_constraints().distribute_local_to_global(
                                      data.local_stiffness_matrix,
                                      data.local_dof_indices,
                                      zeroth_step_system_matrix);
  auxiliary_scalar->get_constraints().distribute_local_to_global(
                                      data.local_stiffness_matrix,
                                      data.local_dof_indices,
                                      projection_step_system_matrix);
}



} // namespace Solvers



} // namespace RMHD



template class RMHD::Solvers::SolverBase<2>;
template class RMHD::Solvers::SolverBase<3>;

template class RMHD::Solvers::ProjectionSolverBase<2>;
template class RMHD::Solvers::ProjectionSolverBase<3>;