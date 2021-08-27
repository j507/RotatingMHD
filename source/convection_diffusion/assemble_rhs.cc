#include <rotatingMHD/convection_diffusion_solver.h>

#include <deal.II/base/work_stream.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/fe/fe_nothing.h>

namespace RMHD
{

using Copy = AssemblyData::HeatEquation::RightHandSide::Copy;

template <int dim>
void ConvectionDiffusionSolver<dim>::assemble_rhs()
{
  if (parameters.verbose)
    *pcout << "  Heat Equation: Assembling right hand side...";

  TimerOutput::Scope  t(*computing_timer,
                        "Convection diffusion: RHS assembly");

  // Reset data
  rhs = 0.;

  // Dummy finite element for when the velocity is given by a function
  const FESystem<dim> dummy_fe_system(FE_Nothing<dim>(2), dim);

  // Create pointer to the pertinent finite element
  const FiniteElement<dim>* const velocity_fe =
              (velocity != nullptr) ? &velocity->get_finite_element() : &dummy_fe_system;

  // Initiate the quadrature formula for exact numerical integration
  const QGauss<dim>   quadrature_formula(temperature->fe_degree() + 1);

  // Initiate the quadrature formula for exact numerical integration
  const QGauss<dim-1>   face_quadrature_formula(temperature->fe_degree() + 1);

  // Set up the lambda function for the local assembly operation
  using Scratch = typename AssemblyData::HeatEquation::RightHandSide::Scratch<dim>;
  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator     &cell,
           Scratch  &scratch,
           Copy     &data)
    {
      this->assemble_local_rhs(cell,
                               scratch,
                               data);
    };

  // Set up the lambda function for the copy local to global operation
  auto copier =
    [this](const Copy &data)
    {
      this->copy_local_to_global_rhs(data);
    };

  // Assemble using the WorkStream approach
  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  const UpdateFlags temperature_update_flags = update_values|
                                               update_gradients|
                                               update_quadrature_points|
                                               update_JxW_values;
  const UpdateFlags temperature_face_update_flags = update_values|
                                                    update_quadrature_points|
                                                    update_JxW_values;
  WorkStream::run
  (CellFilter(IteratorFilters::LocallyOwnedCell(),
              temperature->get_dof_handler().begin_active()),
   CellFilter(IteratorFilters::LocallyOwnedCell(),
              temperature->get_dof_handler().end()),
   worker,
   copier,
   Scratch(*mapping,
           quadrature_formula,
           face_quadrature_formula,
           temperature->get_finite_element(),
           temperature_update_flags,
           temperature_face_update_flags,
           *velocity_fe,
           update_values),
   Copy(temperature->get_finite_element().dofs_per_cell));

  // Compress global data
  rhs.compress(VectorOperation::add);

  // Compute the L2 norm of the right hand side
  rhs_norm = rhs.l2_norm();

  if (parameters.verbose)
    *pcout << " done!" << std::endl
           << "    Right-hand side's L2-norm = "
           << std::scientific << std::setprecision(6)
           << rhs_norm
           << std::endl;
}

template <int dim>
void ConvectionDiffusionSolver<dim>::assemble_local_rhs
(const typename DoFHandler<dim>::active_cell_iterator     &cell,
 AssemblyData::HeatEquation::RightHandSide::Scratch<dim>  &scratch,
 Copy &data)
{
  const typename Entities::ScalarBoundaryConditions<dim>::NeumannBCMapping
  &neumann_bcs = temperature->get_neumann_boundary_conditions();

  // Reset local data
  data.local_rhs                          = 0.;
  data.local_matrix_for_inhomogeneous_bc  = 0.;

  // VSIMEX coefficients
  const std::vector<double> alpha = time_stepping.get_alpha();
  const std::vector<double> beta  = time_stepping.get_beta();
  const std::vector<double> gamma = time_stepping.get_gamma();

  // Taylor extrapolation coefficients
  const std::vector<double> eta   = time_stepping.get_eta();

  const bool explicit_advection =
    (parameters.time_discretization ==
     RunTimeParameters::ConvectiveTermTimeDiscretization::fully_explicit) &&
     (velocity != nullptr || velocity_function_ptr != nullptr);

  // Local to global indices mapping
  cell->get_dof_indices(data.local_dof_indices);

  // Initialize weak forms of the right-hand side's terms
  std::vector<double>         explicit_temperature_term(scratch.n_q_points);
  std::vector<Tensor<1,dim>>  diffusion_term(scratch.n_q_points);
  std::vector<double>         heat_source_term(scratch.n_q_points);
  std::vector<double>         advection_term(scratch.n_q_points);

  // Temperature
  scratch.temperature_fe_values.reinit(cell);

  scratch.temperature_fe_values.get_function_values(
    temperature->old_solution,
    scratch.old_temperature_values);

  scratch.temperature_fe_values.get_function_values(
    temperature->old_old_solution,
    scratch.old_old_temperature_values);

  scratch.temperature_fe_values.get_function_gradients(
    temperature->old_solution,
    scratch.old_temperature_gradients);

  scratch.temperature_fe_values.get_function_gradients(
    temperature->old_old_solution,
    scratch.old_old_temperature_gradients);

  // Heat source
  if (source_term_ptr != nullptr)
  {
    source_term_ptr->set_time(time_stepping.get_previous_time());
    source_term_ptr->value_list(
      scratch.temperature_fe_values.get_quadrature_points(),
      scratch.old_old_source_term_values);

    source_term_ptr->set_time(time_stepping.get_current_time());
    source_term_ptr->value_list(
      scratch.temperature_fe_values.get_quadrature_points(),
      scratch.old_source_term_values);

    source_term_ptr->set_time(time_stepping.get_next_time());
    source_term_ptr->value_list(
      scratch.temperature_fe_values.get_quadrature_points(),
      scratch.source_term_values);

    // Loop over quadrature points
    for (unsigned int q = 0; q < scratch.n_q_points; ++q)
      heat_source_term[q] =
        1.0 * /* parameters.C7 */
        (gamma[0] *
         scratch.source_term_values[q]
         +
         gamma[1] *
         scratch.old_source_term_values[q]
         +
         gamma[2] *
         scratch.old_old_source_term_values[q]);
  }

  // Advection
  if (velocity != nullptr)
  {
    typename DoFHandler<dim>::active_cell_iterator
      velocity_cell(&temperature->get_triangulation(),
                    cell->level(),
                    cell->index(),
                    // Pointer to the velocity's DoFHandler
                    &velocity->get_dof_handler());

    scratch.velocity_fe_values.reinit(velocity_cell);

    const FEValuesExtractors::Vector  vector_extractor(0);

    scratch.velocity_fe_values[vector_extractor].get_function_values(
      velocity->old_solution,
      scratch.old_velocity_values);

    scratch.velocity_fe_values[vector_extractor].get_function_values(
      velocity->old_old_solution,
      scratch.old_old_velocity_values);

    if (explicit_advection)
      for (unsigned int q = 0; q < scratch.n_q_points; ++q)
        advection_term[q] =
          beta[0] *
          scratch.old_velocity_values[q] *
          scratch.old_temperature_gradients[q]
          +
          beta[1] *
          scratch.old_old_velocity_values[q] *
          scratch.old_old_temperature_gradients[q];
  }
  else if (velocity_function_ptr != nullptr)
  {
    velocity_function_ptr->value_list(
      scratch.temperature_fe_values.get_quadrature_points(),
      scratch.velocity_values);

    if (explicit_advection)
      for (unsigned int q = 0; q < scratch.n_q_points; ++q)
        advection_term[q] =
          beta[0] *
          scratch.velocity_values[q] *
          scratch.old_temperature_gradients[q]
          +
          beta[1] *
          scratch.velocity_values[q] *
          scratch.old_old_temperature_gradients[q];
  }

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Evaluate the weak form of the right-hand side's terms at
    // the quadrature point
    explicit_temperature_term[q] =
              alpha[1] / time_stepping.get_next_step_size() *
              scratch.old_temperature_values[q]
              +
              alpha[2] / time_stepping.get_next_step_size() *
              scratch.old_old_temperature_values[q];

    diffusion_term[q] =
              parameters.equation_coefficient *
              (gamma[1] *
               scratch.old_temperature_gradients[q]
               +
               gamma[2] *
               scratch.old_old_temperature_gradients[q]);

    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.phi[i]      = scratch.temperature_fe_values.shape_value(i,q);
      scratch.grad_phi[i] = scratch.temperature_fe_values.shape_grad(i,q);
    }

    // Loop over local degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      // Local right hand side (Domain integrals)
      data.local_rhs(i) -=
          (scratch.grad_phi[i] *
           diffusion_term[q]
           -
           scratch.phi[i] *
           (heat_source_term[q]
            -
            explicit_temperature_term[q]
            -
            advection_term[q])) *
          scratch.temperature_fe_values.JxW(q);

      // Loop over the i-th column's rows of the local matrix
      // for the case of inhomogeneous Dirichlet boundary conditions
      if (temperature->get_constraints().is_inhomogeneously_constrained(
            data.local_dof_indices[i]))
        for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
        {
          data.local_matrix_for_inhomogeneous_bc(j,i) +=
              (alpha[0] /
               time_stepping.get_next_step_size() *
               scratch.phi[j] *
               scratch.phi[i]
               +
               gamma[0] *
               parameters.equation_coefficient *
               scratch.grad_phi[j] *
               scratch.grad_phi[i]) *
              scratch.temperature_fe_values.JxW(q);

          if ((parameters.time_discretization ==
                RunTimeParameters::ConvectiveTermTimeDiscretization::semi_implicit &&
              (velocity != nullptr || velocity_function_ptr != nullptr)))
            data.local_matrix_for_inhomogeneous_bc(j,i) +=
              (scratch.phi[j] * (
                 (velocity != nullptr)
                 ? (eta[0] *
                    scratch.old_velocity_values[q]
                    +
                    eta[1] *
                    scratch.old_old_velocity_values[q])
                 : scratch.velocity_values[q]) *
               scratch.grad_phi[i]) *
              scratch.temperature_fe_values.JxW(q);
        } // Loop over the i-th column's rows of the local matrix
    } // Loop over local degrees of freedom
  } // Loop over quadrature points

  // Loop over the faces of the cell
  if (!neumann_bcs.empty())
    if (cell->at_boundary())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() &&
            neumann_bcs.find(face->boundary_id()) != neumann_bcs.end())
        {
          // Neumann boundary condition
          scratch.temperature_fe_face_values.reinit(cell, face);

          const types::boundary_id  boundary_id{face->boundary_id()};
          const std::vector<Point<dim>> face_quadrature_points{scratch.temperature_fe_face_values.get_quadrature_points()};

          neumann_bcs.at(boundary_id)->set_time(time_stepping.get_previous_time());
          neumann_bcs.at(boundary_id)->value_list(face_quadrature_points,
                                                  scratch.old_old_neumann_bc_values);

          neumann_bcs.at(boundary_id)->set_time(time_stepping.get_current_time());
          neumann_bcs.at(boundary_id)->value_list(face_quadrature_points,
                                                  scratch.old_neumann_bc_values);

          neumann_bcs.at(boundary_id)->set_time(time_stepping.get_next_time());
          neumann_bcs.at(boundary_id)->value_list(face_quadrature_points,
                                                  scratch.neumann_bc_values);

          // Loop over face quadrature points
          for (unsigned int q = 0; q < scratch.n_face_q_points; ++q)
          {
            // Extract the test function's values at the face quadrature points
            for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
              scratch.face_phi[i] =
                scratch.temperature_fe_face_values.shape_value(i,q);

            // Loop over the degrees of freedom
            for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
              data.local_rhs(i) +=
                scratch.face_phi[i] * (
                  gamma[0] *
                  scratch.neumann_bc_values[q]
                  +
                  gamma[1] *
                  scratch.old_neumann_bc_values[q]
                  +
                  gamma[2] *
                  scratch.old_old_neumann_bc_values[q]) *
                scratch.temperature_fe_face_values.JxW(q);
          } // Loop over face quadrature points
        } // Loop over the faces of the cell
} // assemble_local_rhs

template <int dim>
void ConvectionDiffusionSolver<dim>::copy_local_to_global_rhs
(const Copy  &data)
{
  temperature->get_constraints().distribute_local_to_global(
    data.local_rhs,
    data.local_dof_indices,
    rhs,
    data.local_matrix_for_inhomogeneous_bc);
}

} // namespace RMHD

// explicit instantiations
template void RMHD::ConvectionDiffusionSolver<2>::assemble_rhs();
template void RMHD::ConvectionDiffusionSolver<3>::assemble_rhs();

template void RMHD::ConvectionDiffusionSolver<2>::assemble_local_rhs
(const typename DoFHandler<2>::active_cell_iterator         &,
 RMHD::AssemblyData::HeatEquation::RightHandSide::Scratch<2>&,
 RMHD::AssemblyData::HeatEquation::RightHandSide::Copy      &);

template void RMHD::ConvectionDiffusionSolver<3>::assemble_local_rhs
(const typename DoFHandler<3>::active_cell_iterator         &,
 RMHD::AssemblyData::HeatEquation::RightHandSide::Scratch<3>&,
 RMHD::AssemblyData::HeatEquation::RightHandSide::Copy      &);

template void RMHD::ConvectionDiffusionSolver<2>::copy_local_to_global_rhs
(const RMHD::AssemblyData::HeatEquation::RightHandSide::Copy &);
template void RMHD::ConvectionDiffusionSolver<3>::copy_local_to_global_rhs
(const RMHD::AssemblyData::HeatEquation::RightHandSide::Copy &);
