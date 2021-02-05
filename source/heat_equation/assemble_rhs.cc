#include <rotatingMHD/heat_equation.h>

#include <deal.II/base/work_stream.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/fe/fe_nothing.h>

namespace RMHD
{

template <int dim>
void HeatEquation<dim>::assemble_rhs()
{
  if (parameters.verbose)
    *pcout << "  Heat Equation: Assembling right hand side...";

  TimerOutput::Scope  t(*computing_timer,
                        "Heat equation: RHS assembly");

  // Reset data
  rhs = 0.;

  // Dummy finite element for when the velocity is given by a function
  const FESystem<dim> dummy_fe_system(FE_Nothing<dim>(2), dim);

  // Create pointer to the pertinent finite element
  const FESystem<dim>* const velocity_fe =
              (velocity != nullptr) ? &velocity->fe : &dummy_fe_system;

  // Set polynomial degree of the velocity. If the velicity is given
  // by a function the degree is hardcoded to 2.
  const unsigned int velocity_fe_degree =
                        (velocity != nullptr) ? velocity->fe_degree : 2;

  // Set polynomial degree of the source function.
  // Hardcoded to match that of the velocity.
  const int p_degree_source_function = temperature->fe_degree;

  // Set polynomial degree of the Neumann boundary condition function.
  // Hardcoded to match that of the velocity.
  const int p_degree_neumann_function = temperature->fe_degree;

  // Compute the highest polynomial degree from all the integrands
  const int p_degree = std::max(temperature->fe_degree + p_degree_source_function,
                                2 * temperature->fe_degree + velocity_fe_degree - 1);

  // Initiate the quadrature formula for exact numerical integration
  const QGauss<dim>   quadrature_formula(std::ceil(0.5 * double(p_degree + 1)));

  // Compute the highest polynomial degree from all the boundary integrands
  const int face_p_degree = temperature->fe_degree + p_degree_neumann_function;

  // Initiate the quadrature formula for exact numerical integration
  const QGauss<dim-1>   face_quadrature_formula(std::ceil(0.5 * double(face_p_degree + 1)));

  // Set up the lamba function for the local assembly operation
  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator     &cell,
           AssemblyData::HeatEquation::RightHandSide::Scratch<dim>  &scratch,
           AssemblyData::HeatEquation::RightHandSide::Copy          &data)
    {
      this->assemble_local_rhs(cell,
                               scratch,
                               data);
    };

  // Set up the lamba function for the copy local to global operation
  auto copier =
    [this](const AssemblyData::HeatEquation::RightHandSide::Copy    &data)
    {
      this->copy_local_to_global_rhs(data);
    };

  // Assemble using the WorkStream approach
  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  WorkStream::run
  (CellFilter(IteratorFilters::LocallyOwnedCell(),
              (temperature->dof_handler)->begin_active()),
   CellFilter(IteratorFilters::LocallyOwnedCell(),
              (temperature->dof_handler)->end()),
   worker,
   copier,
   AssemblyData::HeatEquation::RightHandSide::Scratch<dim>(
    *mapping,
    quadrature_formula,
    face_quadrature_formula,
    temperature->fe,
    update_JxW_values |
    update_values|
    update_gradients|
    update_quadrature_points,
    update_JxW_values |
    update_values |
    update_quadrature_points,
    *velocity_fe,
    update_values),
   AssemblyData::HeatEquation::RightHandSide::Copy(temperature->fe.dofs_per_cell));

  // Compress global data
  rhs.compress(VectorOperation::add);

  // Compute the L2 norm of the right hand side
  rhs_norm = rhs.l2_norm();

  if (parameters.verbose)
    *pcout << " done!" << std::endl
           << "    Right-hand side's L2-norm = "
           << rhs_norm
           << std::endl;
}

template <int dim>
void HeatEquation<dim>::assemble_local_rhs(
  const typename DoFHandler<dim>::active_cell_iterator      &cell,
  AssemblyData::HeatEquation::RightHandSide::Scratch<dim>   &scratch,
  AssemblyData::HeatEquation::RightHandSide::Copy           &data)
{
  // Reset local data
  data.local_rhs                          = 0.;
  data.local_matrix_for_inhomogeneous_bc  = 0.;

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

  // Velocity
  if (velocity != nullptr)
  {
    typename DoFHandler<dim>::active_cell_iterator
      velocity_cell(&temperature->get_triangulation(),
                    cell->level(),
                    cell->index(),
                    // Pointer to the velocity's DoFHandler
                    velocity->dof_handler.get());

    scratch.velocity_fe_values.reinit(velocity_cell);

    const FEValuesExtractors::Vector  vector_extractor(0);

    scratch.velocity_fe_values[vector_extractor].get_function_values(
      velocity->old_solution,
      scratch.old_velocity_values);

    scratch.velocity_fe_values[vector_extractor].get_function_values(
      velocity->old_old_solution,
      scratch.old_old_velocity_values);
  }
  else if (velocity_function_ptr != nullptr)
    velocity_function_ptr->value_list(
      scratch.temperature_fe_values.get_quadrature_points(),
      scratch.velocity_values);
  else
    ZeroTensorFunction<1,dim>().value_list(
      scratch.temperature_fe_values.get_quadrature_points(),
      scratch.velocity_values);

  // Source term
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
  }
  else
  {
    ZeroFunction<dim>().value_list(
      scratch.temperature_fe_values.get_quadrature_points(),
      scratch.source_term_values);

    scratch.old_source_term_values      = scratch.source_term_values;
    scratch.old_old_source_term_values  = scratch.source_term_values;
  }

  // VSIMEX coefficients
  const std::vector<double> alpha = time_stepping.get_alpha();
  const std::vector<double> beta  = time_stepping.get_beta();
  const std::vector<double> gamma = time_stepping.get_gamma();

  // Taylor extrapolation coefficients
  const std::vector<double> eta   = time_stepping.get_eta();

  // Local to global indices mapping
  cell->get_dof_indices(data.local_dof_indices);

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
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
          (alpha[1] /
           time_stepping.get_next_step_size() *
           scratch.phi[i] *
           scratch.old_temperature_values[q]
           +
           alpha[2] /
           time_stepping.get_next_step_size() *
           scratch.phi[i] *
           scratch.old_old_temperature_values[q]
           -
           gamma[0] *
           scratch.phi[i] *
           scratch.source_term_values[q]
           +
           gamma[1] * (
             parameters.C4 *
             scratch.grad_phi[i] *
             scratch.old_temperature_gradients[q]
             -
             scratch.phi[i] *
             scratch.old_source_term_values[q])
           +
           gamma[2] * (
             parameters.C4 *
             scratch.grad_phi[i] *
             scratch.old_old_temperature_gradients[q]
             -
             scratch.phi[i] *
             scratch.old_old_source_term_values[q])) *
          scratch.temperature_fe_values.JxW(q);
      if (parameters.convective_term_time_discretization ==
            RunTimeParameters::ConvectiveTermTimeDiscretization::fully_explicit &&
          !flag_ignore_advection)
        data.local_rhs(i) -=
          (beta[0] *
           scratch.phi[i] *
           ((velocity != nullptr)
            ? scratch.old_velocity_values[q]
            : scratch.velocity_values[q]) *
           scratch.old_temperature_gradients[q]
           +
           beta[1] *
           scratch.phi[i] *
           ((velocity != nullptr)
            ? scratch.old_old_velocity_values[q]
            : scratch.velocity_values[q]) *
           scratch.old_old_temperature_gradients[q]) *
          scratch.temperature_fe_values.JxW(q);

      // Loop over the i-th column's rows of the local matrix
      // for the case of inhomogeneous Dirichlet boundary conditions
      if (temperature->constraints.is_inhomogeneously_constrained(
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
               parameters.C4 *
               scratch.grad_phi[j] *
               scratch.grad_phi[i]) *
              scratch.temperature_fe_values.JxW(q);

          if (parameters.convective_term_time_discretization ==
                RunTimeParameters::ConvectiveTermTimeDiscretization::semi_implicit &&
              !flag_ignore_advection)
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
  if (cell->at_boundary())
    for (const auto &face : cell->face_iterators())
      if (face->at_boundary() &&
          temperature->boundary_conditions.neumann_bcs.find(face->boundary_id())
          != temperature->boundary_conditions.neumann_bcs.end())
      {
        // Neumann boundary condition
        scratch.temperature_fe_face_values.reinit(cell, face);

        temperature->boundary_conditions.neumann_bcs[face->boundary_id()]->set_time(
          time_stepping.get_current_time() - time_stepping.get_previous_step_size());
        temperature->boundary_conditions.neumann_bcs[face->boundary_id()]->value_list(
          scratch.temperature_fe_face_values.get_quadrature_points(),
          scratch.old_old_neumann_bc_values);

        temperature->boundary_conditions.neumann_bcs[face->boundary_id()]->set_time(
          time_stepping.get_current_time() + time_stepping.get_next_step_size());
        temperature->boundary_conditions.neumann_bcs[face->boundary_id()]->value_list(
          scratch.temperature_fe_face_values.get_quadrature_points(),
          scratch.neumann_bc_values);

        temperature->boundary_conditions.neumann_bcs[face->boundary_id()]->set_time(
          time_stepping.get_current_time());
        temperature->boundary_conditions.neumann_bcs[face->boundary_id()]->value_list(
          scratch.temperature_fe_face_values.get_quadrature_points(),
          scratch.old_neumann_bc_values);

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
void HeatEquation<dim>::copy_local_to_global_rhs
(const AssemblyData::HeatEquation::RightHandSide::Copy  &data)
{
  temperature->constraints.distribute_local_to_global(
    data.local_rhs,
    data.local_dof_indices,
    rhs,
    data.local_matrix_for_inhomogeneous_bc);
}

} // namespace RMHD

// explicit instantiations
template void RMHD::HeatEquation<2>::assemble_rhs();
template void RMHD::HeatEquation<3>::assemble_rhs();

template void RMHD::HeatEquation<2>::assemble_local_rhs
(const typename DoFHandler<2>::active_cell_iterator         &,
 RMHD::AssemblyData::HeatEquation::RightHandSide::Scratch<2>&,
 RMHD::AssemblyData::HeatEquation::RightHandSide::Copy      &);

template void RMHD::HeatEquation<3>::assemble_local_rhs
(const typename DoFHandler<3>::active_cell_iterator         &,
 RMHD::AssemblyData::HeatEquation::RightHandSide::Scratch<3>&,
 RMHD::AssemblyData::HeatEquation::RightHandSide::Copy      &);

template void RMHD::HeatEquation<2>::copy_local_to_global_rhs
(const RMHD::AssemblyData::HeatEquation::RightHandSide::Copy &);
template void RMHD::HeatEquation<3>::copy_local_to_global_rhs
(const RMHD::AssemblyData::HeatEquation::RightHandSide::Copy &);
