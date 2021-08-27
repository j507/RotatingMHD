#include <rotatingMHD/convection_diffusion_solver.h>

#include <deal.II/base/work_stream.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/grid/filtered_iterator.h>
#include <deal.II/fe/fe_nothing.h>

namespace RMHD
{

using Copy = AssemblyData::HeatEquation::AdvectionMatrix::Copy;

template <int dim>
void ConvectionDiffusionSolver<dim>::assemble_advection_matrix()
{
  if (parameters.verbose)
    *pcout << "  Heat Equation: Assembling advection matrix...";

  // Reset data
  advection_matrix = 0.;

  // Dummy finite element for when the velocity is given by a function
  const FESystem<dim> dummy_fe_system(FE_Nothing<dim>(1), dim);

  // Create pointer to the pertinent finite element
  const FiniteElement<dim>* const velocity_fe =
              (velocity != nullptr) ? &velocity->get_finite_element() : &dummy_fe_system;

  // Initiate the quadrature formula for exact numerical integration
  const QGauss<dim>   quadrature_formula(temperature->fe_degree() + 1);

  // Set up the lambda function for the local assembly operation
  using Scratch = typename AssemblyData::HeatEquation::AdvectionMatrix::Scratch<dim>;
  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator     &cell,
           Scratch  &scratch,
           Copy     &data)
    {
      this->assemble_local_advection_matrix(cell,
                                             scratch,
                                             data);
    };

  // Set up the lambda function for the copy local to global operation
  auto copier =
    [this](const Copy  &data)
    {
      this->copy_local_to_global_advection_matrix(data);
    };

  // Assemble using the WorkStream approach
  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  const UpdateFlags advection_update_flags = update_values|
                                             update_gradients|
                                             update_JxW_values |
                                             update_quadrature_points;
  WorkStream::run
  (CellFilter(IteratorFilters::LocallyOwnedCell(),
              temperature->get_dof_handler().begin_active()),
   CellFilter(IteratorFilters::LocallyOwnedCell(),
              temperature->get_dof_handler().end()),
   worker,
   copier,
   Scratch(*mapping,
           quadrature_formula,
           temperature->get_finite_element(),
           advection_update_flags,
           *velocity_fe,
           update_values),
   Copy(temperature->get_finite_element().dofs_per_cell));

  // Compress global data
  advection_matrix.compress(VectorOperation::add);

  if (parameters.verbose)
    *pcout << " done!" << std::endl;
}

template <int dim>
void ConvectionDiffusionSolver<dim>::assemble_local_advection_matrix
(const typename DoFHandler<dim>::active_cell_iterator       &cell,
 AssemblyData::HeatEquation::AdvectionMatrix::Scratch<dim>  &scratch,
 Copy &data)
{
  // Reset local data
  data.local_matrix = 0.;

  // Temperature's cell data
  scratch.temperature_fe_values.reinit(cell);

  // Velocity's cell data
  if (velocity != nullptr)
  {
    typename DoFHandler<dim>::active_cell_iterator
    velocity_cell(&temperature->get_triangulation(),
                  cell->level(),
                  cell->index(),
                  //Pointer to the velocity's DoFHandler
                  &velocity->get_dof_handler());

    scratch.velocity_fe_values.reinit(velocity_cell);

    const FEValuesExtractors::Vector vector_extractor(0);

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
      scratch.phi[i]      = scratch.temperature_fe_values.shape_value(i, q);
      scratch.grad_phi[i] = scratch.temperature_fe_values.shape_grad(i, q);
    }

    // Loop over local degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
      for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
        // Local matrix
        data.local_matrix(i, j) +=
              (scratch.phi[i] * (
                (velocity != nullptr)
                 ? (eta[0] *
                    scratch.old_velocity_values[q]
                    +
                    eta[1] *
                    scratch.old_old_velocity_values[q])
                 : scratch.velocity_values[q]) *
               scratch.grad_phi[j]) *
              scratch.temperature_fe_values.JxW(q);
  } // Loop over quadrature points
} // assemble_local_advection_matrix

template <int dim>
void ConvectionDiffusionSolver<dim>::copy_local_to_global_advection_matrix
(const Copy &data)
{
  temperature->get_constraints().distribute_local_to_global(
                                      data.local_matrix,
                                      data.local_dof_indices,
                                      advection_matrix);
}

} // namespace RMHD

// explicit instantiations
template void RMHD::ConvectionDiffusionSolver<2>::assemble_advection_matrix();
template void RMHD::ConvectionDiffusionSolver<3>::assemble_advection_matrix();

template void RMHD::ConvectionDiffusionSolver<2>::assemble_local_advection_matrix
(const typename DoFHandler<2>::active_cell_iterator             &,
 RMHD::AssemblyData::HeatEquation::AdvectionMatrix::Scratch<2>  &,
 RMHD::AssemblyData::HeatEquation::AdvectionMatrix::Copy        &);
template void RMHD::ConvectionDiffusionSolver<3>::assemble_local_advection_matrix
(const typename DoFHandler<3>::active_cell_iterator             &,
 RMHD::AssemblyData::HeatEquation::AdvectionMatrix::Scratch<3>  &,
 RMHD::AssemblyData::HeatEquation::AdvectionMatrix::Copy        &);

template void RMHD::ConvectionDiffusionSolver<2>::copy_local_to_global_advection_matrix
(const RMHD::AssemblyData::HeatEquation::AdvectionMatrix::Copy  &);
template void RMHD::ConvectionDiffusionSolver<3>::copy_local_to_global_advection_matrix
(const RMHD::AssemblyData::HeatEquation::AdvectionMatrix::Copy  &);
