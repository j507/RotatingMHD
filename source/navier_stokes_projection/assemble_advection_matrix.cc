#include <rotatingMHD/navier_stokes_projection.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/grid/filtered_iterator.h>

namespace RMHD
{

using Copy = AssemblyData::NavierStokesProjection::AdvectionMatrix::Copy;

template <int dim>
void NavierStokesProjection<dim>::assemble_velocity_advection_matrix()
{
  if (parameters.verbose)
    *pcout << "  Navier Stokes: Assembling advection matrix...";

  TimerOutput::Scope  t(*computing_timer, "Navier Stokes: Advection matrix assembly");

  // Reset data
  velocity_advection_matrix = 0.;

  // Initiate the quadrature formula for exact numerical integration
  const QGauss<dim>   quadrature_formula(velocity->fe_degree() + 1);

  // Set up the lambda function for the local assembly operation
  using Scratch = typename AssemblyData::NavierStokesProjection::AdvectionMatrix::Scratch<dim>;
  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
           Scratch  &scratch,
           Copy     &data)
    {
      this->assemble_local_velocity_advection_matrix(cell,
                                                     scratch,
                                                     data);
    };

  // Set up the lambda function for the copy local to global operation
  auto copier =
    [this](const AssemblyData::NavierStokesProjection::AdvectionMatrix::Copy    &data)
    {
      this->copy_local_to_global_velocity_advection_matrix(data);
    };

  // Assemble using the WorkStream approach
  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  const UpdateFlags advection_update_flags = update_values|
                                             update_gradients|
                                             update_JxW_values;
  WorkStream::run
  (CellFilter(IteratorFilters::LocallyOwnedCell(),
              velocity->get_dof_handler().begin_active()),
   CellFilter(IteratorFilters::LocallyOwnedCell(),
              velocity->get_dof_handler().end()),
   worker,
   copier,
   Scratch(*mapping,
           quadrature_formula,
           velocity->get_finite_element(),
           advection_update_flags),
   Copy(velocity->get_finite_element().dofs_per_cell));

  // Compress global data
  velocity_advection_matrix.compress(VectorOperation::add);

  if (parameters.verbose)
    *pcout << " done!" << std::endl;
}



template <int dim>
void NavierStokesProjection<dim>::assemble_local_velocity_advection_matrix
(const typename DoFHandler<dim>::active_cell_iterator  &cell,
 AssemblyData::NavierStokesProjection::AdvectionMatrix::Scratch<dim>&scratch,
 Copy        &data)
{
  // Reset local data
  data.local_matrix = 0.;

  // Velocity's cell data
  scratch.fe_values.reinit(cell);

  const FEValuesExtractors::Vector  vector_extractor(0);

  scratch.fe_values[vector_extractor].get_function_values(
    velocity->old_solution,
    scratch.old_velocity_values);

  scratch.fe_values[vector_extractor].get_function_values(
    velocity->old_old_solution,
    scratch.old_old_velocity_values);

  if ((parameters.convective_term_weak_form ==
        RunTimeParameters::ConvectiveTermWeakForm::divergence) ||
      (parameters.convective_term_weak_form ==
              RunTimeParameters::ConvectiveTermWeakForm::skewsymmetric))
  {
    scratch.fe_values[vector_extractor].get_function_divergences(
      velocity->old_solution,
      scratch.old_velocity_divergences);

    scratch.fe_values[vector_extractor].get_function_divergences(
      velocity->old_old_solution,
      scratch.old_old_velocity_divergences);
  }

  if (parameters.convective_term_weak_form ==
        RunTimeParameters::ConvectiveTermWeakForm::rotational)
  {
    scratch.fe_values[vector_extractor].get_function_curls(
      velocity->old_solution,
      scratch.old_velocity_curls);
    scratch.fe_values[vector_extractor].get_function_curls(
      velocity->old_old_solution,
      scratch.old_old_velocity_curls);
  }

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
      scratch.phi[i]        = scratch.fe_values[vector_extractor].value(i, q);
      scratch.grad_phi[i]   = scratch.fe_values[vector_extractor].gradient(i, q);
      /*! @note As above, should I leave this if in? */
      if (parameters.convective_term_weak_form ==
        RunTimeParameters::ConvectiveTermWeakForm::rotational)
        scratch.curl_phi[i] = scratch.fe_values[vector_extractor].curl(i,q);
    }

    const Tensor<1,dim> extrapolated_velocity_value =
        eta[0] * scratch.old_velocity_values[q] +
        eta[1] * scratch.old_old_velocity_values[q];

    switch (parameters.convective_term_weak_form)
    {
      case RunTimeParameters::ConvectiveTermWeakForm::standard:
      {
        // Loop over local degrees of freedom
        for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
          for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
            data.local_matrix(i, j) +=
                          scratch.phi[i] *
                          scratch.grad_phi[j] *
                          extrapolated_velocity_value *
                          scratch.fe_values.JxW(q);
        break;
      }
      case RunTimeParameters::ConvectiveTermWeakForm::skewsymmetric:
      {
        const double extrapolated_velocity_divergence =
            eta[0] * scratch.old_velocity_divergences[q] +
            eta[1] * scratch.old_old_velocity_divergences[q];

        // Loop over local degrees of freedom
        for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
          for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
            data.local_matrix(i, j) += (
                          scratch.phi[i] *
                          scratch.grad_phi[j] *
                          extrapolated_velocity_value
                          +
                          0.5 * extrapolated_velocity_divergence *
                          scratch.phi[i] *
                          scratch.phi[j]) *
                          scratch.fe_values.JxW(q);
        break;
      }
      case RunTimeParameters::ConvectiveTermWeakForm::divergence:
      {
        const double extrapolated_velocity_divergence =
            eta[0] * scratch.old_velocity_divergences[q] +
            eta[1] * scratch.old_old_velocity_divergences[q];

        // Loop over local degrees of freedom
        for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
          for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
            data.local_matrix(i, j) += (
                          scratch.phi[i] *
                          scratch.grad_phi[j] *
                          extrapolated_velocity_value
                          +
                          extrapolated_velocity_divergence *
                          scratch.phi[i] *
                          scratch.phi[j]) *
                          scratch.fe_values.JxW(q);
        break;
      }
      case RunTimeParameters::ConvectiveTermWeakForm::rotational:
      {
        // The minus sign in the argument of cross_product_2d
        // method is due to how the method is defined.
        if constexpr(dim == 2)
          // Loop over local degrees of freedom
          for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
            for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
              data.local_matrix(i, j) +=
                          scratch.phi[i] *
                          scratch.curl_phi[j][0] *
                          cross_product_2d(-extrapolated_velocity_value) *
                          scratch.fe_values.JxW(q);
        else if constexpr(dim == 3)
          // Loop over local degrees of freedom
          for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
            for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
              data.local_matrix(i, j) +=
                          scratch.phi[i] *
                          cross_product_3d(scratch.curl_phi[j],
                                           extrapolated_velocity_value) *
                          scratch.fe_values.JxW(q);
        break;
      }
      default:
        Assert(false, ExcNotImplemented());
    }; // End of switch
  } // Loop over quadrature points

}



template <int dim>
void NavierStokesProjection<dim>::copy_local_to_global_velocity_advection_matrix
(const Copy &data)
{
  velocity->get_constraints().distribute_local_to_global(
                                      data.local_matrix,
                                      data.local_dof_indices,
                                      velocity_advection_matrix);
}

} // namespace RMHD

// explicit instantiations
template void RMHD::NavierStokesProjection<2>::assemble_velocity_advection_matrix();
template void RMHD::NavierStokesProjection<3>::assemble_velocity_advection_matrix();

template void RMHD::NavierStokesProjection<2>::assemble_local_velocity_advection_matrix
(const typename DoFHandler<2>::active_cell_iterator                       &,
 RMHD::AssemblyData::NavierStokesProjection::AdvectionMatrix::Scratch<2>  &,
 RMHD::AssemblyData::NavierStokesProjection::AdvectionMatrix::Copy        &);
template void RMHD::NavierStokesProjection<3>::assemble_local_velocity_advection_matrix
(const typename DoFHandler<3>::active_cell_iterator                       &,
 RMHD::AssemblyData::NavierStokesProjection::AdvectionMatrix::Scratch<3>  &,
 RMHD::AssemblyData::NavierStokesProjection::AdvectionMatrix::Copy        &);

template void RMHD::NavierStokesProjection<2>::copy_local_to_global_velocity_advection_matrix
(const RMHD::AssemblyData::NavierStokesProjection::AdvectionMatrix::Copy &);
template void RMHD::NavierStokesProjection<3>::copy_local_to_global_velocity_advection_matrix
(const RMHD::AssemblyData::NavierStokesProjection::AdvectionMatrix::Copy &);
