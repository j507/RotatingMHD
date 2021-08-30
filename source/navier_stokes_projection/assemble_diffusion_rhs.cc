#include <rotatingMHD/navier_stokes_projection.h>

#include <deal.II/base/work_stream.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/grid/filtered_iterator.h>

namespace RMHD
{

namespace
{


template <int dim>
void compute_body_force
(TensorFunction<1, dim>* const  ptr,
 const std::vector<Point<dim>> &quadrature_points,
 const std::vector<double>     &gamma,
 const double                   previous_time,
 const double                   current_time,
 const double                   next_time,
 std::vector<Tensor<1,dim>>    &body_force)
{
  AssertDimension(body_force.size(), quadrature_points.size());

  std::vector<Tensor<1,dim>>  old_old_body_force_values(quadrature_points.size());
  ptr->set_time(previous_time);
  ptr->value_list(quadrature_points,
                  old_old_body_force_values);

  std::vector<Tensor<1,dim>>  old_body_force_values(quadrature_points.size());
  ptr->set_time(current_time);
  ptr->value_list(quadrature_points,
                  old_body_force_values);

  std::vector<Tensor<1,dim>>  body_force_values(quadrature_points.size());
  ptr->set_time(next_time);
  ptr->value_list(quadrature_points,
                  body_force_values);

  // Loop over quadrature points
  for (std::size_t q=0; q<quadrature_points.size(); ++q)
    body_force[q] =
      (gamma[0] * body_force_values[q] +
       gamma[1] * old_body_force_values[q] +
       gamma[2] * old_old_body_force_values[q]);
}



template <int dim>
void compute_buoyancy_term
(const TensorFunction<1, dim>* const  ptr,
 const Entities::FE_ScalarField<dim> &temperature,
 const typename DoFHandler<dim>::active_cell_iterator &cell,
 const std::vector<Point<dim>>       &quadrature_points,
 const std::vector<double>           &eta,
 const double                         coefficient,
 FEValues<dim>                       &fe_values,
 std::vector<Tensor<1,dim>>          &buoyancy_term)
{
  AssertDimension(buoyancy_term.size(), quadrature_points.size());

  Assert(ptr != nullptr,
         ExcMessage("Gravity field has not been specified."));

  typename DoFHandler<dim>::active_cell_iterator
  temperature_cell(&temperature.get_triangulation(),
                   cell->level(),
                   cell->index(),
                   &temperature.get_dof_handler());

  fe_values.reinit(temperature_cell);

  std::vector<double>  old_temperature_values(quadrature_points.size());
  fe_values.get_function_values(temperature.old_solution,
                                old_temperature_values);

  std::vector<double>  old_old_temperature_values(quadrature_points.size());
  fe_values.get_function_values(temperature.old_old_solution,
                                old_old_temperature_values);

  std::vector<Tensor<1,dim>>  gravity_vector_values(quadrature_points.size());
  ptr->value_list(quadrature_points,
                  gravity_vector_values);

  // Loop over quadrature points
  for (std::size_t q=0; q<quadrature_points.size(); ++q)
    buoyancy_term[q] = coefficient * gravity_vector_values[q] *
      (eta[0] * old_temperature_values[q] +
       eta[1] * old_old_temperature_values[q]);
}



template <int dim>
void compute_coriolis_acceleration_term
(AngularVelocity<dim>* const    ptr,
 const std::vector<Tensor<1,dim>> &old_velocity_values,
 const std::vector<Tensor<1,dim>> &old_old_velocity_values,
 const std::vector<double>     &eta,
 const double                   previous_time,
 const double                   current_time,
 const double                   coefficient,
 std::vector<Tensor<1,dim>>    &coriolis_acceleration)
{
  AssertDimension(coriolis_acceleration.size(), old_velocity_values.size());
  AssertDimension(coriolis_acceleration.size(), old_old_velocity_values.size());

  ptr->set_time(previous_time);
  const typename AngularVelocity<dim>::value_type
  old_old_angular_velocity = ptr->value();

  ptr->set_time(current_time);
  const typename AngularVelocity<dim>::value_type
  old_angular_velocity = ptr->value();

  if constexpr(dim == 2)
    // Loop over quadrature points
    for (std::size_t q=0; q<coriolis_acceleration.size(); ++q)
      coriolis_acceleration[q] =
        coefficient *
        (eta[0] *
         old_angular_velocity[0] * cross_product_2d(-old_velocity_values[q])
         +
         eta[1] *
         old_old_angular_velocity[0] * cross_product_2d(-old_old_velocity_values[q]));
  else if constexpr(dim == 3)
    // Loop over quadrature points
    for (std::size_t q=0; q<coriolis_acceleration.size(); ++q)
      coriolis_acceleration[q] =
        coefficient *
        (eta[0] *
         cross_product_3d(old_angular_velocity, old_velocity_values[q])
         +
         eta[1] *
         cross_product_3d(old_old_angular_velocity, old_old_velocity_values[q]));
}


template <int dim>
void compute_advection_term
(const RunTimeParameters::ConvectiveTermWeakForm  weak_form,
 const std::vector<Tensor<1,dim>>   &old_values,
 const std::vector<Tensor<1,dim>>   &old_old_values,
 const std::vector<Tensor<2,dim>>   &old_gradients,
 const std::vector<Tensor<2,dim>>   &old_old_gradients,
 const std::vector<double>          &beta,
 std::vector<Tensor<1,dim>>         &advection_term)
{

  AssertDimension(old_values.size(), old_old_values.size());
  AssertDimension(old_values.size(), old_old_values.size());
  AssertDimension(old_values.size(), old_gradients.size());
  AssertDimension(old_values.size(), old_old_gradients.size());
  AssertDimension(old_values.size(), advection_term.size());

  switch (weak_form)
  {
    case RunTimeParameters::ConvectiveTermWeakForm::standard:
    {
      // Loop over quadrature points
      for (std::size_t q=0; q<advection_term.size(); ++q)
        advection_term[q] =
              beta[0] * old_gradients[q] * old_values[q]
              +
              beta[1] * old_old_gradients[q] * old_old_values[q];
      break;
    }
    case RunTimeParameters::ConvectiveTermWeakForm::skewsymmetric:
    {
      // Loop over quadrature points
      for (std::size_t q=0; q<advection_term.size(); ++q)
      {
        const double old_divergence = trace(old_gradients[q]);
        const double old_old_divergence = trace(old_old_gradients[q]);

        advection_term[q] =
              beta[0] *
              (old_gradients[q] * old_values[q]
               +
               0.5 * old_divergence * old_values[q])
              +
              beta[1] *
              (old_old_gradients[q] * old_old_values[q]
               +
               0.5 * old_old_divergence * old_old_values[q]);
      }
      break;
    }
    case RunTimeParameters::ConvectiveTermWeakForm::divergence:
    {
      // Loop over quadrature points
      for (std::size_t q=0; q<advection_term.size(); ++q)
      {
        const double old_divergence = trace(old_gradients[q]);
        const double old_old_divergence = trace(old_old_gradients[q]);

        advection_term[q] =
              beta[0] *
              (old_gradients[q] * old_values[q]
               +
               old_divergence * old_values[q])
              +
              beta[1] *
              (old_old_gradients[q] * old_old_values[q]
               +
               old_old_divergence * old_old_values[q]);
      }
      break;
    }
    case RunTimeParameters::ConvectiveTermWeakForm::rotational:
    {
      typename FEValuesViews::Vector<dim>::curl_type old_curl, old_old_curl;

      // The minus sign in the argument of cross_product_2d
      // method is due to how the method is defined.
      if constexpr(dim == 2)
      {
         // Loop over quadrature points
        for (std::size_t q=0; q<advection_term.size(); ++q)
        {
          old_curl[0] = old_gradients[q][1][0] - old_gradients[q][0][1];
          old_old_curl[0] = old_old_gradients[q][1][0] - old_old_gradients[q][0][1];

          advection_term[q] =
              beta[0] * old_curl * cross_product_2d(-old_values[q])
              +
              beta[1] * old_old_curl * cross_product_2d(-old_old_values[q]);
        }
      }
      else if constexpr(dim == 3)
      {

        // Loop over quadrature points
        for (std::size_t q=0; q<advection_term.size(); ++q)
        {
          old_curl[0] = old_gradients[q][3][2] - old_gradients[q][2][3];
          old_curl[1] = old_gradients[q][0][3] - old_gradients[q][3][0];
          old_curl[2] = old_gradients[q][1][0] - old_gradients[q][0][1];
          old_old_curl[0] = old_old_gradients[q][3][2] - old_old_gradients[q][2][3];
          old_old_curl[1] = old_old_gradients[q][0][3] - old_old_gradients[q][3][0];
          old_old_curl[2] = old_old_gradients[q][1][0] - old_old_gradients[q][0][1];

          advection_term[q] =
              beta[0] * cross_product_3d(old_curl, old_values[q])
              +
              beta[1] * cross_product_3d(old_old_curl, old_old_values[q]);
        }
      }
      break;
    }
    default:
      Assert(false, ExcNotImplemented());
  }
}


template <int dim>
void compute_advection_matrix_for_bc
(const RunTimeParameters::ConvectiveTermWeakForm  weak_form,
 const std::vector<Tensor<1,dim>>   &phi,
 const std::vector<Tensor<2,dim>>   &grad_phi,
 const Tensor<1,dim>                &old_value,
 const Tensor<1,dim>                &old_old_value,
 const Tensor<2,dim>                &old_gradient,
 const Tensor<2,dim>                &old_old_gradient,
 const std::vector<double>          &eta,
 const double                        JxW_value,
 const unsigned int                  i,
 FullMatrix<double>                 &local_matrix)
{
  AssertDimension(phi.size(), local_matrix.m());
  AssertDimension(grad_phi.size(), local_matrix.m());

  const Tensor<1,dim> extrapolated_value = eta[0] * old_value + eta[1] * old_old_value;

  switch (weak_form)
  {
    case RunTimeParameters::ConvectiveTermWeakForm::standard:
    {
      // Loop over the i-th column's rows of the local matrix
      for (std::size_t j=0; j<local_matrix.m(); ++j)
        local_matrix(j, i) +=
          phi[j] * grad_phi[i] * extrapolated_value * JxW_value;
      break;
    }
    case RunTimeParameters::ConvectiveTermWeakForm::skewsymmetric:
    {
      const double old_divergence = trace(old_gradient);
      const double old_old_divergence = trace(old_old_gradient);

      const double extrapolated_divergence =
          eta[0] * old_divergence + eta[1] * old_old_divergence;

      // Loop over the i-th column's rows of the local matrix
      for (std::size_t j=0; j<local_matrix.m(); ++j)
        local_matrix(j, i) +=
            (phi[j] * grad_phi[i] * extrapolated_value +
             0.5 * extrapolated_divergence * phi[j] * phi[i]) * JxW_value;
      break;
    }
    case RunTimeParameters::ConvectiveTermWeakForm::divergence:
    {
      const double old_divergence = trace(old_gradient);
      const double old_old_divergence = trace(old_old_gradient);

      const double extrapolated_divergence =
          eta[0] * old_divergence + eta[1] * old_old_divergence;

      // Loop over the i-th column's rows of the local matrix
      for (std::size_t j=0; j<local_matrix.m(); ++j)
        local_matrix(j, i) +=
            (phi[j] * grad_phi[i] * extrapolated_value +
             extrapolated_divergence * phi[j] * phi[i]) * JxW_value;
      break;
    }
    case RunTimeParameters::ConvectiveTermWeakForm::rotational:
    {
      typename FEValuesViews::Vector<dim>::curl_type curl_phi;

      // The minus sign in the argument of cross_product_2d
      // method is due to how the method is defined.
      if constexpr(dim == 2)
      {
        curl_phi[0] = grad_phi[i][1][0] - grad_phi[i][0][1];

        // Loop over the i-th column's rows of the local matrix
        for (std::size_t j=0; j<local_matrix.m(); ++j)
          local_matrix(j, i) +=
            phi[j] * curl_phi[0] * cross_product_2d(-extrapolated_value) * JxW_value;
      }
      else if constexpr(dim == 3)
      {
        curl_phi[0] = grad_phi[i][3][2] - grad_phi[i][2][3];
        curl_phi[1] = grad_phi[i][0][3] - grad_phi[i][3][0];
        curl_phi[2] = grad_phi[i][1][0] - grad_phi[i][0][1];

        // Loop over the i-th column's rows of the local matrix
        for (std::size_t j=0; j<local_matrix.m(); ++j)
          local_matrix(j, i) +=
            phi[j] * cross_product_3d(curl_phi, extrapolated_value) * JxW_value;
      }
      break;
    }
    default:
      Assert(false, ExcNotImplemented());
  };

}

}  // namespace

using Copy = AssemblyData::NavierStokesProjection::DiffusionStepRHS::Copy;



template <int dim>
void NavierStokesProjection<dim>::
assemble_diffusion_step_rhs()
{
  if (parameters.verbose)
    *pcout << "  Navier Stokes: Assembling diffusion step's right hand side...";

  TimerOutput::Scope  t(*computing_timer, "Navier Stokes: Diffusion step - RHS assembly");

  // Reset data
  diffusion_step_rhs  = 0.;

  // Dummy finite element for when there is no buoyancy
  const FE_Nothing<dim> dummy_fe(1);

  // Create pointer to the pertinent finite element
  const FiniteElement<dim> * const temperature_fe_ptr =
          (temperature.get() != nullptr) ? &temperature->get_finite_element() : &dummy_fe;

    // Initiate the quadrature formula for exact numerical integration
  const QGauss<dim>   quadrature_formula(velocity->fe_degree() + 2);

  // Initiate the face quadrature formula for exact numerical integration
  const QGauss<dim-1> face_quadrature_formula(velocity->fe_degree() + 2);

  // Set up the lambda function for the local assembly operation
  using Scratch = AssemblyData::NavierStokesProjection::DiffusionStepRHS::Scratch<dim>;
  auto worker =
    [this](const typename DoFHandler<dim>::active_cell_iterator &cell,
           Scratch  &scratch,
           Copy     &data)
    {
      this->assemble_local_diffusion_step_rhs(cell,
                                              scratch,
                                              data);
    };

  // Set up the lambda function for the copy local to global operation
  auto copier =
    [this](const Copy   &data)
    {
      this->copy_local_to_global_diffusion_step_rhs(data);
    };

  // Assemble using the WorkStream approach
  using CellFilter =
    FilteredIterator<typename DoFHandler<dim>::active_cell_iterator>;

  const UpdateFlags velocity_update_flags = update_values|
                                            update_gradients|
                                            update_quadrature_points|
                                            update_JxW_values;
  const UpdateFlags velocity_face_update_flags = update_values|
                                                 update_quadrature_points|
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
           face_quadrature_formula,
           velocity->get_finite_element(),
           velocity_update_flags,
           velocity_face_update_flags,
           pressure->get_finite_element(),
           update_values,
           *temperature_fe_ptr,
           update_values),
   Copy(velocity->get_finite_element().dofs_per_cell));

  // Compress global data
  diffusion_step_rhs.compress(VectorOperation::add);

  // Compute the L2 norm of the right hand side
  norm_diffusion_rhs = diffusion_step_rhs.l2_norm();

  if (parameters.verbose)
    *pcout << " done!" << std::endl
           << "    Right-hand side's L2-norm = "
           << std::scientific << std::setprecision(6)
           << norm_diffusion_rhs
           << std::endl;
}



template <int dim>
void NavierStokesProjection<dim>::assemble_local_diffusion_step_rhs
(const typename DoFHandler<dim>::active_cell_iterator                 &cell,
 AssemblyData::NavierStokesProjection::DiffusionStepRHS::Scratch<dim> &scratch,
 Copy &data)
{
  const typename Entities::VectorBoundaryConditions<dim>::NeumannBCMapping
  &neumann_bcs = velocity->get_neumann_boundary_conditions();

  // Reset local data
  data.local_rhs                          = 0.;
  data.local_matrix_for_inhomogeneous_bc  = 0.;

  // VSIMEX coefficients
  const std::vector<double> alpha = time_stepping.get_alpha();
  const std::vector<double> beta  = time_stepping.get_beta();
  const std::vector<double> gamma = time_stepping.get_gamma();

  // Taylor extrapolation coefficients
  const std::vector<double> eta   = time_stepping.get_eta();

  // Local to global indices mapping
  cell->get_dof_indices(data.local_dof_indices);

  // Initialize weak forms of the right-hand side's terms
  std::vector<Tensor<1,dim>>  acceleration_term(scratch.n_q_points);
  std::vector<double>         pressure_gradient_term(scratch.n_q_points);
  std::vector<Tensor<2,dim>>  diffusion_term(scratch.n_q_points);
  std::vector<Tensor<1,dim>>  body_force_term(scratch.n_q_points);
  std::vector<Tensor<1,dim>>  buoyancy_term(scratch.n_q_points);
  std::vector<Tensor<1,dim>>  coriolis_acceleration_term(scratch.n_q_points);
  std::vector<Tensor<1,dim>>  advection_term(scratch.n_q_points);

  // Velocity
  scratch.velocity_fe_values.reinit(cell);

  const FEValuesExtractors::Vector  vector_extractor(0);

  scratch.velocity_fe_values[vector_extractor].get_function_values(
    velocity->old_solution,
    scratch.old_velocity_values);

  scratch.velocity_fe_values[vector_extractor].get_function_values(
    velocity->old_old_solution,
    scratch.old_old_velocity_values);

  scratch.velocity_fe_values[vector_extractor].get_function_gradients(
    velocity->old_solution,
    scratch.old_velocity_gradients);

  scratch.velocity_fe_values[vector_extractor].get_function_gradients(
    velocity->old_old_solution,
    scratch.old_old_velocity_gradients);

  // Pressure
  typename DoFHandler<dim>::active_cell_iterator
  pressure_cell(&velocity->get_triangulation(),
                 cell->level(),
                 cell->index(),
                &pressure->get_dof_handler());

  scratch.pressure_fe_values.reinit(pressure_cell);

  scratch.pressure_fe_values.get_function_values(
    pressure->old_solution,
    scratch.old_pressure_values);

  // Phi
  scratch.pressure_fe_values.get_function_values(
    phi->old_solution,
    scratch.old_phi_values);

  scratch.pressure_fe_values.get_function_values(
    phi->old_old_solution,
    scratch.old_old_phi_values);

  // Body force term
  if (body_force_ptr != nullptr)
  {
    compute_body_force(body_force_ptr,
                       scratch.velocity_fe_values.get_quadrature_points(),
                       gamma,
                       time_stepping.get_previous_time(),
                       time_stepping.get_current_time(),
                       time_stepping.get_next_time(),
                       body_force_term);
  }

  // Buoyancy term
  if (temperature != nullptr)
  {
    compute_buoyancy_term(gravity_vector_ptr,
                          *temperature,
                          cell,
                          scratch.velocity_fe_values.get_quadrature_points(),
                          beta,
                          parameters.C3,
                          scratch.temperature_fe_values,
                          buoyancy_term);
  }

  // Coriolis acceleration term
  if (angular_velocity_vector_ptr != nullptr)
  {
    compute_coriolis_acceleration_term(angular_velocity_vector_ptr,
                                       scratch.old_velocity_values,
                                       scratch.old_old_velocity_values,
                                       beta,
                                       time_stepping.get_previous_time(),
                                       time_stepping.get_current_time(),
                                       parameters.C1,
                                       coriolis_acceleration_term);
  }

  // Advection term
  if (parameters.convective_term_time_discretization ==
      RunTimeParameters::ConvectiveTermTimeDiscretization::fully_explicit)
  {
  }

  // Loop over quadrature points
  for (unsigned int q = 0; q < scratch.n_q_points; ++q)
  {
    // Evaluate the weak form of the right-hand side's terms at
    // the quadrature point
    acceleration_term[q] =
              alpha[1] / time_stepping.get_next_step_size() *
              scratch.old_velocity_values[q]
              +
              alpha[2] / time_stepping.get_next_step_size() *
              scratch.old_old_velocity_values[q];

    pressure_gradient_term[q] =
              parameters.C6 *
              (scratch.old_pressure_values[q]
               -
               previous_step_sizes[0] / time_stepping.get_next_step_size() *
               alpha[1] / previous_alpha_zeros[0] *
               scratch.old_phi_values[q]
               -
               previous_step_sizes[1] / time_stepping.get_next_step_size() *
               alpha[2] / previous_alpha_zeros[1] *
               scratch.old_old_phi_values[q]);

    diffusion_term[q] =
              parameters.C2 *
              (gamma[1] *
               scratch.old_velocity_gradients[q]
               +
               gamma[2] *
               scratch.old_old_velocity_gradients[q]);

    // Extract test function values at the quadrature points
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      scratch.phi[i]      = scratch.velocity_fe_values[vector_extractor].value(i,q);
      scratch.grad_phi[i] = scratch.velocity_fe_values[vector_extractor].gradient(i,q);
      scratch.div_phi[i]  = scratch.velocity_fe_values[vector_extractor].divergence(i,q);
    }

    // Loop over local degrees of freedom
    for (unsigned int i = 0; i < scratch.dofs_per_cell; ++i)
    {
      data.local_rhs(i) +=
                    (scratch.div_phi[i] * pressure_gradient_term[q] -
                     scalar_product(scratch.grad_phi[i], diffusion_term[q]) +
                     scratch.phi[i] *
                     (body_force_term[q] - buoyancy_term[q] -
                      coriolis_acceleration_term[q] - acceleration_term[q] -
                      advection_term[q])
                    ) *
                    scratch.velocity_fe_values.JxW(q);

      // Loop over the i-th column's rows of the local matrix
      // in case of inhomogeneous Dirichlet boundary conditions
      if (velocity->get_constraints().is_inhomogeneously_constrained(
            data.local_dof_indices[i]))
      {
        for (unsigned int j = 0; j < scratch.dofs_per_cell; ++j)
        {
          data.local_matrix_for_inhomogeneous_bc(j,i) +=
                (alpha[0] / time_stepping.get_next_step_size() *
                 scratch.phi[j] *
                 scratch.phi[i]
                 +
                 gamma[0] * parameters.C2 *
                 scalar_product(scratch.grad_phi[j],
                                scratch.grad_phi[i])) *
                scratch.velocity_fe_values.JxW(q);
        } // Loop over the i-th column's rows of the local matrix

        if (parameters.convective_term_time_discretization ==
            RunTimeParameters::ConvectiveTermTimeDiscretization::semi_implicit)
          compute_advection_matrix_for_bc(parameters.convective_term_weak_form,
                                          scratch.phi,
                                          scratch.grad_phi,
                                          scratch.old_velocity_values[q],
                                          scratch.old_old_velocity_values[q],
                                          scratch.old_velocity_gradients[q],
                                          scratch.old_old_velocity_gradients[q],
                                          eta,
                                          scratch.velocity_fe_values.JxW(q),
                                          i,
                                          data.local_matrix_for_inhomogeneous_bc);

      }
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
          scratch.velocity_fe_face_values.reinit(cell, face);

          const types::boundary_id  boundary_id{face->boundary_id()};
          const std::vector<Point<dim>> face_quadrature_points{scratch.velocity_fe_face_values.get_quadrature_points()};

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
                scratch.velocity_fe_face_values[vector_extractor].value(i,q);

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
                scratch.velocity_fe_face_values.JxW(q);
          } // Loop over face quadrature points
        } // Loop over the faces of the cell
}



template <int dim>
void NavierStokesProjection<dim>::copy_local_to_global_diffusion_step_rhs
(const Copy &data)
{
  velocity->get_constraints().distribute_local_to_global(
                                data.local_rhs,
                                data.local_dof_indices,
                                diffusion_step_rhs,
                                data.local_matrix_for_inhomogeneous_bc);
}

} // namespace RMHD

// explicit instantiations
template void RMHD::NavierStokesProjection<2>::assemble_diffusion_step_rhs();
template void RMHD::NavierStokesProjection<3>::assemble_diffusion_step_rhs();

template void RMHD::NavierStokesProjection<2>::assemble_local_diffusion_step_rhs
(const typename DoFHandler<2>::active_cell_iterator                       &,
 RMHD::AssemblyData::NavierStokesProjection::DiffusionStepRHS::Scratch<2> &,
 RMHD::AssemblyData::NavierStokesProjection::DiffusionStepRHS::Copy       &);
template void RMHD::NavierStokesProjection<3>::assemble_local_diffusion_step_rhs
(const typename DoFHandler<3>::active_cell_iterator                       &,
 RMHD::AssemblyData::NavierStokesProjection::DiffusionStepRHS::Scratch<3> &,
 RMHD::AssemblyData::NavierStokesProjection::DiffusionStepRHS::Copy       &);

template void RMHD::NavierStokesProjection<2>::copy_local_to_global_diffusion_step_rhs
(const RMHD::AssemblyData::NavierStokesProjection::DiffusionStepRHS::Copy &);
template void RMHD::NavierStokesProjection<3>::copy_local_to_global_diffusion_step_rhs
(const RMHD::AssemblyData::NavierStokesProjection::DiffusionStepRHS::Copy &);
