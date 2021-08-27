/*
 * assembly_data.cc
 *
 *  Created on: Jul 19, 2020
 *      Author: sg
 */

#include <rotatingMHD/navier_stokes_projection/assembly_data.h>

namespace RMHD
{

using namespace dealii;

namespace AssemblyData
{

namespace NavierStokesProjection
{

namespace VelocityConstantMatrices
{

template <int dim>
Scratch<dim>::Scratch(
  const Mapping<dim>        &mapping,
  const Quadrature<dim>     &quadrature_formula,
  const FiniteElement<dim>  &fe,
  const UpdateFlags         update_flags)
:
Generic::Matrix::Scratch<dim>(mapping,
                              quadrature_formula,
                              fe,
                              update_flags),
phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell)
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
Generic::Matrix::Scratch<dim>(data),
phi(data.dofs_per_cell),
grad_phi(data.dofs_per_cell)
{}

// explicit instantiations
template struct Scratch<2>;
template struct Scratch<3>;

} // namespace VelocityConstantMatrices

namespace PressureConstantMatrices
{

template <int dim>
Scratch<dim>::Scratch(
  const Mapping<dim>        &mapping,
  const Quadrature<dim>     &quadrature_formula,
  const FiniteElement<dim>  &fe,
  const UpdateFlags         update_flags)
:
Generic::Matrix::Scratch<dim>(mapping,
                              quadrature_formula,
                              fe,
                              update_flags),
phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell)
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
Generic::Matrix::Scratch<dim>(data),
phi(data.dofs_per_cell),
grad_phi(data.dofs_per_cell)
{}

// explicit instantiations
template struct Scratch<2>;
template struct Scratch<3>;

} // namespace PressureConstantMatrices

namespace AdvectionMatrix
{

template <int dim>
Scratch<dim>::Scratch(
  const Mapping<dim>        &mapping,
  const Quadrature<dim>     &quadrature_formula,
  const FiniteElement<dim>  &fe,
  const UpdateFlags         update_flags)
:
Generic::Matrix::Scratch<dim>(mapping,
                              quadrature_formula,
                              fe,
                              update_flags),
old_velocity_values(this->n_q_points),
old_old_velocity_values(this->n_q_points),
old_velocity_divergences(this->n_q_points),
old_old_velocity_divergences(this->n_q_points),
old_velocity_curls(this->n_q_points),
old_old_velocity_curls(this->n_q_points),
phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell),
curl_phi(this->dofs_per_cell)
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
Generic::Matrix::Scratch<dim>(data),
old_velocity_values(data.n_q_points),
old_old_velocity_values(data.n_q_points),
old_velocity_divergences(data.n_q_points),
old_old_velocity_divergences(data.n_q_points),
old_velocity_curls(data.n_q_points),
old_old_velocity_curls(data.n_q_points),
phi(data.dofs_per_cell),
grad_phi(data.dofs_per_cell),
curl_phi(data.dofs_per_cell)
{}

// explicit instantiations
template struct Scratch<2>;
template struct Scratch<3>;

} // namespace AdvectionMatrix

namespace DiffusionStepRHS
{

template <int dim>
HDScratch<dim>::HDScratch
(const Mapping<dim>        &mapping,
 const Quadrature<dim>     &quadrature_formula,
 const Quadrature<dim-1>   &face_quadrature_formula,
 const FiniteElement<dim>  &velocity_fe,
 const UpdateFlags         velocity_update_flags,
 const UpdateFlags         velocity_face_update_flags,
 const FiniteElement<dim>  &pressure_fe,
 const UpdateFlags         pressure_update_flags)
:
ScratchBase<dim>(quadrature_formula,
                 velocity_fe),
velocity_fe_values(mapping,
                   velocity_fe,
                   quadrature_formula,
                   velocity_update_flags),
velocity_fe_face_values(mapping,
                        velocity_fe,
                        face_quadrature_formula,
                        velocity_face_update_flags),
pressure_fe_values(mapping,
                   pressure_fe,
                   quadrature_formula,
                   pressure_update_flags),
n_face_q_points(face_quadrature_formula.size()),
old_pressure_values(this->n_q_points),
old_phi_values(this->n_q_points),
old_old_phi_values(this->n_q_points),
old_velocity_values(this->n_q_points),
old_old_velocity_values(this->n_q_points),
old_velocity_gradients(this->n_q_points),
old_old_velocity_gradients(this->n_q_points),
neumann_bc_values(n_face_q_points),
old_neumann_bc_values(n_face_q_points),
old_old_neumann_bc_values(n_face_q_points),
phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell),
div_phi(this->dofs_per_cell),
face_phi(this->dofs_per_cell)
{}



template <int dim>
HDScratch<dim>::HDScratch(const HDScratch<dim> &data)
:
ScratchBase<dim>(data),
velocity_fe_values(data.velocity_fe_values.get_mapping(),
                   data.velocity_fe_values.get_fe(),
                   data.velocity_fe_values.get_quadrature(),
                   data.velocity_fe_values.get_update_flags()),
velocity_fe_face_values(data.velocity_fe_face_values.get_mapping(),
                        data.velocity_fe_face_values.get_fe(),
                        data.velocity_fe_face_values.get_quadrature(),
                        data.velocity_fe_face_values.get_update_flags()),
pressure_fe_values(data.pressure_fe_values.get_mapping(),
                   data.pressure_fe_values.get_fe(),
                   data.pressure_fe_values.get_quadrature(),
                   data.pressure_fe_values.get_update_flags()),
n_face_q_points(data.n_face_q_points),
old_pressure_values(this->n_q_points),
old_phi_values(this->n_q_points),
old_old_phi_values(this->n_q_points),
old_velocity_values(this->n_q_points),
old_old_velocity_values(this->n_q_points),
old_velocity_gradients(this->n_q_points),
old_old_velocity_gradients(this->n_q_points),
neumann_bc_values(n_face_q_points),
old_neumann_bc_values(n_face_q_points),
old_old_neumann_bc_values(n_face_q_points),
phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell),
div_phi(this->dofs_per_cell),
face_phi(this->dofs_per_cell)
{}



template <int dim>
HDCScratch<dim>::HDCScratch
(const Mapping<dim>        &mapping,
 const Quadrature<dim>     &quadrature_formula,
 const Quadrature<dim-1>   &face_quadrature_formula,
 const FiniteElement<dim>  &velocity_fe,
 const UpdateFlags         velocity_update_flags,
 const UpdateFlags         velocity_face_update_flags,
 const FiniteElement<dim>  &pressure_fe,
 const UpdateFlags         pressure_update_flags,
 const FiniteElement<dim>  &temperature_fe,
 const UpdateFlags         temperature_update_flags)
:
HDScratch<dim>(mapping,
               quadrature_formula,
               face_quadrature_formula,
               velocity_fe,
               velocity_update_flags,
               velocity_face_update_flags,
               pressure_fe,
               pressure_update_flags),
temperature_fe_values(mapping,
                      temperature_fe,
                      quadrature_formula,
                      temperature_update_flags),
old_temperature_values(this->n_q_points),
old_old_temperature_values(this->n_q_points),
gravity_vector_values(this->n_q_points)
{}



template <int dim>
HDCScratch<dim>::HDCScratch(const HDCScratch<dim> &data)
:
HDScratch<dim>(data),
temperature_fe_values(data.temperature_fe_values.get_mapping(),
                      data.temperature_fe_values.get_fe(),
                      data.temperature_fe_values.get_quadrature(),
                      data.temperature_fe_values.get_update_flags()),
old_temperature_values(this->n_q_points),
old_old_temperature_values(this->n_q_points),
gravity_vector_values(this->n_q_points)
{}



template <int dim>
MHDScratch<dim>::MHDScratch
(const Mapping<dim>        &mapping,
 const Quadrature<dim>     &quadrature_formula,
 const Quadrature<dim-1>   &face_quadrature_formula,
 const FiniteElement<dim>  &velocity_fe,
 const UpdateFlags         velocity_update_flags,
 const UpdateFlags         velocity_face_update_flags,
 const FiniteElement<dim>  &pressure_fe,
 const UpdateFlags         pressure_update_flags,
 const FiniteElement<dim>  &magnetic_fe,
 const UpdateFlags         magnetic_update_flags)
:
HDScratch<dim>(mapping,
               quadrature_formula,
               face_quadrature_formula,
               velocity_fe,
               velocity_update_flags,
               velocity_face_update_flags,
               pressure_fe,
               pressure_update_flags),
magnetic_fe_values(mapping,
                   magnetic_fe,
                   quadrature_formula,
                   magnetic_update_flags),
old_magnetic_field_values(this->n_q_points),
old_old_magnetic_field_values(this->n_q_points),
old_magnetic_field_curls(this->n_q_points),
old_old_magnetic_field_curls(this->n_q_points)
{}



template <int dim>
MHDScratch<dim>::MHDScratch(const MHDScratch<dim> &data)
:
HDScratch<dim>(data),
magnetic_fe_values(data.magnetic_fe_values.get_mapping(),
                   data.magnetic_fe_values.get_fe(),
                   data.magnetic_fe_values.get_quadrature(),
                   data.magnetic_fe_values.get_update_flags()),
old_magnetic_field_values(this->n_q_points),
old_old_magnetic_field_values(this->n_q_points),
old_magnetic_field_curls(this->n_q_points),
old_old_magnetic_field_curls(this->n_q_points)
{}



template <int dim>
MHDCScratch<dim>::MHDCScratch
(const Mapping<dim>        &mapping,
 const Quadrature<dim>     &quadrature_formula,
 const Quadrature<dim-1>   &face_quadrature_formula,
 const FiniteElement<dim>  &velocity_fe,
 const UpdateFlags         velocity_update_flags,
 const UpdateFlags         velocity_face_update_flags,
 const FiniteElement<dim>  &pressure_fe,
 const UpdateFlags         pressure_update_flags,
 const FiniteElement<dim>  &temperature_fe,
 const UpdateFlags         temperature_update_flags,
 const FiniteElement<dim>  &magnetic_fe,
 const UpdateFlags         magnetic_update_flags)
:
HDCScratch<dim>(mapping,
               quadrature_formula,
               face_quadrature_formula,
               velocity_fe,
               velocity_update_flags,
               velocity_face_update_flags,
               pressure_fe,
               pressure_update_flags,
               temperature_fe,
               temperature_update_flags),
magnetic_fe_values(mapping,
                   magnetic_fe,
                   quadrature_formula,
                   magnetic_update_flags),
old_magnetic_field_values(this->n_q_points),
old_old_magnetic_field_values(this->n_q_points),
old_magnetic_field_curls(this->n_q_points),
old_old_magnetic_field_curls(this->n_q_points)
{}



template <int dim>
MHDCScratch<dim>::MHDCScratch(const MHDCScratch<dim> &data)
:
HDCScratch<dim>(data),
magnetic_fe_values(data.magnetic_fe_values.get_mapping(),
                   data.magnetic_fe_values.get_fe(),
                   data.magnetic_fe_values.get_quadrature(),
                   data.magnetic_fe_values.get_update_flags()),
old_magnetic_field_values(this->n_q_points),
old_old_magnetic_field_values(this->n_q_points),
old_magnetic_field_curls(this->n_q_points),
old_old_magnetic_field_curls(this->n_q_points)
{}

// explicit instantiations
template struct HDScratch<2>;
template struct HDScratch<3>;

template struct HDCScratch<2>;
template struct HDCScratch<3>;

template struct MHDScratch<2>;
template struct MHDScratch<3>;

template struct MHDCScratch<2>;
template struct MHDCScratch<3>;

} // namespace DiffusionStepRHS

namespace ProjectionStepRHS
{

Copy::Copy(const unsigned int dofs_per_cell)
:
CopyBase(dofs_per_cell),
local_projection_step_rhs(dofs_per_cell),
local_correction_step_rhs(dofs_per_cell)
{}



template <int dim>
Scratch<dim>::Scratch
(const Mapping<dim>        &mapping,
 const Quadrature<dim>     &quadrature_formula,
 const FiniteElement<dim>  &velocity_fe,
 const UpdateFlags         velocity_update_flags,
 const FiniteElement<dim>  &pressure_fe,
 const UpdateFlags         pressure_update_flags)
:
ScratchBase<dim>(quadrature_formula,
                 pressure_fe),
velocity_fe_values(
  mapping,
  velocity_fe,
  quadrature_formula,
  velocity_update_flags),
pressure_fe_values(
  mapping,
  pressure_fe,
  quadrature_formula,
  pressure_update_flags),
velocity_divergences(this->n_q_points),
phi(this->dofs_per_cell)
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
velocity_fe_values(
  data.velocity_fe_values.get_mapping(),
  data.velocity_fe_values.get_fe(),
  data.velocity_fe_values.get_quadrature(),
  data.velocity_fe_values.get_update_flags()),
pressure_fe_values(
  data.pressure_fe_values.get_mapping(),
  data.pressure_fe_values.get_fe(),
  data.pressure_fe_values.get_quadrature(),
  data.pressure_fe_values.get_update_flags()),
velocity_divergences(this->n_q_points),
phi(this->dofs_per_cell)
{}

// explicit instantiations
template struct Scratch<2>;
template struct Scratch<3>;

} // namespace ProjectionStepRHS

namespace PoissonStepRHS
{

template <int dim>
Scratch<dim>::Scratch
(const Mapping<dim>        &mapping,
 const Quadrature<dim>     &quadrature_formula,
 const Quadrature<dim-1>   &face_quadrature_formula,
 const FiniteElement<dim>  &velocity_fe,
 const UpdateFlags         velocity_update_flags,
 const UpdateFlags         velocity_face_update_flags,
 const FiniteElement<dim>  &pressure_fe,
 const UpdateFlags         pressure_update_flags,
 const UpdateFlags         pressure_face_update_flags,
 const FiniteElement<dim>  &temperature_fe,
 const UpdateFlags         temperature_update_flags)
:
ScratchBase<dim>(quadrature_formula,
                 pressure_fe),
velocity_fe_values(
  mapping,
  velocity_fe,
  quadrature_formula,
  velocity_update_flags),
velocity_fe_face_values(
  mapping,
  velocity_fe,
  face_quadrature_formula,
  velocity_face_update_flags),
pressure_fe_values(
  mapping,
  pressure_fe,
  quadrature_formula,
  pressure_update_flags),
pressure_fe_face_values(
  mapping,
  pressure_fe,
  face_quadrature_formula,
  pressure_face_update_flags),
temperature_fe_values(
  mapping,
  temperature_fe,
  quadrature_formula,
  temperature_update_flags),
n_face_q_points(face_quadrature_formula.size()),
velocity_values(this->n_q_points),
velocity_laplacians(n_face_q_points),
temperature_values(this->n_q_points),
gravity_vector_values(this->n_q_points),
body_force_values(this->n_q_points),
normal_vectors(n_face_q_points),
grad_phi(this->dofs_per_cell),
face_phi(this->dofs_per_cell)
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
velocity_fe_values(
  data.velocity_fe_values.get_mapping(),
  data.velocity_fe_values.get_fe(),
  data.velocity_fe_values.get_quadrature(),
  data.velocity_fe_values.get_update_flags()),
velocity_fe_face_values(
  data.velocity_fe_face_values.get_mapping(),
  data.velocity_fe_face_values.get_fe(),
  data.velocity_fe_face_values.get_quadrature(),
  data.velocity_fe_face_values.get_update_flags()),
pressure_fe_values(
  data.pressure_fe_values.get_mapping(),
  data.pressure_fe_values.get_fe(),
  data.pressure_fe_values.get_quadrature(),
  data.pressure_fe_values.get_update_flags()),
pressure_fe_face_values(
  data.pressure_fe_face_values.get_mapping(),
  data.pressure_fe_face_values.get_fe(),
  data.pressure_fe_face_values.get_quadrature(),
  data.pressure_fe_face_values.get_update_flags()),
temperature_fe_values(
  data.temperature_fe_values.get_mapping(),
  data.temperature_fe_values.get_fe(),
  data.temperature_fe_values.get_quadrature(),
  data.temperature_fe_values.get_update_flags()),
n_face_q_points(data.n_face_q_points),
velocity_values(this->n_q_points),
velocity_laplacians(n_face_q_points),
temperature_values(this->n_q_points),
gravity_vector_values(this->n_q_points),
body_force_values(this->n_q_points),
normal_vectors(n_face_q_points),
grad_phi(this->dofs_per_cell),
face_phi(this->dofs_per_cell)
{}

// explicit instantiations
template struct Scratch<2>;
template struct Scratch<3>;

} // namespace PoissonStepRHS

} // namespace NavierStokesProjection

} // namespace AssemblyData

} // namespace RMHD
