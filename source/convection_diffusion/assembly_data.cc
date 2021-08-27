#include <rotatingMHD/convection_diffusion/assembly_data.h>

namespace RMHD
{

namespace AssemblyData
{

namespace HeatEquation
{

namespace ConstantMatrices
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

} // namespace ConstantMatrices

namespace AdvectionMatrix
{

template <int dim>
Scratch<dim>::Scratch(
  const Mapping<dim>        &mapping,
  const Quadrature<dim>     &quadrature_formula,
  const FiniteElement<dim>  &temperature_fe,
  const UpdateFlags         temperature_update_flags,
  const FiniteElement<dim>  &velocity_fe,
  const UpdateFlags         velocity_update_flags)
:
ScratchBase<dim>(quadrature_formula,
                 temperature_fe),
temperature_fe_values(mapping,
                      temperature_fe,
                      quadrature_formula,
                      temperature_update_flags),
velocity_fe_values(mapping,
                   velocity_fe,
                   quadrature_formula,
                   velocity_update_flags),
velocity_values(this->n_q_points),
old_velocity_values(this->n_q_points),
old_old_velocity_values(this->n_q_points),
phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell)
{}



template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
temperature_fe_values(data.temperature_fe_values.get_mapping(),
                      data.temperature_fe_values.get_fe(),
                      data.temperature_fe_values.get_quadrature(),
                      data.temperature_fe_values.get_update_flags()),
velocity_fe_values(data.velocity_fe_values.get_mapping(),
                   data.velocity_fe_values.get_fe(),
                   data.velocity_fe_values.get_quadrature(),
                   data.velocity_fe_values.get_update_flags()),
velocity_values(data.n_q_points),
old_velocity_values(data.n_q_points),
old_old_velocity_values(data.n_q_points),
phi(data.dofs_per_cell),
grad_phi(data.dofs_per_cell)
{}

// explicit instantiations
template struct Scratch<2>;
template struct Scratch<3>;

} // namespace AdvectionMatrix

namespace RightHandSide
{

template <int dim>
CDScratch<dim>::CDScratch
(const Mapping<dim>        &mapping,
 const Quadrature<dim>     &quadrature_formula,
 const Quadrature<dim-1>   &face_quadrature_formula,
 const FiniteElement<dim>  &temperature_fe,
 const UpdateFlags         temperature_update_flags,
 const UpdateFlags         temperature_face_update_flags)
:
ScratchBase<dim>(quadrature_formula,
                 temperature_fe),
temperature_fe_values(mapping,
                      temperature_fe,
                      quadrature_formula,
                      temperature_update_flags),
temperature_fe_face_values(mapping,
                           temperature_fe,
                           face_quadrature_formula,
                           temperature_face_update_flags),
n_face_q_points(face_quadrature_formula.size()),
old_temperature_values(this->n_q_points),
old_old_temperature_values(this->n_q_points),
old_temperature_gradients(this->n_q_points),
old_old_temperature_gradients(this->n_q_points),
neumann_bc_values(n_face_q_points),
old_neumann_bc_values(n_face_q_points),
old_old_neumann_bc_values(n_face_q_points),
phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell),
face_phi(this->dofs_per_cell)
{}



template <int dim>
CDScratch<dim>::CDScratch(const CDScratch<dim> &data)
:
ScratchBase<dim>(data),
temperature_fe_values(
  data.temperature_fe_values.get_mapping(),
  data.temperature_fe_values.get_fe(),
  data.temperature_fe_values.get_quadrature(),
  data.temperature_fe_values.get_update_flags()),
temperature_fe_face_values(
  data.temperature_fe_face_values.get_mapping(),
  data.temperature_fe_face_values.get_fe(),
  data.temperature_fe_face_values.get_quadrature(),
  data.temperature_fe_face_values.get_update_flags()),
n_face_q_points(data.n_face_q_points),
old_temperature_values(this->n_q_points),
old_old_temperature_values(this->n_q_points),
old_temperature_gradients(this->n_q_points),
old_old_temperature_gradients(this->n_q_points),
neumann_bc_values(n_face_q_points),
old_neumann_bc_values(n_face_q_points),
old_old_neumann_bc_values(n_face_q_points),
phi(this->dofs_per_cell),
grad_phi(this->dofs_per_cell),
face_phi(this->dofs_per_cell)
{}



template <int dim>
HDCDScratch<dim>::HDCDScratch
(const Mapping<dim>        &mapping,
 const Quadrature<dim>     &quadrature_formula,
 const Quadrature<dim-1>   &face_quadrature_formula,
 const FiniteElement<dim>  &temperature_fe,
 const UpdateFlags         temperature_update_flags,
 const UpdateFlags         temperature_face_update_flags,
 const FiniteElement<dim>  &velocity_fe,
 const UpdateFlags         velocity_update_flags)
:
CDScratch<dim>(mapping,
               quadrature_formula,
               face_quadrature_formula,
               temperature_fe,
               temperature_update_flags,
               temperature_face_update_flags),
velocity_fe_values(mapping,
                   velocity_fe,
                   quadrature_formula,
                   velocity_update_flags),
old_velocity_values(this->n_q_points),
old_old_velocity_values(this->n_q_points)
{}



template <int dim>
HDCDScratch<dim>::HDCDScratch(const HDCDScratch<dim> &data)
:
CDScratch<dim>(data),
velocity_fe_values(data.velocity_fe_values.get_mapping(),
                   data.velocity_fe_values.get_fe(),
                   data.velocity_fe_values.get_quadrature(),
                   data.velocity_fe_values.get_update_flags()),
old_velocity_values(this->n_q_points),
old_old_velocity_values(this->n_q_points)
{}

// explicit instantiations
template struct CDScratch<2>;
template struct CDScratch<3>;

template struct HDCDScratch<2>;
template struct HDCDScratch<3>;

} // namespace RightHandSide

} // namespace HeatEquation

} // namespace AssemblyData

} // namespace RMHD
