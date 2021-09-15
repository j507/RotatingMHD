#include <rotatingMHD/assembly_data.h>

namespace RMHD
{

namespace AssemblyData
{

CopyBase::CopyBase(const unsigned int dofs_per_cell)
:
dofs_per_cell(dofs_per_cell),
local_dof_indices(dofs_per_cell)
{}

template <int dim>
ScratchBase<dim>::ScratchBase(
  const Quadrature<dim>     &quadrature_formula,
  const FiniteElement<dim>  &fe)
:
n_q_points(quadrature_formula.size()),
dofs_per_cell(fe.dofs_per_cell)
{}

template <int dim>
ScratchBase<dim>::ScratchBase(const ScratchBase<dim> &data)
:
n_q_points(data.n_q_points),
dofs_per_cell(data.dofs_per_cell)
{}

namespace Generic
{

namespace Matrix
{

Copy::Copy(const unsigned int dofs_per_cell)
:
CopyBase(dofs_per_cell),
local_matrix(dofs_per_cell, dofs_per_cell)
{}

MassStiffnessCopy::MassStiffnessCopy(const unsigned int dofs_per_cell)
:
CopyBase(dofs_per_cell),
local_mass_matrix(dofs_per_cell, dofs_per_cell),
local_stiffness_matrix(dofs_per_cell, dofs_per_cell)
{}

template <int dim>
Scratch<dim>::Scratch(
  const Mapping<dim>        &mapping,
  const Quadrature<dim>     &quadrature_formula,
  const FiniteElement<dim>  &fe,
  const UpdateFlags         update_flags)
:
ScratchBase<dim>(quadrature_formula, fe),
fe_values(mapping,
          fe,
          quadrature_formula,
          update_flags)
{}

template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
fe_values(data.fe_values.get_mapping(),
          data.fe_values.get_fe(),
          data.fe_values.get_quadrature(),
          data.fe_values.get_update_flags())
{}

} // namespace Matrix

namespace RightHandSide
{

Copy::Copy(const unsigned int dofs_per_cell)
:
CopyBase(dofs_per_cell),
local_rhs(dofs_per_cell),
local_matrix_for_inhomogeneous_bc(dofs_per_cell, dofs_per_cell)
{}

template <int dim>
Scratch<dim>::Scratch(
  const Mapping<dim>        &mapping,
  const Quadrature<dim>     &quadrature_formula,
  const FiniteElement<dim>  &fe,
  const UpdateFlags         update_flags)
:
ScratchBase<dim>(quadrature_formula, fe),
fe_values(mapping,
          fe,
          quadrature_formula,
          update_flags)
{}

template <int dim>
Scratch<dim>::Scratch(const Scratch<dim> &data)
:
ScratchBase<dim>(data),
fe_values(data.fe_values.get_mapping(),
          data.fe_values.get_fe(),
          data.fe_values.get_quadrature(),
          data.fe_values.get_update_flags())
{}

} // namespace RightHandSide

} // namespace Generic



namespace ProjectionSolverBase
{



namespace VectorFieldConstantMatrices
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



} // namespace VectorFieldConstantMatrices



namespace ScalarFieldsConstantMatrices
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



} // namespace ScalarFieldsConstantMatrices


} // namespace ProjectionSolverBase

} // namespace AssemblyData

} // namespace RMHD

template struct RMHD::AssemblyData::ScratchBase<2>;
template struct RMHD::AssemblyData::ScratchBase<3>;

template struct RMHD::AssemblyData::Generic::Matrix::Scratch<2>;
template struct RMHD::AssemblyData::Generic::Matrix::Scratch<3>;

template struct RMHD::AssemblyData::Generic::RightHandSide::Scratch<2>;
template struct RMHD::AssemblyData::Generic::RightHandSide::Scratch<3>;

template struct RMHD::AssemblyData::ProjectionSolverBase::VectorFieldConstantMatrices::Scratch<2>;
template struct RMHD::AssemblyData::ProjectionSolverBase::VectorFieldConstantMatrices::Scratch<3>;

template struct RMHD::AssemblyData::ProjectionSolverBase::ScalarFieldsConstantMatrices::Scratch<2>;
template struct RMHD::AssemblyData::ProjectionSolverBase::ScalarFieldsConstantMatrices::Scratch<3>;

