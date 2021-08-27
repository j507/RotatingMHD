#ifndef INCLUDE_ROTATINGMHD_ASSEMBLY_DATA_BASE_H_
#define INCLUDE_ROTATINGMHD_ASSEMBLY_DATA_BASE_H_

#include <deal.II/base/quadrature.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

namespace RMHD
{

using namespace dealii;

namespace AssemblyData
{

struct CopyBase
{
  CopyBase(const unsigned int dofs_per_cell);

  unsigned int                          dofs_per_cell;

  std::vector<types::global_cell_index> local_dof_indices;
};

template <int dim>
struct ScratchBase
{
  ScratchBase(const Quadrature<dim>     &quadrature_formula,
              const FiniteElement<dim>  &fe);

  ScratchBase(const ScratchBase<dim>    &data);

  const unsigned int  n_q_points;

  const unsigned int  dofs_per_cell;
};

namespace Generic
{

namespace Matrix
{

struct Copy : CopyBase
{
  Copy(const unsigned int dofs_per_cell);

  FullMatrix<double>  local_matrix;
};

struct MassStiffnessCopy : CopyBase
{
  MassStiffnessCopy(const unsigned int dofs_per_cell);

  FullMatrix<double>  local_mass_matrix;

  FullMatrix<double>  local_stiffness_matrix;
};

template <int dim>
struct Scratch : ScratchBase<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const FiniteElement<dim>  &fe,
          const UpdateFlags         update_flags);

  Scratch(const Scratch<dim>        &data);

  FEValues<dim> fe_values;
};

} // namespace Matrix

namespace RightHandSide
{

struct Copy : CopyBase
{
  Copy(const unsigned int dofs_per_cell);

  Vector<double>      local_rhs;

  FullMatrix<double>  local_matrix_for_inhomogeneous_bc;
};

template <int dim>
struct Scratch : ScratchBase<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const FiniteElement<dim>  &fe,
          const UpdateFlags         update_flags);

  Scratch(const Scratch<dim>        &data);

  FEValues<dim> fe_values;
};

} // namespace RightHandSide

} // namespace Generic

} // namespace AssemblyData

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_ASSEMBLY_DATA_BASE_H_ */
