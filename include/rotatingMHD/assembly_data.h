#ifndef INCLUDE_ROTATINGMHD_ASSEMBLY_DATA_H_
#define INCLUDE_ROTATINGMHD_ASSEMBLY_DATA_H_

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/numerics/matrix_tools.h>

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



namespace ProjectionSolverBase
{



namespace VectorFieldConstantMatrices
{



using Copy = Generic::Matrix::MassStiffnessCopy;

template <int dim>
struct Scratch : Generic::Matrix::Scratch<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const FiniteElement<dim>  &fe,
          const UpdateFlags         update_flags);

  Scratch(const Scratch<dim>    &data);

  std::vector<Tensor<1, dim>> phi;

  std::vector<Tensor<2, dim>> grad_phi;
};



} // namespace VectorFieldConstantMatrices



namespace ScalarFieldsConstantMatrices
{




using Copy = Generic::Matrix::MassStiffnessCopy;

template <int dim>
struct Scratch : Generic::Matrix::Scratch<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const FiniteElement<dim>  &fe,
          const UpdateFlags         update_flags);

  Scratch(const Scratch<dim>        &data);

  std::vector<double>         phi;

  std::vector<Tensor<1, dim>> grad_phi;
};



} // namespace ScalarFieldsConstantMatrices



} // namespace ProjectionSolverBase


namespace NavierStokesProjection
{

namespace VelocityConstantMatrices
{

using Copy = Generic::Matrix::MassStiffnessCopy;

template <int dim>
struct Scratch : Generic::Matrix::Scratch<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const FiniteElement<dim>  &fe,
          const UpdateFlags         update_flags);

  Scratch(const Scratch<dim>    &data);

  std::vector<Tensor<1, dim>> phi;

  std::vector<Tensor<2, dim>> grad_phi;
};

} // namespace VelocityConstantMatrices

namespace PressureConstantMatrices
{

using Copy = Generic::Matrix::MassStiffnessCopy;

template <int dim>
struct Scratch : Generic::Matrix::Scratch<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const FiniteElement<dim>  &fe,
          const UpdateFlags         update_flags);

  Scratch(const Scratch<dim>        &data);

  std::vector<double>         phi;

  std::vector<Tensor<1, dim>> grad_phi;
};

} // namespace PressureConstantMatrices

namespace AdvectionMatrix
{

using Copy = Generic::Matrix::Copy;

template <int dim>
struct Scratch : Generic::Matrix::Scratch<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const FiniteElement<dim>  &fe,
          const UpdateFlags         update_flags);

  Scratch(const Scratch<dim>    &data);

  /*! @note Should I make it global inside the AssemblyData namespace
      or leave it locally in each struct for readability? */
  using curl_type = typename FEValuesViews::Vector< dim >::curl_type;

  std::vector<Tensor<1,dim>>  old_velocity_values;

  std::vector<Tensor<1,dim>>  old_old_velocity_values;

  std::vector<double>         old_velocity_divergences;

  std::vector<double>         old_old_velocity_divergences;

  std::vector<curl_type>      old_velocity_curls;

  std::vector<curl_type>      old_old_velocity_curls;

  std::vector<Tensor<1,dim>>  phi;

  std::vector<Tensor<2,dim>>  grad_phi;

  std::vector<curl_type>      curl_phi;
};

} // namespace AdvectionMatrix

namespace DiffusionStepRHS
{

using Copy = Generic::RightHandSide::Copy;

template <int dim>
struct Scratch : ScratchBase<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const Quadrature<dim-1>   &face_quadrature_formula,
          const FiniteElement<dim>  &velocity_fe,
          const UpdateFlags         velocity_update_flags,
          const UpdateFlags         velocity_face_update_flags,
          const FiniteElement<dim>  &pressure_fe,
          const UpdateFlags         pressure_update_flags,
          const FiniteElement<dim>  &temperature_fe,
          const UpdateFlags         temperature_update_flags);

  Scratch(const Scratch<dim>    &data);

  using curl_type = typename FEValuesViews::Vector< dim >::curl_type;

  FEValues<dim>               velocity_fe_values;

  FEFaceValues<dim>           velocity_fe_face_values;

  FEValues<dim>               pressure_fe_values;

  FEValues<dim>               temperature_fe_values;

  const unsigned int          n_face_q_points;

  std::vector<double>         old_pressure_values;

  std::vector<double>         old_phi_values;

  std::vector<double>         old_old_phi_values;

  std::vector<Tensor<1,dim>>  old_velocity_values;

  std::vector<Tensor<1,dim>>  old_old_velocity_values;

  std::vector<Tensor<2,dim>>  old_velocity_gradients;

  std::vector<Tensor<2,dim>>  old_old_velocity_gradients;

  std::vector<double>         old_velocity_divergences;

  std::vector<double>         old_old_velocity_divergences;

  std::vector<curl_type>      old_velocity_curls;

  std::vector<curl_type>      old_old_velocity_curls;

  curl_type                   old_angular_velocity_value;

  curl_type                   old_old_angular_velocity_value;

  std::vector<double>         old_temperature_values;

  std::vector<double>         old_old_temperature_values;

  std::vector<Tensor<1,dim>>  gravity_vector_values;

  std::vector<Tensor<1,dim>>  body_force_values;

  std::vector<Tensor<1,dim>>  old_body_force_values;

  std::vector<Tensor<1,dim>>  old_old_body_force_values;

  std::vector<Tensor<1,dim>>  neumann_bc_values;

  std::vector<Tensor<1,dim>>  old_neumann_bc_values;

  std::vector<Tensor<1,dim>>  old_old_neumann_bc_values;

  std::vector<Tensor<1,dim>>  phi;

  std::vector<Tensor<2,dim>>  grad_phi;

  std::vector<double>         div_phi;

  std::vector<curl_type>      curl_phi;

  std::vector<Tensor<1,dim>>  face_phi;
};

} // DiffusionStepRHS

namespace ProjectionStepRHS
{

struct Copy : CopyBase
{
  Copy(const unsigned int dofs_per_cell);

  Vector<double>      local_projection_step_rhs;

  Vector<double>      local_correction_step_rhs;
};

template <int dim>
struct Scratch : ScratchBase<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const FiniteElement<dim>  &velocity_fe,
          const UpdateFlags         velocity_update_flags,
          const FiniteElement<dim>  &pressure_fe,
          const UpdateFlags         pressure_update_flags);

  Scratch(const Scratch<dim>    &data);

  FEValues<dim>       velocity_fe_values;

  FEValues<dim>       pressure_fe_values;

  std::vector<double> velocity_divergences;

  std::vector<double> phi;
};

} // ProjectionStepRHS

namespace PoissonStepRHS
{

using Copy = Generic::RightHandSide::Copy;

template <int dim>
struct Scratch : ScratchBase<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const Quadrature<dim-1>   &face_quadrature_formula,
          const FiniteElement<dim>  &velocity_fe,
          const UpdateFlags         velocity_update_flags,
          const UpdateFlags         velocity_face_update_flags,
          const FiniteElement<dim>  &pressure_fe,
          const UpdateFlags         pressure_update_flags,
          const UpdateFlags         pressure_face_update_flags,
          const FiniteElement<dim>  &temperature_fe,
          const UpdateFlags         temperature_update_flags);

  Scratch(const Scratch<dim>    &data);

  using curl_type = typename FEValuesViews::Vector< dim >::curl_type;

  FEValues<dim>               velocity_fe_values;

  FEFaceValues<dim>           velocity_fe_face_values;

  FEValues<dim>               pressure_fe_values;

  FEFaceValues<dim>           pressure_fe_face_values;

  FEValues<dim>               temperature_fe_values;

  const unsigned int          n_face_q_points;

  std::vector<Tensor<1,dim>>  velocity_values;

  std::vector<Tensor<1,dim>>  velocity_laplacians;

  curl_type                   angular_velocity_value;

  std::vector<double>         temperature_values;

  std::vector<Tensor<1,dim>>  gravity_vector_values;

  std::vector<Tensor<1,dim>>  body_force_values;

  std::vector<Tensor<1,dim>>  normal_vectors;

  std::vector<Tensor<1,dim>>  grad_phi;

  std::vector<double>         face_phi;
};

} // namespace PoissonStepRHS

} // namespace NavierStokesProjection

namespace HeatEquation
{

namespace ConstantMatrices
{

using Copy = Generic::Matrix::MassStiffnessCopy;

template<int dim>
struct Scratch : Generic::Matrix::Scratch<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const FiniteElement<dim>  &fe,
          const UpdateFlags         update_flags);

  Scratch(const Scratch<dim>    &data);

  std::vector<double>         phi;

  std::vector<Tensor<1, dim>> grad_phi;
};

} // namespace ConstantMatrices

namespace AdvectionMatrix
{

using Copy = Generic::Matrix::Copy;

template <int dim>
struct Scratch : ScratchBase<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const FiniteElement<dim>  &temperature_fe,
          const UpdateFlags         temperature_update_flags,
          const FiniteElement<dim>  &velocity_fe,
          const UpdateFlags         velocity_update_flags);

  Scratch(const Scratch<dim>    &data);

  FEValues<dim>               temperature_fe_values;

  FEValues<dim>               velocity_fe_values;

  std::vector<Tensor<1,dim>>  velocity_values;

  std::vector<Tensor<1,dim>>  old_velocity_values;

  std::vector<Tensor<1,dim>>  old_old_velocity_values;

  std::vector<double>         phi;

  std::vector<Tensor<1,dim>>  grad_phi;
};

} // namespace AdvectionMatrix

namespace RightHandSide
{

using Copy = Generic::RightHandSide::Copy;

template <int dim>
struct Scratch : ScratchBase<dim>
{
  Scratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const Quadrature<dim-1>   &face_quadrature_formula,
          const FiniteElement<dim>  &temperature_fe,
          const UpdateFlags         temperature_update_flags,
          const UpdateFlags         temperature_face_update_flags,
          const FiniteElement<dim>  &velocity_fe,
          const UpdateFlags         velocity_update_flags);

  Scratch(const Scratch<dim>    &data);

  FEValues<dim>               temperature_fe_values;

  FEFaceValues<dim>           temperature_fe_face_values;

  FEValues<dim>               velocity_fe_values;

  const unsigned int          n_face_q_points;

  std::vector<double>         old_temperature_values;

  std::vector<double>         old_old_temperature_values;

  std::vector<Tensor<1,dim>>  velocity_values;

  std::vector<Tensor<1,dim>>  old_velocity_values;

  std::vector<Tensor<1,dim>>  old_old_velocity_values;

  std::vector<Tensor<1,dim>>  old_temperature_gradients;

  std::vector<Tensor<1,dim>>  old_old_temperature_gradients;

  std::vector<double>         source_term_values;

  std::vector<double>         old_source_term_values;

  std::vector<double>         old_old_source_term_values;

  std::vector<double>         neumann_bc_values;

  std::vector<double>         old_neumann_bc_values;

  std::vector<double>         old_old_neumann_bc_values;

  std::vector<double>         phi;

  std::vector<Tensor<1,dim>>  grad_phi;

  std::vector<double>         face_phi;
};

} // namespace RightHandSide

} // namespace HeatEquation

} // namespace AssemblyData

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_ASSEMBLY_DATA_H_ */
