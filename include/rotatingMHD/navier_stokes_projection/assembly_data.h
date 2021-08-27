#ifndef INCLUDE_ROTATINGMHD_NAVIER_STOKES_PROJECTION_ASSEMBLY_DATA_H_
#define INCLUDE_ROTATINGMHD_NAVIER_STOKES_PROJECTION_ASSEMBLY_DATA_H_

#include <deal.II/base/tensor.h>

#include <rotatingMHD/assembly_data_base.h>

#include <vector>

namespace RMHD
{

namespace AssemblyData
{

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
struct HDScratch : ScratchBase<dim>
{
  HDScratch(const Mapping<dim>        &mapping,
            const Quadrature<dim>     &quadrature_formula,
            const Quadrature<dim-1>   &face_quadrature_formula,
            const FiniteElement<dim>  &velocity_fe,
            const UpdateFlags         velocity_update_flags,
            const UpdateFlags         velocity_face_update_flags,
            const FiniteElement<dim>  &pressure_fe,
            const UpdateFlags         pressure_update_flags);

  HDScratch(const HDScratch<dim>    &data);

  using curl_type = typename FEValuesViews::Vector< dim >::curl_type;

  FEValues<dim>               velocity_fe_values;

  FEFaceValues<dim>           velocity_fe_face_values;

  FEValues<dim>               pressure_fe_values;

  const unsigned int          n_face_q_points;

  std::vector<double>         old_pressure_values;

  std::vector<double>         old_phi_values;

  std::vector<double>         old_old_phi_values;

  std::vector<Tensor<1,dim>>  old_velocity_values;

  std::vector<Tensor<1,dim>>  old_old_velocity_values;

  std::vector<Tensor<2,dim>>  old_velocity_gradients;

  std::vector<Tensor<2,dim>>  old_old_velocity_gradients;

  std::vector<Tensor<1,dim>>  neumann_bc_values;

  std::vector<Tensor<1,dim>>  old_neumann_bc_values;

  std::vector<Tensor<1,dim>>  old_old_neumann_bc_values;

  std::vector<Tensor<1,dim>>  phi;

  std::vector<Tensor<2,dim>>  grad_phi;

  std::vector<double>         div_phi;

  std::vector<Tensor<1,dim>>  face_phi;
};



template <int dim>
struct HDCScratch : HDScratch<dim>
{
  HDCScratch(const Mapping<dim>        &mapping,
             const Quadrature<dim>     &quadrature_formula,
             const Quadrature<dim-1>   &face_quadrature_formula,
             const FiniteElement<dim>  &velocity_fe,
             const UpdateFlags         velocity_update_flags,
             const UpdateFlags         velocity_face_update_flags,
             const FiniteElement<dim>  &pressure_fe,
             const UpdateFlags         pressure_update_flags,
             const FiniteElement<dim>  &temperature_fe,
             const UpdateFlags         temperature_update_flags);

  HDCScratch(const HDCScratch<dim>    &data);

  FEValues<dim>               temperature_fe_values;

  std::vector<double>         old_temperature_values;
  std::vector<double>         old_old_temperature_values;

  std::vector<Tensor<1,dim>>  gravity_vector_values;
};



template <int dim>
struct MHDScratch : HDScratch<dim>
{
  MHDScratch(const Mapping<dim>        &mapping,
             const Quadrature<dim>     &quadrature_formula,
             const Quadrature<dim-1>   &face_quadrature_formula,
             const FiniteElement<dim>  &velocity_fe,
             const UpdateFlags         velocity_update_flags,
             const UpdateFlags         velocity_face_update_flags,
             const FiniteElement<dim>  &pressure_fe,
             const UpdateFlags         pressure_update_flags,
             const FiniteElement<dim>  &magnetic_fe,
             const UpdateFlags         magnetic_update_flags);

  MHDScratch(const MHDScratch<dim>    &data);

  FEValues<dim>               magnetic_fe_values;

  std::vector<Tensor<1, dim>> old_magnetic_field_values;
  std::vector<Tensor<1, dim>> old_old_magnetic_field_values;

  std::vector<typename HDScratch<dim>::curl_type> old_magnetic_field_curls;
  std::vector<typename HDScratch<dim>::curl_type> old_old_magnetic_field_curls;

};



template <int dim>
struct MHDCScratch : HDCScratch<dim>
{
  MHDCScratch(const Mapping<dim>        &mapping,
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
              const UpdateFlags         magnetic_update_flags);

  MHDCScratch(const MHDCScratch<dim>    &data);

  FEValues<dim>               magnetic_fe_values;

  std::vector<Tensor<1, dim>> old_magnetic_field_values;
  std::vector<Tensor<1, dim>> old_old_magnetic_field_values;

  std::vector<typename HDCScratch<dim>::curl_type> old_magnetic_field_curls;
  std::vector<typename HDCScratch<dim>::curl_type> old_old_magnetic_field_curls;

};



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

  std::vector<double>         old_temperature_values;

  std::vector<double>         old_old_temperature_values;

  std::vector<Tensor<1,dim>>  neumann_bc_values;

  std::vector<Tensor<1,dim>>  old_neumann_bc_values;

  std::vector<Tensor<1,dim>>  old_old_neumann_bc_values;

  std::vector<Tensor<1,dim>>  phi;

  std::vector<Tensor<2,dim>>  grad_phi;

  std::vector<double>         div_phi;

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

} // namespace AssemblyData

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_NAVIER_STOKES_PROJECTION_ASSEMBLY_DATA_H_ */
