#ifndef INCLUDE_ROTATINGMHD_CONVECTION_DIFFUSION_ASSEMBLY_DATA_H_
#define INCLUDE_ROTATINGMHD_CONVECTION_DIFFUSION_ASSEMBLY_DATA_H_

#include <deal.II/base/tensor.h>

#include <rotatingMHD/assembly_data_base.h>

#include <vector>

namespace RMHD
{

namespace AssemblyData
{

namespace HeatEquation
{

namespace ConstantMatrices
{

using Copy = AssemblyData::Generic::Matrix::MassStiffnessCopy;

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
struct CDScratch : ScratchBase<dim>
{
  CDScratch(const Mapping<dim>        &mapping,
          const Quadrature<dim>     &quadrature_formula,
          const Quadrature<dim-1>   &face_quadrature_formula,
          const FiniteElement<dim>  &temperature_fe,
          const UpdateFlags         temperature_update_flags,
          const UpdateFlags         temperature_face_update_flags);

  CDScratch(const CDScratch<dim>    &data);

  FEValues<dim>               temperature_fe_values;

  FEFaceValues<dim>           temperature_fe_face_values;

  const unsigned int          n_face_q_points;

  std::vector<double>         old_temperature_values;

  std::vector<double>         old_old_temperature_values;

  std::vector<Tensor<1,dim>>  old_temperature_gradients;

  std::vector<Tensor<1,dim>>  old_old_temperature_gradients;

  std::vector<double>         neumann_bc_values;

  std::vector<double>         old_neumann_bc_values;

  std::vector<double>         old_old_neumann_bc_values;

  std::vector<double>         phi;

  std::vector<Tensor<1,dim>>  grad_phi;

  std::vector<double>         face_phi;
};

template <int dim>
struct HDCDScratch : CDScratch<dim>
{
  HDCDScratch(const Mapping<dim>        &mapping,
              const Quadrature<dim>     &quadrature_formula,
              const Quadrature<dim-1>   &face_quadrature_formula,
              const FiniteElement<dim>  &temperature_fe,
              const UpdateFlags         temperature_update_flags,
              const UpdateFlags         temperature_face_update_flags,
              const FiniteElement<dim>  &velocity_fe,
              const UpdateFlags         velocity_update_flags);

  HDCDScratch(const HDCDScratch<dim>    &data);

  FEValues<dim>               velocity_fe_values;

  std::vector<Tensor<1,dim>>  old_velocity_values;
  std::vector<Tensor<1,dim>>  old_old_velocity_values;

};

} // namespace RightHandSide

} // namespace HeatEquation

} // namespace AssemblyData

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_CONVECTION_DIFFUSION_ASSEMBLY_DATA_H_ */
