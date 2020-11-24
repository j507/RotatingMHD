#ifndef INCLUDE_ROTATINGMHD_HEAT_EQUATION_H_
#define INCLUDE_ROTATINGMHD_HEAT_EQUATION_H_

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/tensor_function.h>

#include <rotatingMHD/assembly_data.h>
#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/global.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/time_discretization.h>

#include <memory>
#include <string>
#include <vector>

namespace RMHD
{

using namespace dealii;

/*!
 * @class HeatEquation
 * 
 * @brief Solves the heat equation.
 * 
 * @details This version is parallelized using deal.ii's MPI facilities and
 * relies either on the Trilinos or the PETSc library. Moreover, for the time
 * discretization an implicit-explicit scheme (IMEX) with variable step size is
 * used.
 * The heat equation solved is derived from the balance of internal
 * energy
 * \f[
 *  \rho \dfrac{\d u}{\d t} = - \nabla \cdot \bs{q} + \rho r + 
 *  \bs{T} \cdott (\nabla \otimes \bs{v}), 
 *  \quad \forall (\bs{x}, t) \in \Omega \times \left[0, T \right]
 * \f]
 * where \f$ u \f$, \f$ \, \bs{q} \f$, \f$ \, r \f$, \f$ \, \bs{T} \f$,
 * \f$ \, \bs{v} \f$, \f$ \, \bs{x} \f$, \f$ \, t \f$, \f$ \,\Omega \f$ and
 * \f$ \, T \f$ are the
 * specific internal energy, heat flux vector, specific heat source,
 * stress tensor, velocity, position vector, time, domain and final
 * time respectively. Considering an isotropic and incompressible fluid,
 * whose heat flux vector is given by the Fourier law; whose 
 * specific heat capacity and thermal conductivity coefficient do not 
 * depend on the temperature; and neglecting the internal dissipation,
 * the adimensional heat equation reduces to
 * \f[
 *  \pd{\vartheta}{t} + \bs{v} \cdot \nabla \vartheta = \dfrac{1}{\mathit{Pe}}
 *  \nabla^2 \vartheta + r, 
 *  \quad \forall (\bs{x}, t) \in \Omega \times \left[0, T \right]
 * \f]
 * where \f$ \vartheta \f$ and \f$ \mathit{Pe} \f$ are the adimensional 
 * temperature and the Peclet number respectively. Do note that we
 * reuse the variables names to denote their adimensional counterpart.
 * @todo Documentation
 * @todo Add dissipation term.
 * @todo Add consistent forms of the advection term.
 */

template <int dim>
class HeatEquation
{

public:
  /*!
   * @brief The constructor of the HeatEquation class for the case 
   * where there is no advection.
   * 
   * @details Stores a local reference to the input parameters and 
   * pointers for the mapping and terminal output entities.
   */
  HeatEquation
  (const RunTimeParameters::ParameterSet        &parameters,
   TimeDiscretization::VSIMEXMethod             &time_stepping,
   std::shared_ptr<Entities::ScalarEntity<dim>> &temperature,
   const std::shared_ptr<Mapping<dim>>          external_mapping =
       std::shared_ptr<Mapping<dim>>(),
   const std::shared_ptr<ConditionalOStream>    external_pcout =
       std::shared_ptr<ConditionalOStream>(),
   const std::shared_ptr<TimerOutput>           external_timer =
       std::shared_ptr<TimerOutput>());

  /*!
   * @brief The constructor of the HeatEquation class for the case
   * where the velocity field is given by a VectorEntity instance.
   * 
   * @details Stores a local reference to the input parameters and 
   * pointers for the mapping and terminal output entities.
   */
  HeatEquation
  (const RunTimeParameters::ParameterSet        &parameters,
   TimeDiscretization::VSIMEXMethod             &time_stepping,
   std::shared_ptr<Entities::ScalarEntity<dim>> &temperature,
   std::shared_ptr<Entities::VectorEntity<dim>> &velocity,
   const std::shared_ptr<Mapping<dim>>          external_mapping =
       std::shared_ptr<Mapping<dim>>(),
   const std::shared_ptr<ConditionalOStream>    external_pcout =
       std::shared_ptr<ConditionalOStream>(),
   const std::shared_ptr<TimerOutput>           external_timer =
       std::shared_ptr<TimerOutput>());

  /*!
   * @brief The constructor of the HeatEquation class for the case
   *  where the velocity is given by a TensorFunction.
   * 
   * @details Stores a local reference to the input parameters and 
   * pointers for the mapping and terminal output entities.
   */
  HeatEquation
  (const RunTimeParameters::ParameterSet        &parameters,
   TimeDiscretization::VSIMEXMethod             &time_stepping,
   std::shared_ptr<Entities::ScalarEntity<dim>> &temperature,
   std::shared_ptr<TensorFunction<1, dim>>      &velocity,
   const std::shared_ptr<Mapping<dim>>          external_mapping =
       std::shared_ptr<Mapping<dim>>(),
   const std::shared_ptr<ConditionalOStream>    external_pcout =
       std::shared_ptr<ConditionalOStream>(),
   const std::shared_ptr<TimerOutput>           external_timer =
       std::shared_ptr<TimerOutput>());
  /*!
   *  @brief Setups and initializes all the internal entities for
   *  the heat equation problem.
   *
   *  @details Initializes the vector and matrices using the information
   *  contained in the ScalarEntity and VectorEntity structs passed on
   *  in the constructor (The temperature and the velocity respectively).
   */
  void setup();

  /*!
   *  @brief Sets the source term of the problem.
   *
   *  @details Stores the memory address of the source term function in 
   *  the pointer @ref suppler_term_ptr.
   */
  void set_source_term(Function<dim> &source_term);

  /*!
   * @brief Computes the temperature field at \f$ t = t_1 \f$ using a 
   * first order time discretization scheme.
   */
  void initialize();

  /*!
   *  @brief Solves the heat equation problem for one single timestep.
   */
  void solve();

  /*!
   *  @brief Returns the norm of the right hand side for the last solved
   * step.
   */
  double get_rhs_norm() const;

private:
  /*!
   * @brief A reference to the parameters which control the solution process.
   */
  const RunTimeParameters::ParameterSet         &parameters;

  /*!
   * @brief The MPI communicator which is equal to `MPI_COMM_WORLD`.
   */
  const MPI_Comm                                &mpi_communicator;

  /*!
   * @brief A reference to the class controlling the temporal discretization.
   */
  const TimeDiscretization::VSIMEXMethod        &time_stepping;

  /*!
   * @brief A shared pointer to a conditional output stream object.
   */
  std::shared_ptr<ConditionalOStream>           pcout;

  /*!
   * @brief A shared pointer to a monitor of the computing times.
   */
  std::shared_ptr<TimerOutput>                  computing_timer;

  /*!
   * @brief A shared pointer to the mapping to be used throughout the solver.
   */
  std::shared_ptr<Mapping<dim>>                 mapping;

  /*!
   * @brief A shared pointer to the entity of the temperature field.
   */
  std::shared_ptr<Entities::ScalarEntity<dim>>  temperature;

  /*!
   * @brief A shared pointer to the entity of velocity field.
   */
  std::shared_ptr<Entities::VectorEntity<dim>>  velocity;

  /*!
   * @brief A shared pointer to the TensorFunction of the velocity field.
   */
  std::shared_ptr<TensorFunction<1,dim>>        velocity_function_ptr;

  /*!
   * @brief A pointer to the supply term function.
   */
  Function<dim>                                 *source_term_ptr;

  /*!
   * @brief System matrix for the heat equation.
   * @details For 
   */
  LinearAlgebra::MPI::SparseMatrix              system_matrix;

  /*!
   * @brief Mass matrix of the temperature.
   * @details This matrix does not change in every timestep. It is stored in
   * memory because otherwise an assembly would be required if the timestep
   * changes.
   * @todo Add formulas
   */
  LinearAlgebra::MPI::SparseMatrix              mass_matrix;

  /*!
   * @brief Stiffness matrix of the temperature.
   * @details This matrix does not change in every timestep. It is stored in
   * memory because otherwise an assembly would be required if the timestep
   * changes.
   * @todo Add formulas
   */
  LinearAlgebra::MPI::SparseMatrix              stiffness_matrix;

  /*!
   * @brief Sum of the mass and stiffness matrix of the temperature.
   * @details If the time step size is constant, this matrix does not 
   * change each step.
   * @todo Add formulas
   */
  LinearAlgebra::MPI::SparseMatrix              mass_plus_stiffness_matrix;

  /*!
   * @brief Advection matrix of the temperature.
   * @details This matrix changes in every timestep and is therefore also
   * assembled in every timestep.
   * @todo Add formulas
   */
  LinearAlgebra::MPI::SparseMatrix              advection_matrix;


  /*!
   * @brief Vector representing the right-hand side of the linear system.
   * @todo Add formulas
   */
  LinearAlgebra::MPI::Vector                    rhs;

  /*!
   * @brief The \f$ L_2 \f$ norm of the right hand side
   */
  double                                        rhs_norm;

  /*!
   * @brief Vector representing the sum of the time discretization terms
   * that belong to the right hand side of the equation.
   * @details For example: A BDF2 scheme with a constant time step
   * expands the time derivative in three terms
   * \f[
   * \frac{\partial u}{\partial t} \approx 
   * \frac{1.5}{\Delta t} u^{n} - \frac{2}{\Delta t} u^{n-1}
   * + \frac{0.5}{\Delta t} u^{n-2},
   * \f] 
   * the last two terms are known quantities so they belong to the 
   * right hand side of the equation. Therefore, we define
   * \f[
   * u_\textrm{tmp} = - \frac{2}{\Delta t} u^{n-1}
   * + \frac{0.5}{\Delta t} u^{n-2},
   * \f].
   * which we use when assembling the right-hand side of the linear
   * system
   */
  LinearAlgebra::MPI::Vector                    temperature_tmp;

  /*!
   * @brief A vector representing the extrapolated velocity at the
   * current timestep using a Taylor expansion
   * @details The Taylor expansion is given by
   * \f{eqnarray*}{
   * u^{n} &\approx& u^{n-1} + \frac{\partial u^{n-1}}{\partial t} \Delta t \\
   *         &\approx& u^{n-1} + \frac{u^{n-1} - u^{n-2}}{\Delta t} \Delta t \\
   *         &\approx& 2 u^{n-1} - u^{n-2}.
   * \f}
   * In the case of a variable time step the approximation is given by
   * \f[
   * u^{n} \approx (1 + \omega) u^{n-1} - \omega u^{n-2}
   * \f] 
   * where  \f$ \omega = \frac{\Delta t_{n-1}}{\Delta t_{n-2}}.\f$
   * @attention The extrapolation is hardcoded to the second order described
   * above. First and higher order are pending.
   */
  LinearAlgebra::MPI::Vector                    extrapolated_velocity;

  /*!
   * @brief Preconditioner of the linear system.
   */
  LinearAlgebra::MPI::PreconditionILU           preconditioner;

  /*!
   * @brief Absolute tolerance of the Krylov solver.
   */
  const double                                  absolute_tolerance = 1.0e-9;

  /*!
   * @brief A flag indicating if the preconditioner is to be
   * initiated.
   */ 
  bool                                          flag_reinit_preconditioner;

  /*!
   * @brief A flag indicating if the sum of the mass and stiffness matrix
   * is to be performed.
   */ 
  bool                                          flag_add_mass_and_stiffness_matrices;

  /*!
   * @brief A flag indicating if the advection term is to be ignored.
   */ 
  bool                                          flag_ignore_advection;

  /*!
   * @brief Setup of the sparsity spatterns of the matrices.
   */
  void setup_matrices();

  /*!
   * @brief Setup of the right-hand side and the auxiliary vectors.
   */
  void setup_vectors();

  /*!
   * @brief Assemble the matrices which change only if the triangulation is
   * refined or coarsened.
   * @todo Add formulas
   */
  void assemble_constant_matrices();

  /*!
   * @brief Assemble the advection matrix.
   * @todo Add formulas
   */
  void assemble_advection_matrix();


  /*!
   * @brief Assembles the right-hand side.
   * @todo Add formulas
   */
  void assemble_rhs();

  /*!
   * @brief Assembles the linear system.
   * @todo Add formulas
   */
  void assemble_linear_system();

  /*!
   * @brief Solves the linear system.
   * @details Pending.
   */
  void solve_linear_system(const bool reinit_preconditioner);

  /*!
   * @brief This method assembles the mass matrix on a single cell.
   */
  void assemble_local_constant_matrices(
    const typename DoFHandler<dim>::active_cell_iterator    &cell,
    TemperatureConstantMatricesAssembly::LocalCellData<dim> &scratch,
    TemperatureConstantMatricesAssembly::MappingData<dim>   &data);

  /*!
   * @brief This method copies the mass matrix into its global
   * conterpart.
   */
  void copy_local_to_global_constant_matrices(
    const TemperatureConstantMatricesAssembly::MappingData<dim>  &data);

  /*!
   * @brief This method assembles the advection matrix on a single cell.
   */
  void assemble_local_advection_matrix(
    const typename DoFHandler<dim>::active_cell_iterator    &cell,
    TemperatureAdvectionMatrixAssembly::LocalCellData<dim>  &scratch,
    TemperatureAdvectionMatrixAssembly::MappingData<dim>    &data);

  /*!
   * @brief This method copies the local advection matrix into their 
   * global conterparts.
   */
  void copy_local_to_global_advection_matrix(
    const TemperatureAdvectionMatrixAssembly::MappingData<dim>  &data);


  /*!
   * @brief This method assembles the right-hand side on a single cell.
   */
  void assemble_local_rhs(
    const typename DoFHandler<dim>::active_cell_iterator    &cell,
    TemperatureRightHandSideAssembly::LocalCellData<dim> &scratch,
    TemperatureRightHandSideAssembly::MappingData<dim>   &data);

  /*!
   * @brief This method copies the local right-hand side into its global
   * conterpart.
   */
  void copy_local_to_global_rhs(
    const TemperatureRightHandSideAssembly::MappingData<dim>  &data);
};

template <int dim>
inline double HeatEquation<dim>::get_rhs_norm() const
{
  return (rhs_norm);
}

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_HEAT_EQUATION_H_ */
