#ifndef INCLUDE_ROTATINGMHD_NAVIER_STOKES_PROJECTION_H_
#define INCLUDE_ROTATINGMHD_NAVIER_STOKES_PROJECTION_H_

#include <rotatingMHD/equation_data.h>
#include <rotatingMHD/entities_structs.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/assembly_data.h>
#include <rotatingMHD/time_discretization.h>

#include <deal.II/lac/trilinos_precondition.h>

#include <string>
#include <vector>

namespace RMHD
{

using namespace dealii;


/*!
 * @brief Solves the Navier Stokes equations with the incremental pressure
 * projection scheme.
 *
 * @details This version is parallelized using deal.ii's MPI facilities and
 * relies either on the Trilinos or the PETSc library. Moreover, for the time
 * discretization an implicit-explicit scheme (IMEX) with variable step size is
 * used.
 */
template <int dim>
class NavierStokesProjection
{

public:
  NavierStokesProjection(
                const RunTimeParameters::ParameterSet   &parameters,
                Entities::VectorEntity<dim>             &velocity,
                Entities::ScalarEntity<dim>             &pressure,
                TimeDiscretization::VSIMEXCoefficients  &VSIMEX,
                TimeDiscretization::VSIMEXMethod        &time_stepping);

  void setup();


  /*!
   *  @brief Solves the problem for one single timestep.
   *
   *  @details Performs the diffusion and the projection step for one single
   *  time step and updates the member variables at the end.
   */

  void solve(const unsigned int step);
  
private:


  /*!
   *  @brief A parameter which determines the type of the pressure update.
   *
   *  @details For the pressure update after the projection step, this parameter
   *  determines whether the irrotational form of the pressure update is used or
   *  not.
   */
  RunTimeParameters::ProjectionMethod     projection_method;


  /*!
   * @brief The Reynolds number.
   *
   *  @details A parameter which determines the ratio of convection to viscous
   *  diffusion.
   */
  const double                            Re;

  /*!
   * @brief A reference to the entity of velocity field.
   */
  Entities::VectorEntity<dim>             &velocity;

  /*!
   * @brief A reference to the entity of the pressure field.
   */
  Entities::ScalarEntity<dim>             &pressure;

  TimeDiscretization::VSIMEXCoefficients  &VSIMEX;

  TimeDiscretization::VSIMEXMethod        &time_stepping;

  /*!
   * @brief System matrix used to solve for the velocity field in the diffusion
   * step.
   *
   * @details This matrix changes in every timestep because the convective term
   * needs to be assembled in every timestep. However, not the entire system
   * matrix is assembled in every timestep but only the part due to the
   * convective term.
   */
  TrilinosWrappers::SparseMatrix        velocity_system_matrix;

  /*!
   * @brief Mass matrix of the velocity.
   *
   * @details This matrix does change not in every timestep. It is stored in
   * memory because otherwise an assembly would be required if the timestep
   * changes.
   */
  TrilinosWrappers::SparseMatrix        velocity_mass_matrix;

  /*!
   * @brief Stiffness matrix of the velocity. Assembly of  the weak of the
   * Laplace operator.
   *
   * @details This matrix does not change in every timestep. It is stored in
   * memory because otherwise an assembly would be required if the timestep
   * changes.
   */
  TrilinosWrappers::SparseMatrix        velocity_laplace_matrix;

  /*!
   * @brief Sum of the mass and the stiffness matrix of the velocity.
   *
   * @details This matrix does not change in every timestep. It is stored in
   * memory because otherwise an assembly would be required if the timestep
   * changes.
   */
  TrilinosWrappers::SparseMatrix        velocity_mass_plus_laplace_matrix;

  /*!
   * @brief Matrix representing the assembly of the skew-symmetric formm of the
   * convective term.
   *
   * @details This matrix changes in every timestep and is therefore also
   * assembled in every timestep.
   */
  TrilinosWrappers::SparseMatrix        velocity_advection_matrix;


  /*!
   * @brief Auxiliary vector which represents the velocity field which is
   * extrapolated to the time of the considered timestep.
   */
  TrilinosWrappers::MPI::Vector         extrapolated_velocity;

  /*!
   * @brief Vector representing the sum of the time discretization terms
   * that belong to the right hand side of the equation.
   * @details For example: A BDF2 scheme with a constant time step
   * expands the time derivative in three terms
   * \f[
   * \frac{\partial u}{\partial t} \approx 
   * \frac{1.5}{\Delta t} u^{n+1} - \frac{2}{\Delta t} u^{n}
   * + \frac{0.5}{\Delta t} u^{n-1},
   * \f] 
   * the last two terms are known quantities so they belong to the 
   * right hand side of the equation. Therefore, we define
   * \f[
   * u_\textrm{tmp} = - \frac{2}{\Delta t} u^{n}
   * + \frac{0.5}{\Delta t} u^{n-1},
   * \f].
   * which we use when assembling the right hand side of the diffusion
   * step.
  */
  TrilinosWrappers::MPI::Vector         velocity_tmp;

  /*!
   * @brief Vector representing the right-hand side of the linear system of the
   * diffusion step.
   */
  TrilinosWrappers::MPI::Vector         velocity_rhs;

  /*!
   * @brief Mass matrix of the pressure field.
   */
  TrilinosWrappers::SparseMatrix        pressure_mass_matrix;

  /*!
   * @brief Stiffness matrix of the pressure field. Assembly of  the weak of the
   * Laplace operator.
   */
  TrilinosWrappers::SparseMatrix        pressure_laplace_matrix;

  /*!
   * @brief Vector representing the pressure used in the diffusion step.
   * @details The pressure is given by
   * \f[
   * p^{\#} = p^\textrm{k} + \frac{4}{3} \phi^\textrm{k} 
   *            - \frac{1}{3} \phi^{\textrm{k}-1}. 
   * \f] 
   * The notation is taken from the dealii tutorial 
   * <a href="https://www.dealii.org/current/doxygen/deal.II/step_35.html#Projectionmethods">step-35</a> , 
   * from which this class is based upon.
   * @attention In the Guermond paper this is an extrapolated pressure,
   * and it is also called like that in the step-35 documentation, but 
   * I do not see how the formula above is an extrapolation.
   */  
  TrilinosWrappers::MPI::Vector         pressure_tmp;

  /*!
   * @brief Vector representing the right-hand side of the linear system of the
   * projection step.
   */
  TrilinosWrappers::MPI::Vector         pressure_rhs;

  /*!
   * @brief Vector representing the pressure update of the current timestep.
   */
  TrilinosWrappers::MPI::Vector         phi;

  /*!
   * @brief Vector representing the pressure update of the previous timestep.
   */
  TrilinosWrappers::MPI::Vector         old_phi;

  TrilinosWrappers::PreconditionILU     diffusion_step_preconditioner;
  TrilinosWrappers::PreconditionILU     projection_step_preconditioner;
  TrilinosWrappers::PreconditionJacobi  correction_step_preconditioner;

  // SG thinks that all of these parameters can go into a parameter structure.
  unsigned int                          solver_max_iterations;
  unsigned int                          solver_krylov_size;
  unsigned int                          solver_off_diagonals;
  unsigned int                          solver_update_preconditioner;
  double                                solver_tolerance;
  double                                solver_diag_strength;
  bool                                  flag_adpative_time_step;


  /*!
   * @brief Setup of the sparsity spatterns of the matrices of the diffusion and
   * projection steps.
   */
  void setup_matrices();

  /*!
   * @brief Setup of the right-hand side and the auxiliary vector of the
   * diffusion and projection step.
   */
  void setup_vectors();

  /*!
   * @brief Assemble the matrices which change only if the triangulation is
   * refined or coarsened.
   */
  void assemble_constant_matrices();

  /*!
   * @brief Currently this method only sets the vector of the two pressure
   * updates @ref phi_n and @ref phi_n_minus_1 to zero.
   */
  void initialize();

  /*!
   * @brief This method performs one complete diffusion step.
   */
  void diffusion_step(const bool reinit_prec);

  /*!
   * @brief This method assembles the system matrix and the right-hand side of
   * the diffusion step.
   */
  void assemble_diffusion_step();

  /*!
   * @brief This method solves the linear system of the diffusion step.
   */
  void solve_diffusion_step(const bool reinit_prec);

  /*!
   * @brief This method performs one complete projection step.
   */
  void projection_step(const bool reinit_prec);

  /*!
   * @brief This method assembles the linear system of the projection step.
   */
  void assemble_projection_step();

  /*!
   * @brief This method solves the linear system of the projection step.
   */
  void solve_projection_step(const bool reinit_prec);

  /*!
   * @brief This method performs the pressure update of the projection step.
   */
  void pressure_correction(const bool reinit_prec);

  /*!
   * @brief This method assembles the mass and the stiffness matrix of the
   * velocity field using the WorkStream approach.
   */
  void assemble_velocity_matrices();

  /*!
   * @brief This method assembles the local mass and the local stiffness
   * matrices of the velocity field on a single cell.
   */
  void assemble_local_velocity_matrices(
    const typename DoFHandler<dim>::active_cell_iterator  &cell,
    VelocityMatricesAssembly::LocalCellData<dim>          &scratch,
    VelocityMatricesAssembly::MappingData<dim>            &data);

  /*!
   * @brief This method copies the local mass and the local stiffness matrices
   * of the velocity field on a single cell into the global matrices.
   */
  void copy_local_to_global_velocity_matrices(
    const VelocityMatricesAssembly::MappingData<dim>      &data);

  /*!
   * @brief This method copies the local mass and the local stiffness matrices
   * of the velocity field into the global matrices.
   */
  void assemble_pressure_matrices();

  /*!
   * @brief This method assembles the local mass and the local stiffness
   * matrices of the velocity field on a single cell.
   */
  void assemble_local_pressure_matrices(
    const typename DoFHandler<dim>::active_cell_iterator  &cell,
    PressureMatricesAssembly::LocalCellData<dim>          &scratch,
    PressureMatricesAssembly::MappingData<dim>            &data);

  /*!
   * @brief This method copies the local mass and the local stiffness matrices
   * of the pressure field on a single cell into the global matrices.
   */
  void copy_local_to_global_pressure_matrices(
    const PressureMatricesAssembly::MappingData<dim>      &data);

  /*!
   * @brief This method assembles the right-hand side of the diffusion step
   * using the WorkStream approach.
   */
  void assemble_diffusion_step_rhs();

  /*!
   * @brief This method assembles the local right-hand side of the diffusion
   * step on a single cell.
   */
  void assemble_local_diffusion_step_rhs(
    const typename DoFHandler<dim>::active_cell_iterator  &cell,
    VelocityRightHandSideAssembly::LocalCellData<dim>     &scratch,
    VelocityRightHandSideAssembly::MappingData<dim>       &data);

  /*!
   * @brief This method copies the local right-hand side of the diffusion step
   * into the global vector.
   */
  void copy_local_to_global_diffusion_step_rhs(
    const VelocityRightHandSideAssembly::MappingData<dim> &data);

  /*!
   * @brief This method assembles the right-hand side of the projection step
   * using the WorkStream approach.
   */
  void assemble_projection_step_rhs();

  /*!
   * @brief This method assembles the local right-hand side of the projection
   * step on a single cell.
   */
  void assemble_local_projection_step_rhs(
    const typename DoFHandler<dim>::active_cell_iterator  &cell,
    PressureRightHandSideAssembly::LocalCellData<dim>     &scratch,
    PressureRightHandSideAssembly::MappingData<dim>       &data);

  /*!
   * @brief This method copies the local right-hand side of the projection step
   * on a single cell to the global vector.
   */
  void copy_local_to_global_projection_step_rhs(
    const PressureRightHandSideAssembly::MappingData<dim> &data);

  /*!
   * @brief This method assembles the velocity advection matrix using the
   * WorkStream approach.
   */
  void assemble_velocity_advection_matrix();

  /*!
   * @brief This method assembles the local velocity advection matrix on a
   * single cell.
   */
  void assemble_local_velocity_advection_matrix(
    const typename DoFHandler<dim>::active_cell_iterator  &cell,
    AdvectionAssembly::LocalCellData<dim>                 &scratch,
    AdvectionAssembly::MappingData<dim>                   &data);

  /*!
   * @brief This method copies the local velocity advection matrix into the
   * global matrix.
   */
  void copy_local_to_global_velocity_advection_matrix(
    const AdvectionAssembly::MappingData<dim>             &data);
};

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_NAVIER_STOKES_PROJECTION_H_ */
