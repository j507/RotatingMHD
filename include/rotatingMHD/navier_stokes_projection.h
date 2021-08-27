#ifndef INCLUDE_ROTATINGMHD_NAVIER_STOKES_PROJECTION_H_
#define INCLUDE_ROTATINGMHD_NAVIER_STOKES_PROJECTION_H_

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/timer.h>

#include <rotatingMHD/angular_velocity.h>
#include <rotatingMHD/finite_element_field.h>
#include <rotatingMHD/global.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/time_discretization.h>
#include <rotatingMHD/navier_stokes_projection/assembly_data.h>

#include <array>
#include <memory>
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
 *
 * The class is coded to solve a diffusion step given by
 * \f[
 * \begin{equation}
 * \dfrac{\alpha_0^n}{\Delta t_{n-1}} \bs{u}^{n} -\dfrac{1}{\textrm{Re}}
 * \Delta \bs{u}^{n} + \bs{u}^{\star, n} \cdot (\nabla \otimes \bs{u}^{n})
 * + \frac{1}{2} (\nabla \cdot \bs{u}^{\star, n}) \bs{u}^{n} = -\nabla p^{\sharp, n}
 * - \sum_{j=1}^{2} \dfrac{\alpha_j^n}{\Delta t_{n-1}}\bs{u}^{n-j},
 * \qquad \forall(\bs{x},t)\in \Omega \times [0, T]
 * \end{equation}
 * \f]
 * with
 * \f[
 * \begin{equation}
 *  p^\sharp = p^{n-1} +
 *  \sum_{j=1}^{2} \frac{\Delta t_{n-1-j}}{\Delta t_{n-1}}
 *  \frac{\alpha_j^n}{\alpha_0^{n-j}}\phi^{n-j}
 * \end{equation}
 * \f]
 * a projection step given by
 * \f[
 * \begin{equation}
 * \Delta \phi^{n} = \dfrac{\alpha_0^n}{\Delta t_{n-1}} \nabla \cdot
 * \bs{u}^{n},  \qquad \forall (\bs{x}, t)
 * \in \Omega \times \left[0, T \right]
 * \end{equation}
 * \f]
 * and a pressure correction step given by
 * \f[
 * \begin{equation*}
 * p^{n} = p^{n-1} + \phi^{n} - \frac{\chi}{\textrm{Re}} \nabla \cdot \bs{u}^{n}
 * \end{equation*}
 * \f]
 * where \f$ \chi \f$ is either 0 or 1 denoting the standard or rotational
 * incremental scheme respectively.
 * @todo Implement a generalized extrapolation scheme.
 * @todo Expand the weak formulation for the case of unconventional
 * boundary conditions.
 * @attention The code is hardcoded for a second order time discretization
 * scheme.
 */
template <int dim>
class NavierStokesProjection
{

public:
  /*!
   * @brief The constructor of the Navier-Stokes projection class where
   * the bouyancy term is neglected.
   *
   * @details Stores local references to the input parameters and
   * pointers for the mapping and terminal output entities.
   */
  NavierStokesProjection
  (const RunTimeParameters::NavierStokesParameters  &parameters,
   TimeDiscretization::VSIMEXMethod             &time_stepping,
   std::shared_ptr<Entities::FE_VectorField<dim>> &velocity,
   std::shared_ptr<Entities::FE_ScalarField<dim>> &pressure,
   const std::shared_ptr<Mapping<dim>>          external_mapping =
       std::shared_ptr<Mapping<dim>>(),
   const std::shared_ptr<ConditionalOStream>    external_pcout =
       std::shared_ptr<ConditionalOStream>(),
   const std::shared_ptr<TimerOutput>           external_timer =
       std::shared_ptr<TimerOutput>());

  /*!
   * @brief The constructor of the Navier-Stokes projection class where
   * the buoyancy term is considered.
   *
   * @details Stores local references to the input parameters and
   * pointers for the mapping and terminal output entities.
   */
  NavierStokesProjection
  (const RunTimeParameters::NavierStokesParameters        &parameters,
   TimeDiscretization::VSIMEXMethod             &time_stepping,
   std::shared_ptr<Entities::FE_VectorField<dim>> &velocity,
   std::shared_ptr<Entities::FE_ScalarField<dim>> &pressure,
   std::shared_ptr<Entities::FE_ScalarField<dim>> &temperature,
   const std::shared_ptr<Mapping<dim>>          external_mapping =
       std::shared_ptr<Mapping<dim>>(),
   const std::shared_ptr<ConditionalOStream>    external_pcout =
       std::shared_ptr<ConditionalOStream>(),
   const std::shared_ptr<TimerOutput>           external_timer =
       std::shared_ptr<TimerOutput>());

  /*!
   * @brief The entity for the scalar field \f$ \phi \f$, which is
   * the field computed during the projection step and later used in
   * the pressure-correction step.
   */
  std::shared_ptr<Entities::FE_ScalarField<dim>>   phi;


  /*!
   *  @brief Reset all internal objects to the initial state and release memory.
   *
   *  @details @ref body_force_ptr and @ref gravity_vector_ptr are also reset.
   */
  void clear();

  /*!
   *  @brief Setups and initializes all the internal entities for
   *  the projection method problem.
   *
   *  @details Initializes the vector and matrices using the information
   *  contained in the FE_VectorField and FE_ScalarField structs passed on
   *  in the constructor (The velocity and the pressure respectively).
   *  The boolean passed as argument control if the pressure is to be
   *  normalized.
   */
  void setup();

  /*!
   *  @brief Sets the body force of the problem.
   *
   *  @details Stores the memory address of the body force function in
   *  the pointer @ref body_force.
   */
  void set_body_force(TensorFunction<1, dim> &body_force);

  /*!
   *  @brief Sets the gravity unit vector of the problem.
   *
   *  @details Stores the memory address of the gravity unit vector
   *  function in the pointer @ref gravity_vector_ptr.
   */
  void set_gravity_vector(TensorFunction<1, dim> &gravity_vector);

  /*!
   *  @brief Sets the angular velocity of the rotating frame of
   *  reference.
   *
   *  @details Stores the memory address of the angular velocity unit vector
   *  function in the pointer @ref angular_velocity_vector_ptr.
   */
  void set_angular_velocity_vector(RMHD::AngularVelocity<dim> &angular_velocity_vector);

  /*!
   *  @brief Solves the problem for one single timestep.
   *
   *  @details Performs the diffusion and the projection step for one single
   *  time step and updates the member variables at the end.
   */
  void solve();

  /*!
   *  @brief Resets the internal entity \f$ \phi \f$
   *  @details Sets all its solution vectors to zero and signals
   *  the solver to set up the entity on the next solve call.
   */
  void reset_phi();

  /*!
   *  @brief Resets the internal linear algebra members.
   *  @details Calls their respective clear method.
   */
  void reset();

  /*!
   * @brief Performs one diffusion step
   * @attention This is just a method for testing
   */
  void perform_diffusion_step();

  /*!
   *  @brief Computes Courant-Friedrichs-Lewy number for the current
   *  velocity field.
   *
   *  @details It is given by
   * \f[
   *    C = \Delta t_{n-1} \min_{K \in \Omega_\textrm{h}}
   *    \left\lbrace \frac{\max_{P \in K} { \left\lVert \bs{v} \right\rVert}}{h_K} \right\rbrace
   * \f]
   * where \f$ C \f$ is the Courant number, \f$ K\f$ denotes the
   * \f$ K\f$-th cell of the tessallation, \f$ \Omega_\textrm{h}\f$ the tessallation,
   * \f$ P \f$ a quadrature point inside the \f$ K\f$-th cell,
   * \f$ \bs{v} \f$ the velocity, \f$ h_K\f$ the largest diagonal of the \f$ K\f$-th
   * cell.
   */
  double get_cfl_number();

  /*!
   * @brief Returns the norm of the right hand side of the diffusion step.
   */
  double get_diffusion_step_rhs_norm() const;

  /*!
   * @brief Returns the norm of the right hand side of the projection step.
   */
  double get_projection_step_rhs_norm() const;

private:
  /*!
   * @brief A reference to the parameters which control the solution process.
   */
  const RunTimeParameters::NavierStokesParameters  &parameters;

  /*!
   * @brief The MPI communicator which is equal to `MPI_COMM_WORLD`.
   */
  const MPI_Comm                          mpi_communicator;

  /*!
   * @brief Pointer to a conditional output stream object.
   */
  std::shared_ptr<ConditionalOStream>     pcout;

  /*!
   * @brief Pointer to a monitor of the computing times.
   */
  std::shared_ptr<TimerOutput>            computing_timer;

  /*!
   * @brief Pointer to the mapping to be used throughout the solver.
   */
  std::shared_ptr<Mapping<dim>>           mapping;

  /*!
   * @brief A reference to the entity of velocity field.
   */
  std::shared_ptr<Entities::FE_VectorField<dim>>  velocity;

  /*!
   * @brief A reference to the entity of the pressure field.
   */
  std::shared_ptr<Entities::FE_ScalarField<dim>>  pressure;

  /*!
   * @brief A reference to the entity of the temperature field.
   */
  std::shared_ptr<const Entities::FE_ScalarField<dim>>  temperature;

  /*!
   * @brief A pointer to the body force function.
   */
  TensorFunction<1, dim>  *body_force_ptr;

  /*!
   * @brief A pointer to the gravity unit vector function.
   */
  TensorFunction<1, dim>  *gravity_vector_ptr;

  /*!
   * @brief A pointer to unit vector function of the angular velocity of
   * the rotating frame of reference.
   */
  RMHD::AngularVelocity<dim>   *angular_velocity_vector_ptr;

  /*!
   * @brief A reference to the class controlling the temporal discretization.
   */
  const TimeDiscretization::VSIMEXMethod     &time_stepping;

  /*!
    * @brief A vector containing the \f$ \alpha_0 \f$ of the previous time steps.
    */
  std::array<double, 2> previous_alpha_zeros = {1.0, 1.0};

  /*!
   * @brief A vector containing the sizes of the previous time steps.
   * @details The DiscreteTime class stores only the previous time step.
   * This member stores \f$ n \f$ time steps prior to it, where \f$ n \f$
   * is the order of the scheme.
   */
  std::array<double, 2> previous_step_sizes = {0.0, 0.0};

  /*!
   * @brief System matrix used to solve for the velocity field in the diffusion
   * step.
   *
   * @details This matrix changes in every timestep because the convective term
   * needs to be assembled in every timestep. However, not the entire system
   * matrix is assembled in every timestep but only the part due to the
   * convective term.
   */
  LinearAlgebra::MPI::SparseMatrix  velocity_system_matrix;

  /*!
   * @brief Mass matrix of the velocity.
   *
   * @details This matrix does change not in every timestep. It is stored in
   * memory because otherwise an assembly would be required if the timestep
   * changes.
   */
  LinearAlgebra::MPI::SparseMatrix  velocity_mass_matrix;

  /*!
   * @brief Stiffness matrix of the velocity. Assembly of  the weak of the
   * Laplace operator.
   *
   * @details This matrix does not change in every timestep. It is stored in
   * memory because otherwise an assembly would be required if the timestep
   * changes.
   */
  LinearAlgebra::MPI::SparseMatrix  velocity_laplace_matrix;

  /*!
   * @brief Sum of the mass and the stiffness matrix of the velocity.
   *
   * @details This matrix does not change in every timestep. It is stored in
   * memory because otherwise an assembly would be required if the timestep
   * changes.
   */
  LinearAlgebra::MPI::SparseMatrix  velocity_mass_plus_laplace_matrix;

  /*!
   * @brief Matrix representing the assembly of the skew-symmetric formm of the
   * convective term.
   *
   * @details This matrix changes in every timestep and is therefore also
   * assembled in every timestep.
   */
  LinearAlgebra::MPI::SparseMatrix  velocity_advection_matrix;

  /*!
   * @brief Vector representing the right-hand side of the linear system of the
   * diffusion step.
   */
  LinearAlgebra::MPI::Vector        diffusion_step_rhs;

  /*!
   * @brief Mass matrix of the pressure field.
   */
  LinearAlgebra::MPI::SparseMatrix  projection_mass_matrix;

  /*!
   * @brief Stiffness matrix of the pressure field. Assembly of  the weak of the
   * Laplace operator.
   */
  LinearAlgebra::MPI::SparseMatrix  pressure_laplace_matrix;

  /*!
   * @brief Stiffness matrix of the phi field. Assembly of  the weak of the
   * Laplace operator.
   */
  LinearAlgebra::MPI::SparseMatrix  phi_laplace_matrix;

  /*!
   * @brief Vector representing the right-hand side of the linear system of the
   * projection step.
   */
  LinearAlgebra::MPI::Vector        projection_step_rhs;

  /*!
   * @brief Vector representing the right-hand side of the linear system of the
   * poisson prestep.
   */
  LinearAlgebra::MPI::Vector        poisson_prestep_rhs;

  /*!
   * @brief Vector representing the right-hand side of the projection
   * performed during the pressure-correction step.
   */
  LinearAlgebra::MPI::Vector        correction_step_rhs;

  /*!
   * @brief The preconditioner of the diffusion step.
   */
  std::shared_ptr<LinearAlgebra::PreconditionBase> diffusion_step_preconditioner;

  /*!
   * @brief The preconditioner of the projection step.
   */
  std::shared_ptr<LinearAlgebra::PreconditionBase> projection_step_preconditioner;

  /*!
   * @brief The preconditioner of the poisson prestep.
   */
  std::shared_ptr<LinearAlgebra::PreconditionBase> poisson_prestep_preconditioner;

  /*!
   * @brief The preconditioner of the correction step.
   */
  std::shared_ptr<LinearAlgebra::PreconditionBase> correction_step_preconditioner;

  /*!
   * @brief The norm of the right hand side of the diffusion step.
   * @details Its value is that of the last computed pressure-correction
   * scheme step.
   */
  double                                  norm_diffusion_rhs;

  /*!
   * @brief The norm of the right hand side of the projection step.
   * @details Its value is that of the last computed pressure-correction
   * scheme step.
   */
  double                                  norm_projection_rhs;

  /*!
   * @brief A flag to normalize the pressure field.
   * @details In the case of an unconstrained formulation in the
   * pressure space, i.e. no Dirichlet boundary conditions, this flag
   * has to be set to true in order to constraint the pressure field.
   */
  bool                                  flag_normalize_pressure;

  /*!
   * @brief A flag indicating if the scalar field  \f$ \phi\f$ is to
   * be initiated.
   * @details The initiation is done by the @ref setup_phi method.
   */
  bool                                  flag_setup_phi;

  /*!
   * @brief A flag indicating if the matrices were updated.
   */
  bool                                  flag_matrices_were_updated;

  /*!
   * @brief A method initiating the scalar field  \f$ \phi\f$.
   * @details Extracts its locally owned and relevant degrees of freedom;
   * sets its boundary conditions and applies them to its AffineConstraints
   * instance.
   */
  void setup_phi();

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
   * @brief This method performs the poisson prestep.
   */
  void poisson_prestep();

  /*!
   * @brief This method assembles the linear system of the poisson
   * prestep.
   */
  void assemble_poisson_prestep();

  /*!
   * @brief This method assembles the right-hand side of the poisson
   * prestep.
   */
  void assemble_poisson_prestep_rhs();

  /*!
   * @brief This method solves the linear system of the poisson prestep.
   */
  void solve_poisson_prestep();

  /*!
   * @brief This method performs one complete diffusion step.
   */
  void diffusion_step(const bool reinit_prec);

  /*!
   * @brief This method assembles the system matrix and the right-hand side of
   * the diffusion step.
   *
   * The system matrix \f$\bs{A}^{(\bs{v})}\f$ is constructed from the mass
   * \f$\bs{M}^{(\bs{v})}\f$, the stiffness \f$\bs{K}^{(\bs{v})}\f$ and
   * the advection matrices \f$\bs{C}^{(\bs{v})}\f$ as follows
   *
   * \f[
   * \bs{A} = \frac{\alpha_0}{\Delta t_{n-1}} \bs{M}^{(\bs{v})}+ \frac{1}{\Reynolds}
   * \bs{K}^{(\bs{v})} + \bs{C}^{(\bs{v})} \,.
   * \f]
   */
  void assemble_diffusion_step();

  /*!
   * @brief This method solves the linear system of the diffusion step. Updates
   * the Entities::FE_VectorField::solution vector of the #velocity.
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
   * @brief This method solves the linear system of the projection step. Updates
   * the Entities::FE_ScalarField::solution vector of the pressure correction
   * #phi.
   */
  void solve_projection_step(const bool reinit_prec);

  /*!
   * @brief This method performs the pressure update of the projection step.
   * Updates the Entities::FE_ScalarField::solution vector of the #pressure.
   */
  void pressure_correction(const bool reinit_prec);

  /*!
   * @brief This method assembles the mass \f$\bs{M}^{(\bs{v})}\f$ and the
   * stiffness matrix \f$\bs{K}^{(\bs{v})}\f$ of the velocity field using
   * the WorkStream approach.
   *
   * This method assembles the following weak forms into the two matrices
   * \f$\bs{M}^{(\bs{v})}\f$ and \f$\bs{K}^{(\bs{v})}\f$
   *
   * \f[
   * \begin{equation*}
   * \begin{aligned}
   * \bs{M}^{(\bs{v})}_{ij} &= \int\limits_\Omega \bs{\varphi}_i\cdot\bs{\varphi}_j\dint{V}\,,\\
   * \bs{K}^{(\bs{v})}_{ij} &= \int\limits_\Omega (\nabla\otimes\bs{\varphi}_i)\cdott
   * (\nabla\otimes\bs{\varphi}_j)\dint{V}\,.
   * \end{aligned}
   * \end{equation*}
   * \f]
   */
  void assemble_velocity_matrices();

  /*!
   * @brief This method assembles the local mass and the local stiffness
   * matrices of the velocity field on a single cell.
   */
  void assemble_local_velocity_matrices(
    const typename DoFHandler<dim>::active_cell_iterator                        &cell,
    AssemblyData::NavierStokesProjection::VelocityConstantMatrices::Scratch<dim>&scratch,
    AssemblyData::NavierStokesProjection::VelocityConstantMatrices::Copy        &data);

  /*!
   * @brief This method copies the local mass and the local stiffness matrices
   * of the velocity field on a single cell into the global matrices.
   */
  void copy_local_to_global_velocity_matrices(
    const AssemblyData::NavierStokesProjection::VelocityConstantMatrices::Copy  &data);

  /*!
   * @brief This method assembles the mass \f$\bs{M}^{(p)}\f$ and the
   * stiffness matrices \f$\bs{K}^{(p)}\f$ of the pressure field using the
   * WorkStream approach.
   *
   * This method assembles the following weak forms into the two matrices
   * \f$\bs{M}^{(p)}\f$ and \f$\bs{K}^{(p)}\f$
   *
   * \f[
   * \begin{equation*}
   * \begin{aligned}
   * \bs{M}^{(p)}_{ij} &= \int\limits_\Omega \varphi_i\varphi_j\dint{V}\,,\\
   * \bs{K}^{(p)}_{ij} &= \int\limits_\Omega \nabla\varphi_i\cdot\nabla\varphi_j\dint{V}\,.
   * \end{aligned}
   * \end{equation*}
   * \f]
   */
  void assemble_pressure_matrices();

  /*!
   * @brief This method assembles the local mass and the local stiffness
   * matrices of the velocity field on a single cell.
   */
  void assemble_local_pressure_matrices(
    const typename DoFHandler<dim>::active_cell_iterator                        &cell,
    AssemblyData::NavierStokesProjection::PressureConstantMatrices::Scratch<dim>&scratch,
    AssemblyData::NavierStokesProjection::PressureConstantMatrices::Copy        &data);

  /*!
   * @brief This method copies the local mass and the local stiffness matrices
   * of the pressure field on a single cell into the global matrices.
   */
  void copy_local_to_global_pressure_matrices(
    const AssemblyData::NavierStokesProjection::PressureConstantMatrices::Copy  &data);

  /*!
   * @brief This method assembles the right-hand side of the poisson
   * prestep using the WorkStream approach.
   */
  void assemble_local_poisson_prestep_rhs(
    const typename DoFHandler<dim>::active_cell_iterator                &cell,
    AssemblyData::NavierStokesProjection::PoissonStepRHS::Scratch<dim>  &scratch,
    AssemblyData::NavierStokesProjection::PoissonStepRHS::Copy          &data);

  /*!
   * @brief This method assembles the local right-hand side of the poisson
   * prestep on a single cell.
   */
  void copy_local_to_global_poisson_prestep_rhs(
    const AssemblyData::NavierStokesProjection::PoissonStepRHS::Copy    &data);

  /*!
   * @brief This method assembles the right-hand side of the diffusion step
   * using the WorkStream approach.
   *
   * This method assembles the following weak form into the vector representing
   * the right-hand side \f$\bs{b}\f$
   * \f[
   * \bs{b}_i = -\int\limits_\Omega (\frac{\alpha_1}{\Delta t_{n-1}}
   * \bs{v}^{n-1} + \frac{\alpha_2}{\Delta t_{n-1}} \bs{v}^{n-2}) \cdot
   * \bs{\varphi}_i \dint{V} + \int\limits_\Omega p^\star
   * (\nabla\cdot\bs{\varphi}_i) \dint{V}\,,
   * \f]
   * where \f$\bs{\varphi}_i\f$ is a test function of the velocity space.
   * Furthermore, \f$p^\star\f$ denotes the extrapolated pressure.
   */
  void assemble_diffusion_step_rhs();

  /*!
   * @brief This method assembles the local right-hand side of the diffusion
   * step on a single cell.
   */
  void assemble_local_diffusion_step_rhs(
    const typename DoFHandler<dim>::active_cell_iterator                 &cell,
    AssemblyData::NavierStokesProjection::DiffusionStepRHS::HDScratch<dim> &scratch,
    AssemblyData::NavierStokesProjection::DiffusionStepRHS::Copy         &data);
  void assemble_local_diffusion_step_rhs(
    const typename DoFHandler<dim>::active_cell_iterator                 &cell,
    AssemblyData::NavierStokesProjection::DiffusionStepRHS::HDCScratch<dim> &scratch,
    AssemblyData::NavierStokesProjection::DiffusionStepRHS::Copy         &data);

  /*!
   * @brief This method copies the local right-hand side of the diffusion step
   * into the global vector.
   */
  void copy_local_to_global_diffusion_step_rhs(
    const AssemblyData::NavierStokesProjection::DiffusionStepRHS::Copy  &data);

  /*!
   * @brief This method assembles the right-hand side of the projection step
   * using the WorkStream approach.
   *
   * This method assembles the following weak form into the vector representing
   * the right-hand side \f$\bs{b}\f$
   * \f[
   * \bs{b}_i = -\int\limits_\Omega (\nabla\cdot\bs{v}) \varphi_i \dint{V}\,,
   * \f]
   * where \f$\varphi_i\f$ is a test function of the pressure space.
   */
  void assemble_projection_step_rhs();

  /*!
   * @brief This method assembles the local right-hand side of the projection
   * step on a single cell.
   */
  void assemble_local_projection_step_rhs(
    const typename DoFHandler<dim>::active_cell_iterator                    &cell,
    AssemblyData::NavierStokesProjection::ProjectionStepRHS::Scratch<dim>   &scratch,
    AssemblyData::NavierStokesProjection::ProjectionStepRHS::Copy           &data);

  /*!
   * @brief This method copies the local right-hand side of the projection step
   * on a single cell to the global vector.
   */
  void copy_local_to_global_projection_step_rhs(
    const AssemblyData::NavierStokesProjection::ProjectionStepRHS::Copy &data);

  /*!
   * @brief This method assembles the velocity advection matrix using the
   * WorkStream approach.
   *
   * This method assembles the following skew-symmetric weak form into the
   * advection matrix \f$\bs{C}^{(\bs{v})}\f$
   * \f[
   * \bs{C}^{(\bs{v})}_{ij} = \int\limits_\Omega
   * \bs{v}^\star\cdot(\nabla\otimes\bs{\varphi}_j)\cdot \bs{\varphi}_i \dint{V} +
   * \int\limits_\Omega \tfrac{1}{2}(\nabla\cdot\bs{v}^\star)
   * \bs{\varphi}_j\cdot\bs{\varphi}_i \dint{V}\,,
   * \f]
   * where \f$\varphi_j\f$ and \f$\varphi_i\f$ are the trial and test functions
   * of the velocity space. Furthermore, \f$\bs{v}^\star\f$ denotes
   * the extrapolated velocity.
   */
  void assemble_velocity_advection_matrix();

  /*!
   * @brief This method assembles the local velocity advection matrix on a
   * single cell.
   */
  void assemble_local_velocity_advection_matrix(
    const typename DoFHandler<dim>::active_cell_iterator                &cell,
    AssemblyData::NavierStokesProjection::AdvectionMatrix::Scratch<dim> &scratch,
    AssemblyData::NavierStokesProjection::AdvectionMatrix::Copy         &data);

  /*!
   * @brief This method copies the local velocity advection matrix into the
   * global matrix.
   */
  void copy_local_to_global_velocity_advection_matrix(
    const AssemblyData::NavierStokesProjection::AdvectionMatrix::Copy   &data);

};

// inline functions
template <int dim>
inline double NavierStokesProjection<dim>::get_diffusion_step_rhs_norm() const
{
  return (norm_diffusion_rhs);
}

// inline functions
template <int dim>
inline double NavierStokesProjection<dim>::get_projection_step_rhs_norm() const
{
  return (norm_projection_rhs);
}

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_NAVIER_STOKES_PROJECTION_H_ */
