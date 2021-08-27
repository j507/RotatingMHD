/*!
 * @file DFG
 *
 * @brief The source file for solving the DFG benchmark.
 *
 */
#include <rotatingMHD/benchmark_data.h>
#include <rotatingMHD/navier_stokes_projection.h>
#include <rotatingMHD/problem_class.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/time_discretization.h>

#include <deal.II/fe/mapping_q.h>
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/utilities.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <rotatingMHD/finite_element_field.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <string>

namespace DFGBenchmark
{

using namespace dealii;
using namespace RMHD;

namespace EquationData
{

/*!
 * @class VelocityInflowBoundaryCondition
 *
 * @brief The velocity profile at the inlet of the channel.
 *
 * @details The velocity profile is given by the following function
 * \f[
 * \bs{v}(y)= v_x(y) \ex= v_0 \left(\frac{2}{H}\right)^2 y (H-y) \ex \,,
 * \f]
 *
 * where \f$ H \f$ denotes the height of the channel and \f$ v_0 \f$ is the
 * maximum velocity in the middle of the channel. Typically, the maximum velocity is
 * chosen as \f$ v_0=\frac{3}{2} \f$ such that the mean velocity is unity.
 *
 */
template <int dim>
class VelocityInflowBoundaryCondition : public Function<dim>
{
public:
  /*!
   * Default constructor.
   */
  VelocityInflowBoundaryCondition(const double time = 0,
                                  const double maximum_velocity = 1.5,
                                  const double height = 4.1);

  /*!
   * Overloading method evaluating the function.
   */
  virtual void vector_value(const Point<dim>  &p,
                            Vector<double>    &values) const override;

private:
  /*!
   * The maximum velocity at the middle of the channel.
   */
  const double maximum_velocity;

  /*!
   * The height of the channel.
   */
  const double height;
};

template <int dim>
VelocityInflowBoundaryCondition<dim>::VelocityInflowBoundaryCondition
(const double time,
 const double maximum_velocity,
 const double height)
:
Function<dim>(dim, time),
maximum_velocity(maximum_velocity),
height(height)
{}



template <int dim>
void VelocityInflowBoundaryCondition<dim>::vector_value
(const Point<dim>  &point,
 Vector<double>    &values) const
{
  values[0] = 4.0 * maximum_velocity * point(1) * ( height - point(1) )
      / ( height * height );

  for (unsigned d=1; d<dim; ++d)
    values[d] = 0.0;
}

template<int dim>
using VelocityInitialCondition = Functions::ZeroFunction<dim>;

template<int dim>
using PressureInitialCondition = Functions::ZeroFunction<dim>;

}  // namespace EquationData

/*!
 * @class DFG
 *
 * @brief This class solves the DFG benchmark 2D-2 of the flow around cylinder
 * which is placed in a channel. This is the time-periodic case with a
 * Reynolds number of \f$ \Reynolds=100 \f$.
 *
 * @details The DFG benchmark considers the flow of fluid inside a channel with
 * a cylinder as an obstacle (See Fig. ).
 *
 * @todo A figure is missing here.
 *
 *
 * **Defintion of the benchmark**
 *
 * The field equations of the problem are given by the incompressible
 * Navier-Stokes equations
 *
 * \f[
 * \begin{equation*}
 * \begin{aligned}
 *    \pd{\bs{v}}{t} + \bs{v} \cdot
 *    (\nabla \otimes \bs{v})&=  - \dfrac{1}{\rho_0}\nabla p + \nu
 *    \Delta \bs{v} + \bs{b}, &
 *    \forall (\bs{x}, t) \in \Omega \times \left[0, T \right]\,, \\
 *    \nabla \cdot \bs{v}&= 0, &
 *    \forall (\bs{x}, t) \in \Omega \times  \left[0, T \right]
 * \end{aligned}
 * \end{equation*}
 * \f]
 *
 * with the following boundary conditions
 *
 * \f[
 * \begin{equation*}
 * \begin{aligned}
 *    \bs{v} &= \bs{v}_\text{in}\,,  &
 *    \forall(\bs{x}, t) &\in \Gamma_0 \times [0, T]\,, \\
 *    \bs{v} &= \bs{0}\,, &
 *    \forall(\bs{x}, t) &\in \Gamma_2 \times [0,T]\,, \\
 *    \bs{v} &= \bs{0}\,, &
 *    \forall(\bs{x}, t) &\in \Gamma_3 \times [0, T]\,, \\
 *    \bs{n} \cdot [-p\bs{1} + \mu \nabla \otimes \bs{v}] &=0\,, &
 *    \forall(\bs{x}, t) &\in \Gamma_1 \times [0, T]\,.
 * \end{aligned}
 * \end{equation*}
 * \f]
 *
 * The velocity profile at the inlet of the channel is given by a quadratic
 * function
 *
 * \f[
 * \begin{equation*}
 *    \bs{v}_\text{in} = v_0\left(\frac{2}{H}\right)^2 y(H-y) \ex \quad
 *    \text{with} \quad v_0 =\frac{3}{2}\frac{\mathrm{m}}{\mathrm{s}}\,.
 * \end{equation*}
 * \f]
 *
 * Furthermore, the density is given by \f$\rho_0 = 1\,\mathrm{kg}\,\mathrm{m}^{-3}\f$
 * and the kinematic viscosity by \f$\nu = 0{.}001\,\mathrm{m}^2\,\mathrm{s}^{-1}\f$.
 * In order to compute the Reynolds number of the problem, a reference velocity
 * and a reference length need to be defined. The average of the velocity
 * at the inlet is chosen as the reference velocity \f$ v_\mathrm{ref} \f$,
 * *i. e.*,
 *
 * \f[
 * \begin{equation*}
 *    v_\mathrm{ref} = \frac{1}{H}\int\limits_0^H \bs{v}_\text{in}(y)\cdot\ex
 *    \dint{y}=\frac{2}{3}v_0=1\frac{\mathrm{m}}{\mathrm{s}}
 * \end{equation*}
 * \f]
 *
 *
 * Moreover, the diameter of the cylinder is chosen as the reference length,
 * *i. e.*, \f$ \ell_\mathrm{ref}=D=0{.}1\,\mathrm{m}\f$. These two choice yield
 * the following definition and value of the Reynolds number of the problem
 * \f[
 * \begin{equation*}
 *    \Reynolds = \frac{v_\mathrm{ref} D}{\nu} = 100\,.
 * \end{equation*}
 * \f]
 *
 * **Benchmark requests**
 *
 * The DFG benchmark requires to specify the pressure difference between two
 * points located at the front and the rear of the cylinder, respectively. These
 * points are located at the following positions
 * \f$ \bs{x}_\text{front} = 0{.}15\,\mathrm{m}\, \ex + 0.20\,\mathrm{m}\, \ey \f$
 * and
 * \f$ \bs{x}_\text{rear} = 0{.}25\,\mathrm{m}\, \ex + 0.20\,\mathrm{m}\, \ey \f$,
 * respectively. Thus, the pressure difference \f$ \Delta p\f$ is computed
 * according to
 *
 * \f[
 * \begin{equation*}
 *    \Delta p = p|_{\bs{x}_\text{front}} - p|_{\bs{x}_\text{rear}}\,.
 * \end{equation*}
 * \f]
 *
 * Additionally, the DFG benchmark request the value of the lift and drag
 * coefficients, which are defined as
 *
 * \f[
 * \begin{equation*}
 *    c_\text{drag} = \dfrac{2}{\rho v_\mathrm{ref}^2 D} F_\text{drag}\,,\qquad
 *    c_\text{lift} = \dfrac{2}{\rho v_\mathrm{ref}^2 D} F_\text{lift}\,.
 * \end{equation*}
 * \f]
 *
 * Here, \f$ F_\text{drag} \f$ and \f$ F_\text{lift} \f$ refer to the horizontal
 * and vertical components of the resulting force acting on the cylinder. The
 * resulting force is computed according to
 *
 * \f[
 * \begin{equation*}
 *	 \bs{F} = F_\text{drag}\ex +  F_\text{lift}\ey
 *	 = \int\limits_{\Gamma_3} \bs{\sigma}\cdot \bs{n} \dint{A}
 *   = \int\limits_{\Gamma_3} \Big(-p \bs{1} + \mu \big(\nabla \otimes \bs{v}
 *   + \bs{v} \otimes \nabla \big) \Big) \cdot \bs{n} \dint{A}\,.
 * \end{equation*}
 * \f]
 *
 * Another quantity requested is the Strouhal number related to the periodic
 * oscillation of the flow. This dimesionless number is defined as
 *
 * \f[
 * \begin{equation*}
 *   \Strouhal = \dfrac{f D}{v_\mathrm{ref}}\,,
 * \end{equation*}
 * \f]
 *
 * where \f$ f \f$ denotes the frequency of the oscillation. This frequency is
 * the reciprocal of the time period \f$ T \f$ of the oscillation, *i. e.*,
 * \f$ f = 1/ T \f$. The period of the oscillation is computed by considering
 * two consecutive maximum values of the lift and drag coefficients. Taking the
 * time difference between these yields the time period of the oscillation.
 * Additionally the minimum, the average and the amplitude of the lift and drag
 * coefficients are also computed.
 *
 * **Dimensionless formulation of the benchmark**
 *
 * The @ref NavierStokesProjection class is based on the dimensionless
 * form of the Navier-Stokes equations. Therefore, the benchmark has to
 * be reformulated into its dimensionless form as follows
 *
 * \f[
 * \begin{equation*}
 * 	\begin{aligned}
 * 		\pd{\tilde{\bs{v}}}{\tilde{t}} +
 *    \tilde{\bs{v}} \cdot (\tilde{\nabla} \otimes \tilde{\bs{v}})
 *    &= -\tilde{\nabla}\tilde{p} +
 *    \frac{1}{\Reynolds} \nabla^2 \tilde{\bs{v}} + \tilde{\bs{b}}\, &
 *    \forall (\tilde{\bs{x}}, \tilde{t})
 *    \in \tilde{\Omega} \times [0, \tilde{T} ] \\
 * 		\tilde{\nabla} \cdot \tilde{\bs{v}}&= 0, &
 * 		\forall (\tilde{\bs{x}}, \tilde{t}) \in\tilde{\Omega} \times [0, \tilde{T} ]
 * 	\end{aligned}
 * \end{equation*}
 * \f]
 *
 * with the following boundary conditions
 *
 * \f[
 * \begin{equation*}
 * \begin{aligned}
 *    \tilde{\bs{v}} &= \tilde{\bs{v}}_\text{in}\,, &
 *    \forall(\bs{x}, t) &\in \Gamma_0 \times [0, T] \\
 *    \tilde{\bs{v}} &= \bs{0}\,, &
 * 		\forall(\bs{x}, t) &\in \Gamma_2 \times [0,T]\\
 * 	  \tilde{\bs{v}} &= \bs{0}, &
 * 		\forall(\bs{x}, t) &\in \Gamma_3 \times [0, T] \\
 * 	  \bs{n}\cdot [-\tilde{p}\bs{1} + \tfrac{1}{\Reynolds}\tilde{\nabla}
 *    \otimes \tilde{\bs{v}}] &=0, &
 *    \forall(\bs{x}, t) &\in \Gamma_1 \times [0, T] \\
 * 	\end{aligned}
 * \end{equation*}
 * \f]
 *
 * The dimensionless velocity profile at the inlet of channel is then given by
 *
 * \f[
 * \begin{equation*}
 *    \tilde{\bs{v}}_\text{in} = \tilde{v}_0 \left(\frac{2}{\tilde{H}}\right)^2
 *    \tilde{y}(\tilde{H}-\tilde{y}) \ex \,, \quad
 *    \text{with}\quad
 * 	  \tilde{v}_0=\frac{3}{2}\,.
 * \end{equation*}
 * \f]
 *
 * The benchmark requests are computed in dimensionless form. In order to compare
 * with those of the DFG benchmark, they need to be scaled back. The reference
 * pressure is given by the dynamic pressure, *i. e.*,
 * \f$ p_\mathrm{ref}=\rho_0 v_\mathrm{ref}^2
 * = 1\,\mathrm{kg}\,\mathrm{m}^{-1}\,\mathrm{s}^{-2} \f$. Thus, the numerical
 * values of the dimensionless and dimensioned pressure are equal. Of course,
 * the same also applies for the pressure differences requested by the DFG
 * benchmark. They are interchangeable.
 *
 * The reference value of the force is \f$ F_\mathrm{ref}=\rho_0
 * v_\mathrm{ref}^2 D^2\f$. Thus the dimensionless resulting force is given by
 *
 * \f[
 * \begin{equation*}
 *    \tilde{\bs{F}} = \frac{1}{\rho_0 v_\mathrm{ref}^2 D^2}
 * 	  \int_{\tilde{\Gamma}_3} \Big(-p\bs{1}
 *    + \nu \big( \nabla \otimes \bs{v} + \bs{v} \otimes \nabla\big)
 *    \Big) \cdot \bs{n} \dint{A}
 *    = \int_{\tilde{\Gamma}_3} \Big(-\tilde{p}\bs{1}
 *    + \tfrac{1}{\Reynolds} \big( \tilde{\nabla} \otimes \tilde{\bs{v}} +
 *    \tilde{\bs{v}} \otimes \tilde{\nabla} \big) \Big) \cdot \bs{n} \dint{\tilde{A}}\,.
 * \end{equation*}
 * \f]
 *
 * With this formula for the dimensionless force, it is easy to see that the
 * coefficients are given by
 *
 * \f[
 * \begin{equation*}
 *    c_\text{drag} = 2 \tilde{F}_\text{drag}\,,\qquad
 *    c_\text{lift} = 2 \tilde{F}_\text{lift}\,.
 * \end{equation*}
 * \f]
 *
 * The frequency is scaled back with the reference time \f$ t_\mathrm{ref} \f$.
 * Hence,
 *
 * \f[
 * \begin{equation*}
 * 	  f = \frac{1}{t_\mathrm{ref}} \tilde{f}
 * \end{equation*}
 * \f]
 *
 * Since the reference time is given by \f$ t_\mathrm{ref} = D / v_\mathrm{ref}\f$,
 * the Strouhal number may also expressed in terms of the dimesionless frequency
 * as follows
 *
 * \f[
 * \begin{equation*}
 *    \Strouhal = \tilde{f}\,.s
 * \end{equation*}
 * \f]
 *
 */
template <int dim>
class DFG : public Problem<dim>
{
public:
  DFG(const RunTimeParameters::ProblemParameters &parameters);

  void run();

private:

  const RunTimeParameters::ProblemParameters &parameters;

  const types::boundary_id  channel_wall_bndry_id{3};
  const types::boundary_id  cylinder_bndry_id{2};
  const types::boundary_id  channel_inlet_bndry_id{0};
  const types::boundary_id  channel_outlet_bndry_id{1};

  std::shared_ptr<Entities::FE_VectorField<dim>>  velocity;

  std::shared_ptr<Entities::FE_ScalarField<dim>>  pressure;

  TimeDiscretization::VSIMEXMethod              time_stepping;

  NavierStokesProjection<dim>                   navier_stokes;

  BenchmarkData::DFGBechmarkRequests<dim>        benchmark_requests;

  EquationData::VelocityInitialCondition<dim>   velocity_initial_condition;

  EquationData::PressureInitialCondition<dim>   pressure_initial_condition;

  double                                        cfl_number;

  void make_grid();

  void setup_dofs();

  void setup_constraints();

  void initialize();

  void postprocessing();

  void output();

  void update_solution_vectors();
};

template <int dim>
DFG<dim>::DFG(const RunTimeParameters::ProblemParameters &parameters)
:
parameters(parameters),
Problem<dim>(parameters),
velocity(std::make_shared<Entities::FE_VectorField<dim>>
         (parameters.fe_degree_velocity,
          this->triangulation,
          "Velocity")),
pressure(std::make_shared<Entities::FE_ScalarField<dim>>
         (parameters.fe_degree_pressure,
          this->triangulation,
          "Pressure")),
time_stepping(parameters.time_discretization_parameters),
navier_stokes(parameters.navier_stokes_parameters,
              time_stepping,
              velocity,
              pressure,
              this->mapping,
              this->pcout,
              this->computing_timer),
benchmark_requests(parameters.Re, cylinder_bndry_id),
velocity_initial_condition(dim),
pressure_initial_condition()
{
  *this->pcout << parameters << std::endl << std::endl;
  make_grid();
  setup_dofs();
  setup_constraints();
  velocity->setup_vectors();
  pressure->setup_vectors();
  initialize();
//  this->container.add_entity(*velocity);
//  this->container.add_entity(*pressure, false);
//  this->container.add_entity(*navier_stokes.phi, false);
}


template <>
void DFG<2>::make_grid()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Triangulation");

  // Create serial triangulation
  Triangulation<2>  tria;
  GridGenerator::channel_with_cylinder(tria, 0.03, 2, 2.0, true);

  // Scale the geometry
  GridTools::scale(10.0, tria);

  // Reset all manifolds
  tria.reset_all_manifolds();

  // Copy triangulation
  this->triangulation.copy_triangulation(tria);

  // Check that manifold ids are correct
  const types::manifold_id polar_manifold_id = 0;
  const types::manifold_id tfi_manifold_id   = 1;
  const std::vector<types::manifold_id> manifold_ids = this->triangulation.get_manifold_ids();
  AssertThrow(std::find(manifold_ids.begin(), manifold_ids.end(), polar_manifold_id) != manifold_ids.end(),
              ExcInternalError());
  AssertThrow(std::find(manifold_ids.begin(), manifold_ids.end(), tfi_manifold_id) != manifold_ids.end(),
              ExcInternalError());

  // Attach new manifolds
  PolarManifold<2>  polar_manifold(Point<2>(2.0, 2.0));
  this->triangulation.set_manifold(0, polar_manifold);

  TransfiniteInterpolationManifold<2> inner_manifold;
  inner_manifold.initialize(this->triangulation);
  this->triangulation.set_manifold(1, inner_manifold);

  // Perform global refinements
  this->triangulation.refine_global(parameters.spatial_discretization_parameters.n_initial_global_refinements);

  // Perform one level local refinement of the cells located at the boundary
  for (unsigned int i=0;
       i<parameters.spatial_discretization_parameters.n_initial_boundary_refinements;
       ++i)
  {
    for (auto &cell: this->triangulation.active_cell_iterators())
      if (cell->at_boundary() && cell->is_locally_owned())
        cell->set_refine_flag();
    this->triangulation.execute_coarsening_and_refinement();
  }

  *this->pcout << "Number of active cells                = "
               << this->triangulation.n_active_cells() << std::endl;
}



template <int dim>
void DFG<dim>::setup_dofs()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - DoFs");

  velocity->setup_dofs();
  pressure->setup_dofs();

  *this->pcout << "Number of velocity degrees of freedom = "
               << velocity->n_dofs()
               << std::endl
               << "Number of pressure degrees of freedom = "
               << pressure->n_dofs()
               << std::endl
               << "Number of total degrees of freedom    = "
               << (pressure->n_dofs() +
                  velocity->n_dofs())
               << std::endl << std::endl;
}



template <int dim>
void DFG<dim>::setup_constraints()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Boundary conditions");

  velocity->clear_boundary_conditions();
  pressure->clear_boundary_conditions();

  velocity->setup_boundary_conditions();
  pressure->setup_boundary_conditions();

  velocity->set_dirichlet_boundary_condition
  (channel_inlet_bndry_id,
   std::make_shared<EquationData::VelocityInflowBoundaryCondition<dim>>
   (time_stepping.get_start_time())
  );

  velocity->set_dirichlet_boundary_condition(channel_wall_bndry_id);
  velocity->set_dirichlet_boundary_condition(cylinder_bndry_id);

  pressure->set_dirichlet_boundary_condition(channel_outlet_bndry_id);

  velocity->close_boundary_conditions();
  pressure->close_boundary_conditions();

  velocity->apply_boundary_conditions();
  pressure->apply_boundary_conditions();
}

template <int dim>
void DFG<dim>::initialize()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Initial conditions");

  this->set_initial_conditions(velocity,
                               velocity_initial_condition,
                               time_stepping);
  this->set_initial_conditions(pressure,
                               pressure_initial_condition,
                               time_stepping);
}

template <int dim>
void DFG<dim>::postprocessing()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Postprocessing");

  benchmark_requests.update(this->time_stepping.get_current_time(),
                           this->time_stepping.get_step_number(),
                           *velocity,
                           *pressure);

  *this->pcout << benchmark_requests << std::endl;
}

template <int dim>
void DFG<dim>::output()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Graphical output");

  std::vector<std::string> names(dim, "velocity");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);

  DataOut<dim>        data_out;

  data_out.add_data_vector(velocity->get_dof_handler(),
                           velocity->solution,
                           names,
                           component_interpretation);

  data_out.add_data_vector(pressure->get_dof_handler(),
                           pressure->solution,
                           "Pressure");

  data_out.build_patches(velocity->fe_degree());

  static int out_index = 0;

  data_out.write_vtu_with_pvtu_record(parameters.output_directory,
                                      "solution",
                                      out_index,
                                      this->mpi_communicator,
                                      5);

  out_index++;
}

template <int dim>
void DFG<dim>::update_solution_vectors()
{
  velocity->update_solution_vectors();
  pressure->update_solution_vectors();
}

template <int dim>
void DFG<dim>::run()
{
  const unsigned int n_steps = parameters.time_discretization_parameters.n_maximum_steps;

  *this->pcout << "Solving until t = 350..." << std::endl;

  *this->pcout << static_cast<TimeDiscretization::DiscreteTime &>(time_stepping)
               << std::endl;
  while (time_stepping.get_current_time() <= 350.0 &&
         (n_steps > 0? time_stepping.get_step_number() < n_steps: true))
  {
    // The VSIMEXMethod instance starts each loop at t^{k-1}

    // Compute CFL number
    cfl_number = navier_stokes.get_cfl_number();

    // Updates the time step, i.e sets the value of t^{k}
    time_stepping.set_desired_next_step_size(
      this->compute_next_time_step(time_stepping, cfl_number));

    // Updates the coefficients to their k-th value
    time_stepping.update_coefficients();

    // Solves the system, i.e. computes the fields at t^{k}
    navier_stokes.solve();

    // Advances the VSIMEXMethod instance to t^{k}
    update_solution_vectors();
    time_stepping.advance_time();
    *this->pcout << static_cast<TimeDiscretization::DiscreteTime &>(time_stepping)
                 << std::endl;

    // Snapshot stage, all time calls should be done with get_current_time()
    if ((time_stepping.get_step_number() %
         parameters.terminal_output_frequency == 0) ||
        (time_stepping.get_next_time() ==
          time_stepping.get_end_time()))
      postprocessing();
  }
  const unsigned int n_remaining_steps{n_steps - time_stepping.get_step_number()};

  *this->pcout << "Restarting..." << std::endl;
  time_stepping.restart();

  velocity->old_old_solution = velocity->solution;
  navier_stokes.clear();

  *this->pcout << "Solving until t = "
               << time_stepping.get_end_time()
               << "..." << std::endl;

  *this->pcout << static_cast<TimeDiscretization::DiscreteTime &>(time_stepping)
               << std::endl;

  while (time_stepping.get_current_time() < time_stepping.get_end_time() &&
         time_stepping.get_step_number() < n_remaining_steps)
  {
    // The VSIMEXMethod instance starts each loop at t^{k-1}

    // Compute CFL number
    cfl_number = navier_stokes.get_cfl_number();

    // Updates the time step, i.e sets the value of t^{k}
    time_stepping.set_desired_next_step_size(
      this->compute_next_time_step(time_stepping, cfl_number));

    // Updates the coefficients to their k-th value
    time_stepping.update_coefficients();

    // Solves the system, i.e. computes the fields at t^{k}
    navier_stokes.solve();

    // Advances the VSIMEXMethod instance to t^{k}
    update_solution_vectors();
    time_stepping.advance_time();
    *this->pcout << static_cast<TimeDiscretization::DiscreteTime &>(time_stepping)
                 << std::endl;

    // Snapshot stage, all time calls should be done with get_current_time()
    if ((time_stepping.get_step_number() %
        parameters.terminal_output_frequency == 0) ||
        (time_stepping.get_next_time() ==
          time_stepping.get_end_time()))
      postprocessing();

    if ((time_stepping.get_step_number() %
         parameters.graphical_output_frequency == 0) ||
        (time_stepping.get_next_time() ==
          time_stepping.get_end_time()))
      output();
  }

  if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
  {
    if (!std::filesystem::exists(parameters.output_directory))
    {
      try
      {
        std::filesystem::create_directories(parameters.output_directory);
      }
      catch (std::exception &exc)
      {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::cerr << "Exception in the creation of the output directory: "
                  << std::endl
                  << exc.what() << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::abort();
      }
      catch (...)
      {
        std::cerr << std::endl << std::endl
                  << "----------------------------------------------------"
                    << std::endl;
        std::cerr << "Unknown exception in the creation of the output directory!"
                  << std::endl
                  << "Aborting!" << std::endl
                  << "----------------------------------------------------"
                  << std::endl;
        std::abort();
      }
    }

    const std::filesystem::path path{parameters.output_directory};

    std::filesystem::path filename = path / "benchmark_data.txt";

    try
    {
      std::ofstream fstream(filename.string());
      benchmark_requests.write_text(fstream);
    }
    catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception in the creation of the output file: "
                << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::abort();
    }
    catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                  << std::endl;
      std::cerr << "Unknown exception in the creation of the output file!"
                << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::abort();
    }
  }
}

} // namespace DFGBenchmark

int main(int argc, char *argv[])
{
  try
  {
      using namespace dealii;
      using namespace DFGBenchmark;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      std::string parameter_filename;
      if (argc >= 2)
        parameter_filename = argv[1];
      else
        parameter_filename = "DFG.prm";

      RunTimeParameters::ProblemParameters parameter_set(parameter_filename);

      DFG<2> simulation(parameter_set);

      simulation.run();
  }
  catch (std::exception &exc)
  {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
  }
  catch (...)
  {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
  }
  return 0;
}
