/*!
 * @file Couette
 *
 * @brief The source file solving the Couette flow problem.
 *
 */
#include <rotatingMHD/convergence_test.h>
#include <rotatingMHD/finite_element_field.h>
#include <rotatingMHD/navier_stokes_projection.h>
#include <rotatingMHD/problem_class.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/time_discretization.h>
#include <rotatingMHD/vector_tools.h>

#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/data_out.h>

#include <filesystem>
#include <memory>

namespace Couette
{

using namespace dealii;
using namespace RMHD;

namespace EquationData
{

/*!
 * @class VelocityExactSolution
 *
 * @brief The velocity's exact solution for the Couette flow, where the
 * displacement of the top plate is driven by a traction vector.
 *
 * @details It is given by
 * \f[ \bs{u} = t_0 \Reynolds \dfrac{y}{H} \bs{e}_\mathrm{x}, \f]
 * where \f$ t_0 \f$, \f$ \Reynolds \f$, \f$ H \f$, \f$ y \f$ and
 * \f$ \bs{e}_\mathrm{x} \f$ are the traction vector magnitude, the
 * Reynolds number, the height of the channel, the \f$ y \f$-component
 * of the position vector and the unit vector in the \f$ x \f$-direction.
 *
 */
template <int dim>
class VelocityExactSolution : public Function<dim>
{
public:
  VelocityExactSolution(const double t_0,
                        const double Re,
                        const double H = 1.0,
                        const double time = 0.0);

  virtual void vector_value(
    const Point<dim>  &p,
    Vector<double>    &values) const override;

  virtual Tensor<1, dim> gradient(
    const Point<dim> &point,
    const unsigned int component) const override;

private:
  /*!
   * @brief The magnitude of the applied traction vector.
   */
  const double traction_magnitude;

  /*!
   * @brief The Reynolds number.
   */
  const double Re;

  /*!
   * @brief The height of the channel.
   */
  const double height;
};


template <int dim>
VelocityExactSolution<dim>::VelocityExactSolution
(const double t_0,
 const double Re,
 const double H,
 const double time)
:
Function<dim>(dim, time),
traction_magnitude(t_0),
Re(Re),
height(H)
{}



template <int dim>
void VelocityExactSolution<dim>::vector_value
(const Point<dim>  &point,
 Vector<double>    &values) const
{
  values[0] = traction_magnitude * Re * point[1] / height;
  values[1] = 0.0;
}



template <int dim>
Tensor<1, dim> VelocityExactSolution<dim>::gradient
(const Point<dim>  &/*point*/,
 const unsigned int component) const
{
  Tensor<1, dim>  gradient;

  // The gradient has to match that of dealii, i.e. from the right.
  if (component == 0)
  {
    gradient[0] = 0.0;
    gradient[1] = traction_magnitude * Re / height;
  }
  else if (component == 1)
  {
    gradient[0] = 0.0;
    gradient[1] = 0.0;
  }

  return gradient;
}



/*!
 * @class TractionVector
 *
 * @brief The traction vector applied on the top plate of the Couette flow.
 *
 * @details It is given by \f[ \bs{t} = t_0 \bs{e}_\mathrm{x}, \f]
 * where \f$ t_0 \f$ and \f$ \bs{e}_\mathrm{x} \f$ are the
 * magnitude of the traction and the unit vector in the \f$ x \f$-direction.
 *
 */
template <int dim>
class TractionVector : public TensorFunction<1,dim>
{
public:
  TractionVector(const double t_0, const double time = 0.);

  virtual Tensor<1, dim> value(const Point<dim> &point) const override;

private:
  /*!
   * @brief The magnitude of the applied traction vector.
   */
  const double t_0;
};



template <int dim>
TractionVector<dim>::TractionVector
(const double t_0,
 const double time)
:
TensorFunction<1, dim>(time),
t_0(t_0)
{}



template <int dim>
Tensor<1, dim> TractionVector<dim>::value(const Point<dim> &/*point*/) const
{
  Tensor<1, dim> traction_vector;

  traction_vector[0] = t_0;
  traction_vector[1] = 0.0;

  return traction_vector;
}


}  // namespace EquationData


/*!
 * @class CouetteFlowProblem
 *
 * @brief This class solves the Couette flow problem.
 *
 * @details The Couette flow was chosen as a test problem to verify the
 * correct implementation of the Neumann boundary conditions in the
 * @ref NavierStokesProjection solver. The problem considers stationary
 * laminar flow of fluid between two infinitely long horiozontal plates,
 * where the lower plate is fixed and the upper one is being moved.
 * The problem is governed by the equation
 * \f[
 * \ddsqr{u_\mathrm{x}}{y} = 0, \quad \forall \bs{x} \in \Omega
 * \f]
 * which can be easily obtained from the momentum equation by
 * considering an isotropic and homogeneous fluid, an unidirectional and
 * stationary flow in the \f$ x\f$-direction, neglecting body forces and
 * a vanishing horizontal pressure gradient. An unique solution is
 * obtained by considering a no-slip boundary condition at the lower
 * plate and a traction vector
 * \f$ \bs{t} = t_0 \bs{e}_\textrm{x} \f$ applied to the upper plate.
 * This leads to
 * \f[
 * \bs{u} = t_0 \Reynolds \dfrac{y}{H} \bs{e}_\textrm{x},
 * \f]
 * where \f$ \Reynolds \f$ and \f$ H \f$ are the Reynolds
 * number and the height of the channel. The stationary solution is
 * obtained at around \f$ t ~ H^2 \Reynolds \f$.
 *
 * @note Periodic boundary conditions are implemented in order to
 * simulate an infinitely long channel.
 *
 */
template <int dim>
class CouetteFlowProblem : public Problem<dim>
{
public:

  CouetteFlowProblem(const RunTimeParameters::ProblemParameters &parameters);

  void run();

private:

  const RunTimeParameters::ProblemParameters   &parameters;

  std::shared_ptr<Entities::FE_VectorField<dim>>  velocity;

  std::shared_ptr<Entities::FE_ScalarField<dim>>  pressure;

  TimeDiscretization::VSIMEXMethod              time_stepping;

  NavierStokesProjection<dim>                   navier_stokes;

  ConvergenceTest::ConvergenceResults           velocity_convergence_table;

  const double  traction_magnitude;

  const double  reynolds_number;

  double        cfl_number;

  std::shared_ptr<EquationData::TractionVector<dim>>
                                                traction_vector;

  void make_grid(const unsigned int &n_global_refinements);

  void setup_dofs();

  void setup_constraints();

  void initialize();

  void postprocessing();

  void output();

  void update_entities();

  void solve(const unsigned int level);
};

template <int dim>
CouetteFlowProblem<dim>::CouetteFlowProblem(const RunTimeParameters::ProblemParameters &parameters)
:
Problem<dim>(parameters),
parameters(parameters),
velocity(std::make_shared<Entities::FE_VectorField<dim>>(parameters.fe_degree_velocity,
                                                         this->triangulation,
                                                         "Velocity")),
pressure(std::make_shared<Entities::FE_ScalarField<dim>>(parameters.fe_degree_pressure,
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
velocity_convergence_table(parameters.convergence_test_parameters.type),
traction_magnitude(1.0),
reynolds_number(parameters.Re),
cfl_number(0.0),
traction_vector(std::make_shared<EquationData::TractionVector<dim>>(traction_magnitude))
{
  // The Couette flow is a 2-dimensional problem.
  AssertDimension(dim, 2);

  *this->pcout << parameters << std::endl << std::endl;
}

template <int dim>
void CouetteFlowProblem<dim>::
make_grid(const unsigned int &n_global_refinements)
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Triangulation");

  // The solution, being independent of the x-component of the position
  // vector, allows a coarser mesh in the x-direction.
  std::vector<unsigned int> repetitions;
  repetitions.emplace_back(1);
  repetitions.emplace_back(3);

  GridGenerator::subdivided_hyper_rectangle(this->triangulation,
                                 repetitions,
                                 Point<2>(0., 0.),
                                 Point<2>(1., 1.),
                                 true);

  // The infinite domain is implemented with periodic boundary conditions,
  // this periodicity has to be first implemented in the triangulation.
  std::vector<GridTools::PeriodicFacePair<
    typename parallel::distributed::Triangulation<dim>::cell_iterator>>
    periodicity_vector;

  GridTools::collect_periodic_faces(this->triangulation,
                                    0,
                                    1,
                                    0,
                                    periodicity_vector);

  this->triangulation.add_periodicity(periodicity_vector);

  this->triangulation.refine_global(n_global_refinements);
}

template <int dim>
void CouetteFlowProblem<dim>::setup_dofs()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - DoFs");

  velocity->setup_dofs();
  pressure->setup_dofs();

  *this->pcout  << "  Number of active cells                = "
                << this->triangulation.n_global_active_cells() << std::endl;
  *this->pcout  << "  Number of velocity degrees of freedom = "
                << velocity->n_dofs()
                << std::endl
                << "  Number of pressure degrees of freedom = "
                << pressure->n_dofs()
                << std::endl
                << "  Number of total degrees of freedom    = "
                << (pressure->n_dofs() + velocity->n_dofs())
                << std::endl;
}

template <int dim>
void CouetteFlowProblem<dim>::setup_constraints()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Boundary conditions");

  velocity->clear_boundary_conditions();
  pressure->clear_boundary_conditions();

  velocity->setup_boundary_conditions();
  pressure->setup_boundary_conditions();

  // The domain represents an infinite channel. In order to obtain the
  // analytical solution, periodic boundary conditions need to be
  // implemented.
  velocity->set_periodic_boundary_condition(0, 1, 0);
  pressure->set_periodic_boundary_condition(0, 1, 0);
  // No-slip boundary conditions on the lower plate
  velocity->set_dirichlet_boundary_condition(2);
  // The upper plate is displaced by a traction vector
  velocity->set_neumann_boundary_condition(3, traction_vector);

  velocity->close_boundary_conditions();
  pressure->close_boundary_conditions();

  velocity->apply_boundary_conditions();
  pressure->apply_boundary_conditions();
}

template <int dim>
void CouetteFlowProblem<dim>::initialize()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Initial conditions");

  // The initial conditions are zero and the velocity's Dirichlet
  // boundary conditions are homogenous. This allows to just
  // set the solution vectors to zero instead of projecting a function
  // or distributing constraints.
  velocity->set_solution_vectors_to_zero();
  pressure->set_solution_vectors_to_zero();
}

template <int dim>
void CouetteFlowProblem<dim>::postprocessing()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Postprocessing");

  std::cout.precision(1);
  *this->pcout << static_cast<TimeDiscretization::DiscreteTime &>(time_stepping)
               << " Norms = ("
               << std::noshowpos << std::scientific
               << navier_stokes.get_diffusion_step_rhs_norm()
               << ", "
               << navier_stokes.get_projection_step_rhs_norm()
               << ") CFL = "
               << cfl_number
               << " ["
               << std::setw(5)
               << std::fixed
               << time_stepping.get_next_time()/time_stepping.get_end_time() * 100.
               << "%] \r";
}

template <int dim>
void CouetteFlowProblem<dim>::output()
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
                           "pressure");

  data_out.build_patches(velocity->fe_degree());

  static int out_index = 0;

  data_out.write_vtu_with_pvtu_record(this->prm.graphical_output_directory,
                                      "solution",
                                      out_index,
                                      this->mpi_communicator,
                                      5);

  out_index++;
}

template <int dim>
void CouetteFlowProblem<dim>::update_entities()
{
  velocity->update_solution_vectors();
  pressure->update_solution_vectors();
}

template <int dim>
void CouetteFlowProblem<dim>::solve(const unsigned int /* level */)
{
  setup_dofs();
  setup_constraints();
  velocity->setup_vectors();
  pressure->setup_vectors();
  initialize();

  // Outputs the fields at t_0, i.e. the initial conditions.
  output();

  while (time_stepping.get_current_time() < time_stepping.get_end_time())
  {
    // The VSIMEXMethod instance starts each loop at t^{k-1}

    // Compute CFL number
    cfl_number = navier_stokes.get_cfl_number();

    // Updates the coefficients to their k-th value
    time_stepping.update_coefficients();

    // Updates the functions and the constraints to t^{k}


    // Solves the system, i.e. computes the fields at t^{k}
    navier_stokes.solve();

    // Advances the VSIMEXMethod instance to t^{k}
    update_entities();
    time_stepping.advance_time();

    // Post-processing
    if ((time_stepping.get_step_number() %
          this->prm.terminal_output_frequency == 0) ||
        (time_stepping.get_current_time() ==
          time_stepping.get_end_time()))
      postprocessing();

    // Graphical output
    if ((time_stepping.get_step_number() %
          this->prm.graphical_output_frequency == 0) ||
        (time_stepping.get_current_time() ==
          time_stepping.get_end_time()))
      output();
  }

  *this->pcout << std::endl << std::endl;
}

template <int dim>
void CouetteFlowProblem<dim>::run()
{
  // Set ups the initial triangulation
  make_grid(parameters.spatial_discretization_parameters.n_initial_global_refinements);

  // The following if allows to perform either spatial or temporal
  // convergence tests, depending on the settings described in the
  // parameter file.
  switch (parameters.convergence_test_parameters.type)
  {
  case ConvergenceTest::Type::spatial:
    for (unsigned int level = parameters.spatial_discretization_parameters.n_initial_global_refinements;
         level < (parameters.spatial_discretization_parameters.n_initial_global_refinements +
                  parameters.convergence_test_parameters.n_spatial_cycles);
         ++level)
    {
      *this->pcout  << std::setprecision(1)
                    << "Solving until t = "
                    << std::fixed << time_stepping.get_end_time()
                    << " with a refinement level of " << level
                    << std::endl;

      time_stepping.restart();

      solve(level);

      {
        EquationData::VelocityExactSolution<dim>  exact_solution(traction_magnitude,
                                                                 reynolds_number);
        const auto error_map = RMHD::VectorTools::compute_error(*this->mapping,
                                                                *velocity,
                                                                exact_solution);
        velocity_convergence_table.update(error_map,
                                          velocity->get_dof_handler());
      }

      this->triangulation.refine_global();

      navier_stokes.clear();
    }
    break;
  case ConvergenceTest::Type::temporal:
    for (unsigned int cycle = 0;
         cycle < parameters.convergence_test_parameters.n_temporal_cycles;
         ++cycle)
    {
      double time_step = parameters.time_discretization_parameters.initial_time_step *
                         pow(parameters.convergence_test_parameters.step_size_reduction_factor,
                             cycle);

      *this->pcout  << std::setprecision(1)
                    << "Solving until t = "
                    << std::fixed << time_stepping.get_end_time()
                    << " with a refinement level of "
                    << parameters.spatial_discretization_parameters.n_initial_global_refinements
                    << std::endl;

      time_stepping.restart();

      time_stepping.set_desired_next_step_size(time_step);

      solve(parameters.spatial_discretization_parameters.n_initial_global_refinements);

      {
        EquationData::VelocityExactSolution<dim>  exact_solution(traction_magnitude,
                                                                 reynolds_number);
        const auto error_map = RMHD::VectorTools::compute_error(*this->mapping,
                                                                *velocity,
                                                                exact_solution);
        velocity_convergence_table.update(error_map,
                                          velocity->get_dof_handler(),
                                          time_step);
      }

      navier_stokes.clear();
    }
    break;
  default:
    break;
  }

  if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
  {
    if (!std::filesystem::exists(this->prm.graphical_output_directory))
    {
      try
      {
        std::filesystem::create_directories(this->prm.graphical_output_directory);
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

    const std::filesystem::path path{this->prm.graphical_output_directory};

    std::ostringstream  sstream;
    sstream << ((velocity_convergence_table.get_type() == ConvergenceTest::Type::spatial)?
                "Couette_SpatialConvergenceTest" : "Couette_TemporalConvergenceTest")
            << "_Re_"
            << Utilities::to_string(reynolds_number)
            << ".txt";

    std::filesystem::path filename = path / sstream.str();

    try
    {
      std::ofstream fstream(filename.string());
      velocity_convergence_table.write_text(fstream);
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

  *this->pcout << "Velocity convergence table" << std::endl;
  *this->pcout << std::string(80, '=') << std::endl;
  *this->pcout << velocity_convergence_table << std::endl;

}

} // namespace Couette

int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;
    using namespace Couette;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    std::string parameter_filename;
    if (argc >= 2)
      parameter_filename = argv[1];
    else
      parameter_filename = "Guermond.prm";

    RunTimeParameters::ProblemParameters parameter_set(parameter_filename, true);

    CouetteFlowProblem<2> simulation(parameter_set);

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
