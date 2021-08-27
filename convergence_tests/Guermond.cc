/*!
 * @file Guermond
 *
 * @brief This source is replicating the numerical test of section 3.7.2 of
 * the Guermond paper.
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
#include <deal.II/numerics/vector_tools.h>

#include <filesystem>
#include <string>
#include <sstream>
#include <memory>

namespace Guermond
{

using namespace dealii;
using namespace RMHD;

namespace EquationData
{

template <int dim>
class VelocityExactSolution : public Function<dim>
{
public:
  VelocityExactSolution(const double time = 0);

  virtual void vector_value(const Point<dim>  &p,
                            Vector<double>    &values) const override;

  virtual Tensor<1, dim> gradient(const Point<dim> &point,
                                  const unsigned int component) const;
};



template <int dim>
VelocityExactSolution<dim>::VelocityExactSolution(const double time)
:
Function<dim>(dim, time)
{}



template <int dim>
void VelocityExactSolution<dim>::vector_value(
                                        const Point<dim>  &point,
                                        Vector<double>    &values) const
{
  double t = this->get_time();
  double x = point(0);
  double y = point(1);
  values[0] = sin(x + t) * sin(y + t);
  values[1] = cos(x + t) * cos(y + t);
}



template <int dim>
Tensor<1, dim> VelocityExactSolution<dim>::gradient(
  const Point<dim>  &point,
  const unsigned int component) const
{
  Tensor<1, dim>  return_value;

  double t = this->get_time();
  double x = point(0);
  double y = point(1);
  // The gradient has to match that of dealii, i.e. from the right.
  if (component == 0)
  {
    return_value[0] = cos(x + t) * sin(y + t);
    return_value[1] = sin(x + t) * cos(y + t);
  }
  else if (component == 1)
  {
    return_value[0] = - sin(x + t) * cos(y + t);
    return_value[1] = - cos(x + t) * sin(y + t);
  }

  return return_value;
}


template <int dim>
class PressureExactSolution : public Function<dim>
{
public:
  PressureExactSolution(const double time = 0);

  virtual double value(const Point<dim> &p,
                      const unsigned int component = 0) const override;

  virtual Tensor<1, dim> gradient(const Point<dim> &point,
                                  const unsigned int = 0) const;
};



template <int dim>
PressureExactSolution<dim>::PressureExactSolution(const double time)
:
Function<dim>(1, time)
{}



template<int dim>
double PressureExactSolution<dim>::value
(const Point<dim> &point,
 const unsigned int /* component */) const
{
  double t = this->get_time();
  double x = point(0);
  double y = point(1);
  return sin(x - y + t);
}



template<int dim>
Tensor<1, dim> PressureExactSolution<dim>::gradient
(const Point<dim> &point,
 const unsigned int /* component */) const
{
  Tensor<1, dim>  return_value;
  double t = this->get_time();
  double x = point(0);
  double y = point(1);

  return_value[0] =   cos(x - y + t);
  return_value[1] = - cos(x - y + t);

  return return_value;
}



template <int dim>
class BodyForce: public TensorFunction<1, dim>
{
public:
  BodyForce(const double Re,
            const double time = 0);

  virtual Tensor<1, dim> value(
    const Point<dim>  &point) const override;

private:
  const double Re;
};



template <int dim>
BodyForce<dim>::BodyForce
(const double Re,
 const double time)
:
TensorFunction<1, dim>(time),
Re(Re)
{}



template <int dim>
Tensor<1, dim> BodyForce<dim>::value(const Point<dim> &point) const
{
  // The commented out lines corresponds to the case where the convection
  // term is ignored.
  Tensor<1, dim> value;

  double t = this->get_time();
  double x = point(0);
  double y = point(1);

  value[0] = cos(t + x - y) + sin(2.*(t + x))/2. +
              (2.*sin(t + x)*sin(t + y))/Re + sin(2.*t + x + y)
              /*cos(t + x - 1.*y) + (2.*sin(t + x)*sin(t + y))/Re
              + sin(2.*t + x + y)*/;
  value[1] = (cos(x - y) + cos(2.*t + x + y) - (Re*(2.*cos(t + x - y) +
              sin(2.*(t + y)) + 2.*sin(2.*t + x + y)))/2.)/Re
              /*(cos(x - 1.*y) + cos(2.*t + x + y) -
              1.*Re*(cos(t + x - 1.*y) + sin(2.*t + x + y)))/Re*/;

  return value;
}

}  // namespace EquationData


/*!
 * @class Guermond
 *
 * @todo Add documentation
 *
 */
template <int dim>
class GuermondProblem : public Problem<dim>
{
public:
  GuermondProblem(const RunTimeParameters::ProblemParameters &parameters);

  void run();

private:

  const RunTimeParameters::ProblemParameters   &parameters;

  std::ofstream                                 log_file;

  std::shared_ptr<Entities::FE_VectorField<dim>>  velocity;

  std::shared_ptr<Entities::FE_ScalarField<dim>>  pressure;

  TimeDiscretization::VSIMEXMethod              time_stepping;

  NavierStokesProjection<dim>                   navier_stokes;

  std::shared_ptr<EquationData::VelocityExactSolution<dim>>
                                                velocity_exact_solution;

  std::shared_ptr<EquationData::PressureExactSolution<dim>>
                                                pressure_exact_solution;

  EquationData::BodyForce<dim>                  body_force;

  ConvergenceTest::ConvergenceResults           velocity_convergence_table;

  ConvergenceTest::ConvergenceResults           pressure_convergence_table;

  double                                        cfl_number;

  const bool                                    flag_set_exact_pressure_constant;

  const bool                                    flag_square_domain;

  void make_grid(const unsigned int &n_global_refinements);

  void setup_dofs();

  void setup_constraints();

  void initialize();

  void postprocessing(const bool flag_point_evaluation);

  void output();

  void update_entities();

  void solve(const unsigned int level);
};

template <int dim>
GuermondProblem<dim>::GuermondProblem(const RunTimeParameters::ProblemParameters &parameters)
:
Problem<dim>(parameters),
parameters(parameters),
log_file("Guermond_Log.csv"),
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
velocity_exact_solution(
  std::make_shared<EquationData::VelocityExactSolution<dim>>(
    parameters.time_discretization_parameters.start_time)),
pressure_exact_solution(
  std::make_shared<EquationData::PressureExactSolution<dim>>(
    parameters.time_discretization_parameters.start_time)),
body_force(parameters.Re, parameters.time_discretization_parameters.start_time),
velocity_convergence_table(parameters.convergence_test_parameters.type),
pressure_convergence_table(parameters.convergence_test_parameters.type),
flag_set_exact_pressure_constant(true),
flag_square_domain(true)
{
  *this->pcout << parameters << std::endl << std::endl;

  log_file << "Step" << ","
           << "Time" << ","
           << "Norm_diffusion" << ","
           << "Norm_projection" << ","
           << "dt" << ","
           << "CFL" << std::endl;
}

template <int dim>
void GuermondProblem<dim>::
make_grid(const unsigned int &n_global_refinements)
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Triangulation");

  if (flag_square_domain)
    GridGenerator::hyper_cube(this->triangulation,
                              0.0,
                              1.0,
                              true);
  else
  {
    const double radius = 0.5;
    GridGenerator::hyper_ball(this->triangulation,
                              Point<dim>(),
                              radius,
                              true);
  }

  this->triangulation.refine_global(n_global_refinements);
}

template <int dim>
void GuermondProblem<dim>::setup_dofs()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - DoFs");

  velocity->setup_dofs();
  pressure->setup_dofs();

  *this->pcout << "  Number of active cells                = "
               << this->triangulation.n_global_active_cells()
               << std::endl;
  *this->pcout << "  Number of velocity degrees of freedom = "
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
void GuermondProblem<dim>::setup_constraints()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Boundary conditions");

  velocity->clear_boundary_conditions();
  pressure->clear_boundary_conditions();

  velocity->setup_boundary_conditions();
  pressure->setup_boundary_conditions();

  velocity_exact_solution->set_time(time_stepping.get_start_time());

  for (const auto& boundary_id : this->triangulation.get_boundary_ids())
    velocity->set_dirichlet_boundary_condition(boundary_id,
                                               velocity_exact_solution,
                                               true);

  pressure->set_datum_boundary_condition();

  velocity->close_boundary_conditions();
  pressure->close_boundary_conditions();

  velocity->apply_boundary_conditions();
  pressure->apply_boundary_conditions();
}

template <int dim>
void GuermondProblem<dim>::initialize()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Initial conditions");

  this->set_initial_conditions(velocity,
                                *velocity_exact_solution,
                                time_stepping);
  this->set_initial_conditions(pressure,
                                *pressure_exact_solution,
                                time_stepping);
}

template <int dim>
void GuermondProblem<dim>::postprocessing(const bool flag_point_evaluation)
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Postprocessing");

  if (flag_set_exact_pressure_constant)
  {
    RMHD::LinearAlgebra::MPI::Vector  analytical_pressure;
    RMHD::LinearAlgebra::MPI::Vector  distributed_analytical_pressure;
    RMHD::LinearAlgebra::MPI::Vector  distributed_numerical_pressure;

    analytical_pressure.reinit(pressure->solution);
    distributed_numerical_pressure.reinit(pressure->distributed_vector);

    RMHD::VectorTools::interpolate(*this->mapping,
                                   *pressure,
                                   *pressure_exact_solution,
                                   analytical_pressure);

    distributed_numerical_pressure = pressure->solution;

    const RMHD::LinearAlgebra::MPI::Vector::value_type analytical_mean_value
      = dealii::VectorTools::compute_mean_value(pressure->get_dof_handler(),
                                                QGauss<dim>(pressure->fe_degree() + 1),
                                                analytical_pressure,
                                                0);

    const RMHD::LinearAlgebra::MPI::Vector::value_type numerical_mean_value
      = dealii::VectorTools::compute_mean_value(pressure->get_dof_handler(),
                                                QGauss<dim>(pressure->fe_degree() + 1),
                                                pressure->solution,
                                                0);

    distributed_numerical_pressure.add(analytical_mean_value -
                                       numerical_mean_value);

    pressure->solution = distributed_numerical_pressure;
  }

  if (flag_point_evaluation)
  {
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

    log_file << time_stepping.get_step_number() << ","
             << time_stepping.get_current_time() << ","
             << navier_stokes.get_diffusion_step_rhs_norm() << ","
             << navier_stokes.get_projection_step_rhs_norm() << ","
             << time_stepping.get_next_step_size() << ","
             << cfl_number << std::endl;
  }
}

template <int dim>
void GuermondProblem<dim>::output()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Graphical output");

  std::vector<std::string> names(dim, "velocity");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  component_interpretation(dim,
                           DataComponentInterpretation::component_is_part_of_vector);

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
void GuermondProblem<dim>::update_entities()
{
  velocity->update_solution_vectors();
  pressure->update_solution_vectors();
}

template <int dim>
void GuermondProblem<dim>::solve(const unsigned int /* level */)
{
  navier_stokes.set_body_force(body_force);
  setup_dofs();
  setup_constraints();
  velocity->setup_vectors();
  pressure->setup_vectors();
  initialize();

  // Outputs the fields at t_0, i.e. the initial conditions.
  {
    velocity->solution = velocity->old_solution;
    pressure->solution = pressure->old_solution;
    velocity_exact_solution->set_time(time_stepping.get_start_time());
    pressure_exact_solution->set_time(time_stepping.get_start_time());
    output();
  }

  while (time_stepping.get_current_time() < time_stepping.get_end_time())
  {
    // The VSIMEXMethod instance starts each loop at t^{k-1}

    // Compute CFL number
    cfl_number = navier_stokes.get_cfl_number();

    // Updates the coefficients to their k-th value
    time_stepping.update_coefficients();

    // Updates the functions and the constraints to t^{k}
    velocity_exact_solution->set_time(time_stepping.get_next_time());
    pressure_exact_solution->set_time(time_stepping.get_next_time());

    velocity_exact_solution->set_time(time_stepping.get_next_time());
    velocity->update_boundary_conditions();

    // Solves the system, i.e. computes the fields at t^{k}
    navier_stokes.solve();

    // Advances the VSIMEXMethod instance to t^{k}
    update_entities();
    time_stepping.advance_time();

    // Snapshot stage
    postprocessing((time_stepping.get_step_number() %
                    this->prm.terminal_output_frequency == 0) ||
                    (time_stepping.get_current_time() ==
                   time_stepping.get_end_time()));

    if ((time_stepping.get_step_number() %
          this->prm.graphical_output_frequency == 0) ||
        (time_stepping.get_current_time() ==
          time_stepping.get_end_time()))
      output();
  }

  Assert(time_stepping.get_current_time() == velocity_exact_solution->get_time(),
         ExcMessage("Time mismatch between the time stepping class and the velocity function"));
  Assert(time_stepping.get_current_time() == pressure_exact_solution->get_time(),
         ExcMessage("Time mismatch between the time stepping class and the pressure function"));

  log_file << std::endl;

  *this->pcout << std::endl << std::endl;
}

template <int dim>
void GuermondProblem<dim>::run()
{
  make_grid(parameters.spatial_discretization_parameters.n_initial_global_refinements);

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
        auto error_map = RMHD::VectorTools::compute_error(*this->mapping,
                                                          *velocity,
                                                          *velocity_exact_solution);
        velocity_convergence_table.update(error_map,
                                          velocity->get_dof_handler());
        error_map.clear();

        error_map = RMHD::VectorTools::compute_error(*this->mapping,
                                                     *pressure,
                                                     *pressure_exact_solution);
        pressure_convergence_table.update(error_map,
                                          pressure->get_dof_handler());
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

      solve(this->prm.spatial_discretization_parameters.n_initial_global_refinements);

      {
        auto error_map = RMHD::VectorTools::compute_error(*this->mapping,
                                                          *velocity,
                                                          *velocity_exact_solution);
        velocity_convergence_table.update(error_map,
                                          velocity->get_dof_handler(),
                                          time_step);
        error_map.clear();

        error_map = RMHD::VectorTools::compute_error(*this->mapping,
                                                     *pressure,
                                                     *pressure_exact_solution);
        pressure_convergence_table.update(error_map,
                                          pressure->get_dof_handler(),
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
                "SpatialConvergenceTest" : "TemporalConvergenceTest")
            << ".txt";

    std::string auxiliary_filename{"Guermond_Velocity_"};
    auxiliary_filename += sstream.str();

    std::filesystem::path velocity_filename = path / auxiliary_filename;

    auxiliary_filename = "Guermond_Pressure_";
    auxiliary_filename += sstream.str();

    std::filesystem::path pressure_filename = path / auxiliary_filename;

    try
    {
      {
        std::ofstream fstream(velocity_filename.string());
        velocity_convergence_table.write_text(fstream);
      }
      {
        std::ofstream fstream(pressure_filename.string());
        velocity_convergence_table.write_text(fstream);
      }
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
  *this->pcout << std::endl;

  *this->pcout << "Pressure convergence table" << std::endl;
  *this->pcout << std::string(80, '=') << std::endl;
  *this->pcout << pressure_convergence_table << std::endl;
  *this->pcout << std::endl;
}

} // namespace Guermond

int main(int argc, char *argv[])
{
  try
  {
      using namespace dealii;
      using namespace Guermond;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      std::string parameter_filename;
      if (argc >= 2)
        parameter_filename = argv[1];
      else
        parameter_filename = "Guermond.prm";

      RunTimeParameters::ProblemParameters parameter_set(parameter_filename, true);

      GuermondProblem<2> simulation(parameter_set);

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
