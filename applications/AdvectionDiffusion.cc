#include <rotatingMHD/convection_diffusion_solver.h>
#include <rotatingMHD/finite_element_field.h>
#include <rotatingMHD/problem_class.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/time_discretization.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <memory>


namespace AdvectionDiffusion
{

using namespace dealii;
using namespace RMHD;

namespace EquationData
{

template <int dim>
class TemperatureExactSolution : public Function<dim>
{
public:
  TemperatureExactSolution(const double time = 0);

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;

  virtual Tensor<1, dim> gradient(const Point<dim> &point,
                                  const unsigned int = 0) const override;

private:
  /*!
   * @brief The wave number.
   */
  const double k = M_PI;
};



template <int dim>
TemperatureExactSolution<dim>::TemperatureExactSolution
(const double time)
:
Function<dim>(1, time)
{}



template<int dim>
double TemperatureExactSolution<dim>::value
(const Point<dim> &point,
 const unsigned int /* component */) const
{
  const double t = this->get_time();
  const double x = point(0);
  const double y = point(1);

  return (std::sin(k * x) * std::sin(k * y) * std::sin(k * t));
}



template<int dim>
Tensor<1, dim> TemperatureExactSolution<dim>::gradient
(const Point<dim> &point,
 const unsigned int /* component */) const
{
  Tensor<1, dim>  return_value;
  const double t = this->get_time();
  const double x = point(0);
  const double y = point(1);

  return_value[0] = k * std::cos(k * x) * std::sin(k * y) * std::sin(k * t);
  return_value[1] = k * std::sin(k * x) * std::cos(k * y) * std::sin(k * t);

  return return_value;
}



template <int dim>
class VelocityExactSolution : public TensorFunction<1, dim>
{
public:
  VelocityExactSolution(const double time = 0);

  virtual Tensor<1, dim> value(const Point<dim>  &p) const override;
};

template <int dim>
VelocityExactSolution<dim>::VelocityExactSolution
(const double time)
:
TensorFunction<1, dim>(time)
{}



template <int dim>
Tensor<1, dim> VelocityExactSolution<dim>::value
(const Point<dim>  &point) const
{
  (void)point;

  Tensor<1, dim>  return_value;

  return_value[0] = 1.0;
  return_value[1] = 0.0;

  return return_value;
}



template <int dim>
class SourceTerm : public Function<dim>
{
public:
  SourceTerm(const double Pe,
             const double time = 0);

  virtual double value(const Point<dim> &point,
                       const unsigned int component = 0) const override;

private:
  const double Pe;
};



template <int dim>
SourceTerm<dim>::SourceTerm(const double Pe,
                            const double time)
:
Function<dim>(1, time),
Pe(Pe)
{}



template <int dim>
double SourceTerm<dim>::value(const Point<dim> &point,
                              const unsigned int /* component */) const
{
  const double t = this->get_time();
  const double x = point(0);
  const double y = point(1);

  return (M_PI*std::cos(M_PI*x)*std::sin(M_PI*t)*std::sin(M_PI*y) //
          + M_PI*std::cos(M_PI*t)*std::sin(M_PI*x)*std::sin(M_PI*y)  //
          + (2*std::pow(M_PI,2)*std::sin(M_PI*t)*std::sin(M_PI*x)*std::sin(M_PI*y))/Pe);
}

}  // namespace EquationData


template <int dim>
class AdvectionDiffusionProblem : public Problem <dim>
{
public:
  AdvectionDiffusionProblem(const RunTimeParameters::ProblemParameters &parameters);

  void run();

private:
  const RunTimeParameters::ProblemParameters    &parameters;

  std::shared_ptr<Entities::FE_ScalarField<dim>>  scalar_field;

  std::shared_ptr<Function<dim>>                exact_solution;

  std::shared_ptr<TensorFunction<1,dim>>        velocity_field;

  EquationData::SourceTerm<dim>                 source_term;

  TimeDiscretization::VSIMEXMethod              time_stepping;

  ConvectionDiffusionSolver<dim>                             advection_diffusion;

  ConvergenceTable                              convergence_table;

  double                                        cfl_number;

  std::ofstream                                 log_file;

  void make_grid(const unsigned int &n_global_refinements);

  void setup_dofs();

  void setup_constraints();

  void initialize();

  void solve(const unsigned int &level);

  void postprocessing();

  void process_solution(const unsigned int cycle);

  void output();

  void update_entities();
};



template <int dim>
AdvectionDiffusionProblem<dim>::AdvectionDiffusionProblem(
  const RunTimeParameters::ProblemParameters &parameters)
:
Problem<dim>(parameters),
parameters(parameters),
scalar_field(std::make_shared<Entities::FE_ScalarField<dim>>(
  parameters.fe_degree_temperature,
  this->triangulation,
  "Scalar field")),
exact_solution(std::make_shared<EquationData::TemperatureExactSolution<dim>>(
  parameters.time_discretization_parameters.start_time)),
velocity_field(std::make_shared<EquationData::VelocityExactSolution<dim>>(
  parameters.time_discretization_parameters.start_time)),
source_term(parameters.Pe, parameters.time_discretization_parameters.start_time),
time_stepping(parameters.time_discretization_parameters),
advection_diffusion(parameters.heat_equation_parameters,
                    time_stepping,
                    scalar_field,
                    velocity_field,
                    this->mapping,
                    this->pcout,
                    this->computing_timer),
cfl_number(std::numeric_limits<double>::min()),
log_file("AdvectionDiffusion_Log.csv")
{}



template <int dim>
void AdvectionDiffusionProblem<dim>::make_grid(
  const unsigned int &n_global_refinements)
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Triangulation");

  GridGenerator::hyper_cube(this->triangulation,
                            0.0,
                            1.0,
                            false);

  this->triangulation.refine_global(n_global_refinements);
}



template <int dim>
void AdvectionDiffusionProblem<dim>::setup_dofs()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - DoFs");

  scalar_field->setup_dofs();
  *this->pcout  << "  Number of active cells                   = "
                << this->triangulation.n_global_active_cells()
                << std::endl
                << "  Number of temperature degrees of freedom = "
                << scalar_field->n_dofs()
                << std::endl;
}



template <int dim>
void AdvectionDiffusionProblem<dim>::setup_constraints()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Boundary conditions");

  scalar_field->clear_boundary_conditions();

  scalar_field->setup_boundary_conditions();

  exact_solution->set_time(time_stepping.get_start_time());

  scalar_field->set_dirichlet_boundary_condition(0);

  scalar_field->close_boundary_conditions();
  scalar_field->apply_boundary_conditions();
}



template <int dim>
void AdvectionDiffusionProblem<dim>::initialize()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Initial conditions");

  this->set_initial_conditions(scalar_field,
                               *exact_solution,
                               time_stepping);
}



template <int dim>
void AdvectionDiffusionProblem<dim>::postprocessing()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Postprocessing");

  std::cout.precision(1);
  *this->pcout  << static_cast<TimeDiscretization::DiscreteTime &>(time_stepping)
                << " Norm = "
                << std::noshowpos << std::scientific
                << advection_diffusion.get_rhs_norm()
                << " Progress ["
                << std::setw(5)
                << std::fixed
                << time_stepping.get_next_time()/time_stepping.get_end_time() * 100.
                << "%] \r";

  log_file << time_stepping.get_step_number()     << ","
           << time_stepping.get_current_time()    << ","
           << advection_diffusion.get_rhs_norm()        << ","
           << time_stepping.get_next_step_size()  << std::endl;
}


template <int dim>
void AdvectionDiffusionProblem<dim>::process_solution(const unsigned int cycle)
{
  const double current_time{time_stepping.get_current_time()};

  using ExactSolution = EquationData::TemperatureExactSolution<dim>;

  exact_solution->set_time(current_time);

  Vector<float> difference_per_cell(this->triangulation.n_active_cells());
  VectorTools::integrate_difference(scalar_field->get_dof_handler(),
                                    scalar_field->solution,
                                    *exact_solution,
                                    difference_per_cell,
                                    QGauss<dim>(scalar_field->fe_degree() + 1),
                                    VectorTools::L2_norm);
  const double L2_error =
    VectorTools::compute_global_error(this->triangulation,
                                      difference_per_cell,
                                      VectorTools::L2_norm);
  VectorTools::integrate_difference(scalar_field->get_dof_handler(),
                                    scalar_field->solution,
                                    *exact_solution,
                                    difference_per_cell,
                                    QGauss<dim>(scalar_field->fe_degree() + 1),
                                    VectorTools::H1_seminorm);
  const double H1_error =
    VectorTools::compute_global_error(this->triangulation,
                                      difference_per_cell,
                                      VectorTools::H1_seminorm);
  const QTrapez<1>     q_trapez;
  const QIterated<dim> q_iterated(q_trapez, scalar_field->fe_degree() * 2 + 1);
  VectorTools::integrate_difference(scalar_field->get_dof_handler(),
                                    scalar_field->solution,
                                    *exact_solution,
                                    difference_per_cell,
                                    q_iterated,
                                    VectorTools::Linfty_norm);
  const double Linfty_error =
    VectorTools::compute_global_error(this->triangulation,
                                      difference_per_cell,
                                      VectorTools::Linfty_norm);

  const unsigned int n_active_cells = this->triangulation.n_global_active_cells();
  const unsigned int n_dofs         = scalar_field->n_dofs();

  convergence_table.add_value("cycle", cycle);
  convergence_table.add_value("time_step", time_stepping.get_previous_step_size());
  convergence_table.add_value("cells", n_active_cells);
  convergence_table.add_value("dofs", n_dofs);
  convergence_table.add_value("L2", L2_error);
  convergence_table.add_value("H1", H1_error);
  convergence_table.add_value("Linfty", Linfty_error);

}



template <int dim>
void AdvectionDiffusionProblem<dim>::output()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Graphical output");

  DataOut<dim>  data_out;

  data_out.add_data_vector(scalar_field->get_dof_handler(),
                           scalar_field->solution,
                           "Scalar");
  data_out.build_patches(scalar_field->fe_degree());

  static int out_index = 0;

  data_out.write_vtu_with_pvtu_record(this->prm.graphical_output_directory,
                                      "Solution",
                                      out_index,
                                      this->mpi_communicator,
                                      5);

  out_index++;
}



template <int dim>
void AdvectionDiffusionProblem<dim>::solve(const unsigned int &/* level */)
{
  advection_diffusion.set_source_term(source_term);
  setup_dofs();
  setup_constraints();
  scalar_field->setup_vectors();
  initialize();

  // Outputs the fields at t_0, i.e. the initial conditions.
  {
    scalar_field->solution = scalar_field->old_solution;
    exact_solution->set_time(time_stepping.get_start_time());
    output();
  }

  while (time_stepping.get_current_time() < time_stepping.get_end_time())
  {
    // The VSIMEXMethod instance starts each loop at t^{k-1}

    // Updates the coefficients to their k-th value
    time_stepping.update_coefficients();

    // Updates the functions and the constraints to t^{k}
    exact_solution->set_time(time_stepping.get_next_time());

    // Solves the system, i.e. computes the fields at t^{k}
    advection_diffusion.solve();

    // Advances the VSIMEXMethod instance to t^{k}
    scalar_field->update_solution_vectors();
    time_stepping.advance_time();

    // Snapshot stage, all time calls should be done with get_next_time()

    if ((time_stepping.get_step_number() %
          this->prm.terminal_output_frequency == 0) ||
        (time_stepping.get_current_time() ==
          time_stepping.get_end_time()))
      postprocessing();

    if ((time_stepping.get_step_number() %
          this->prm.graphical_output_frequency == 0) ||
        (time_stepping.get_current_time() ==
          time_stepping.get_end_time()))
      output();
  }

  Assert(time_stepping.get_current_time() == exact_solution->get_time(),
    ExcMessage("Time mismatch between the time stepping class and the temperature function"));

  log_file << "\n";

  *this->pcout << std::endl << std::endl;
}



template <int dim>
void AdvectionDiffusionProblem<dim>::run()
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

      this->triangulation.refine_global();

      process_solution(level);
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

      process_solution(cycle);
    }
    break;
  default:
    break;
  }

  convergence_table.set_precision("L2", 3);
  convergence_table.set_precision("H1", 3);
  convergence_table.set_precision("Linfty", 3);
  convergence_table.set_scientific("L2", true);
  convergence_table.set_scientific("H1", true);
  convergence_table.set_scientific("Linfty", true);

  convergence_table.evaluate_convergence_rates("L2",
                                               "time_step",
                                               ConvergenceTable::RateMode::reduction_rate_log2,
                                               1);
  convergence_table.evaluate_convergence_rates("H1",
                                               "time_step",
                                               ConvergenceTable::RateMode::reduction_rate_log2,
                                               1);
  convergence_table.evaluate_convergence_rates("Linfty",
                                               "time_step",
                                               ConvergenceTable::RateMode::reduction_rate_log2,
                                               1);

  *this->pcout << std::endl;
  if (this->pcout->is_active())
    convergence_table.write_text(this->pcout->get_stream());

}



} // namespace AdvectionDiffusion



int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;
    using namespace AdvectionDiffusion;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,
                                                        argv,
                                                        1);

    RunTimeParameters::ProblemParameters parameter_set(
      "AdvectionDiffusion.prm",
      true);

    AdvectionDiffusionProblem<2> simulation(parameter_set);

    simulation.run();
  }
  catch(std::exception& exc)
  {
    std::cerr << std::endl << std::endl
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
    std::cerr << std::endl << std::endl
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
