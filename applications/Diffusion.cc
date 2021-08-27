#include <rotatingMHD/convection_diffusion_solver.h>
#include <rotatingMHD/finite_element_field.h>
#include <rotatingMHD/problem_class.h>
#include <rotatingMHD/run_time_parameters.h>
#include <rotatingMHD/time_discretization.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/function.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <memory>

namespace RMHD
{


template <int dim>
class ExactSolution : public Function<dim>
{
public:
  ExactSolution(const double Pe,
                const double time = 0);

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override;

  virtual Tensor<1, dim> gradient(const Point<dim> &point,
                                  const unsigned int = 0) const override;

private:
  /*!
   * @brief The Peclet number.
   */
  const double Pe;

  /*!
   * @brief The wave number.
   */
  const double k = 2. * M_PI;
};


template <int dim>
ExactSolution<dim>::ExactSolution
(const double Pe,
 const double time)
:
Function<dim>(1, time),
Pe(Pe)
{}

template<int dim>
double ExactSolution<dim>::value
(const Point<dim> &point,
 const unsigned int /* component */) const
{
  const double t = this->get_time();
  const double x = point(0);
  const double y = point(1);

  const double F = exp(-2.0 * k * k  / Pe * t);

  return (F *(cos(k * x) * sin(k * y)));
}

template<int dim>
Tensor<1, dim> ExactSolution<dim>::gradient
(const Point<dim> &point,
 const unsigned int /* component */) const
{
  Tensor<1, dim>  return_value;
  const double t = this->get_time();
  const double x = point(0);
  const double y = point(1);

  const double F = exp(-2.0 * k * k  / Pe * t);

  return_value[0] = - F * k * sin(k * x) * sin(k * y);
  return_value[1] = + F * k * cos(k * x) * cos(k * y);

  return return_value;
}


template <int dim>
class Diffusion : public Problem <dim>
{
public:
  Diffusion(const RunTimeParameters::ProblemParameters &parameters);

  void run();

private:
  const RunTimeParameters::ProblemParameters    &parameters;

  std::shared_ptr<Entities::FE_ScalarField<dim>>  scalar_field;

  TimeDiscretization::VSIMEXMethod              time_stepping;

  ConvectionDiffusionSolver<dim>                             advection_diffusion;

  ConvergenceTable                              convergence_table;

  void make_grid(const unsigned int n_global_refinements);

  void setup_dofs();

  void setup_constraints();

  void initialize();

  void solve();

  void postprocessing();

  void process_solution(const unsigned int);

  void output();

  void update_entities();
};



template <int dim>
Diffusion<dim>::Diffusion(
  const RunTimeParameters::ProblemParameters &parameters)
:
Problem<dim>(parameters),
parameters(parameters),
scalar_field(std::make_shared<Entities::FE_ScalarField<dim>>(
  parameters.fe_degree_temperature,
  this->triangulation,
  "Scalar field")),
time_stepping(parameters.time_discretization_parameters),
advection_diffusion(parameters.heat_equation_parameters,
                    time_stepping,
                    scalar_field,
                    this->mapping,
                    this->pcout,
                    this->computing_timer)
{}



template <int dim>
void Diffusion<dim>::make_grid(const unsigned int n_global_refinements)
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Triangulation");

  GridGenerator::hyper_cube(this->triangulation,
                            0.0,
                            1.0,
                            true);

  std::vector<GridTools::PeriodicFacePair<
    typename parallel::distributed::Triangulation<dim>::cell_iterator>>
    periodicity_vector;

  GridTools::collect_periodic_faces(this->triangulation,
                                    0,
                                    1,
                                    0,
                                    periodicity_vector);
  GridTools::collect_periodic_faces(this->triangulation,
                                    2,
                                    3,
                                    1,
                                    periodicity_vector);

  this->triangulation.add_periodicity(periodicity_vector);

  this->triangulation.refine_global(n_global_refinements);
}



template <int dim>
void Diffusion<dim>::setup_dofs()
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
void Diffusion<dim>::setup_constraints()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Boundary conditions");

  scalar_field->clear_boundary_conditions();
  scalar_field->setup_boundary_conditions();
  scalar_field->set_periodic_boundary_condition(0, 1, 0);
  scalar_field->set_periodic_boundary_condition(2, 3, 1);
  scalar_field->close_boundary_conditions();
  scalar_field->apply_boundary_conditions();
}



template <int dim>
void Diffusion<dim>::initialize()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - Initial conditions");

  ExactSolution<dim> initial_condition(parameters.Pe);

  this->set_initial_conditions(scalar_field,
                               initial_condition,
                               time_stepping);
}



template <int dim>
void Diffusion<dim>::process_solution(const unsigned int cycle)
{
  const double current_time{time_stepping.get_current_time()};

  Vector<float> difference_per_cell(this->triangulation.n_active_cells());
  VectorTools::integrate_difference(scalar_field->get_dof_handler(),
                                    scalar_field->solution,
                                    ExactSolution<dim>(parameters.Pe, current_time),
                                    difference_per_cell,
                                    QGauss<dim>(scalar_field->fe_degree() + 1),
                                    VectorTools::L2_norm);
  const double L2_error =
    VectorTools::compute_global_error(this->triangulation,
                                      difference_per_cell,
                                      VectorTools::L2_norm);
  VectorTools::integrate_difference(scalar_field->get_dof_handler(),
                                    scalar_field->solution,
                                    ExactSolution<dim>(parameters.Pe, current_time),
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
                                    ExactSolution<dim>(parameters.Pe, current_time),
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
void Diffusion<dim>::output()
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
void Diffusion<dim>::solve()
{
  setup_dofs();

  setup_constraints();

  scalar_field->setup_vectors();

  initialize();

  // Outputs the fields at t_0, i.e. the initial conditions.
  {
    scalar_field->solution = scalar_field->old_solution;
    output();
  }

  while (time_stepping.get_current_time() < time_stepping.get_end_time())
  {
    // The VSIMEXMethod instance starts each loop at t^{k-1}

    // Updates the coefficients to their k-th value
    *this->pcout << static_cast<TimeDiscretization::DiscreteTime &>(time_stepping)
                 << std::endl;
    time_stepping.update_coefficients();

    // Solves the system, i.e. computes the fields at t^{k}
    advection_diffusion.solve();

    // Advances the VSIMEXMethod instance to t^{k}
    scalar_field->update_solution_vectors();
    time_stepping.advance_time();

    // Snapshot stage, all time calls should be done with get_next_time()
    if ((time_stepping.get_step_number() %
          this->prm.graphical_output_frequency == 0) ||
        (time_stepping.get_current_time() ==
          time_stepping.get_end_time()))
      output();
  }

  *this->pcout << std::endl << std::endl;
}



template <int dim>
void Diffusion<dim>::run()
{
  make_grid(7);

  for (unsigned int cycle = 0; cycle < 6; ++cycle)
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
    solve();
    process_solution(cycle);
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


} // namespace RMHD



int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;
    using namespace RMHD;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc,
                                                        argv,
                                                        1);

    RunTimeParameters::ProblemParameters parameter_set("Diffusion.prm");
    Diffusion<2> simulation(parameter_set);
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
