/*
 * heat_equation.cc
 *
 *  Created on: Aug 9, 2021
 *      Author: sg
 */

#include <deal.II/base/utilities.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/matrix_tools.h>
#include <fstream>
#include <iostream>
namespace Step26
{
  using namespace dealii;
  template <int dim>
  class HeatEquation
  {
  public:
    HeatEquation();
    void run();
  private:
    void setup_system();
    void solve_time_step();
    void output_results() const;
    void refine_mesh(const unsigned int min_grid_level,
                     const unsigned int max_grid_level);
    void process_solution(const unsigned int cycle);
    Triangulation<dim> triangulation;
    FE_Q<dim>          fe;
    DoFHandler<dim>    dof_handler;
    AffineConstraints<double> constraints;
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> mass_matrix;
    SparseMatrix<double> laplace_matrix;
    SparseMatrix<double> system_matrix;
    Vector<double> solution;
    Vector<double> old_solution;
    Vector<double> system_rhs;
    ConvergenceTable convergence_table;
    double       current_time;
    double       time_step;
    unsigned int timestep_number;
    const double theta;
    const double peclet;
  };

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
    const double Pe;
    const double k;
  };


  template <int dim>
  ExactSolution<dim>::ExactSolution
  (const double Pe,
   const double time)
  :
  Function<dim>(1, time),
  Pe(Pe),
  k(2.0 * numbers::PI)
  {}

  template<int dim>
  double ExactSolution<dim>::value
  (const Point<dim> &point,
   const unsigned int /* component */) const
  {
    const double F = exp(-2.0 * k * k  / Pe * this->get_time());

    return (F *(sin(k * point[0]) * sin(k * point[1])));
  }

  template<int dim>
  Tensor<1, dim> ExactSolution<dim>::gradient
  (const Point<dim> &point,
   const unsigned int /* component */) const
  {
    Tensor<1, dim>  return_value;
    const double F = exp(-2.0 * k * k  / Pe * this->get_time());

    return_value[0] = F * k * cos(k * point[0]) * sin(k * point[1]);
    return_value[1] = F * k * sin(k * point[0]) * cos(k * point[1]);

    return return_value;
  }

  template <int dim>
  class BoundaryValues : public Function<dim>
  {
  public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };
  template <int dim>
  double BoundaryValues<dim>::value(const Point<dim> & /*p*/,
                                    const unsigned int component) const
  {
    (void)component;
    Assert(component == 0, ExcIndexRange(component, 0, 1));
    return 0;
  }

  template <int dim>
  HeatEquation<dim>::HeatEquation()
    : fe(1)
    , dof_handler(triangulation)
    , current_time(0.0)
    , time_step(1. / 10)
    , timestep_number(0)
    , theta(0.5)
    , peclet(100.0)
  {}

  template <int dim>
  void HeatEquation<dim>::setup_system()
  {
    dof_handler.distribute_dofs(fe);
    std::cout << std::endl
              << "===========================================" << std::endl
              << "Number of active cells: " << triangulation.n_active_cells()
              << std::endl
              << "Number of degrees of freedom: " << dof_handler.n_dofs()
              << std::endl
              << std::endl;
    constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler, constraints);
    constraints.close();
    DynamicSparsityPattern dsp(dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    constraints,
                                    /*keep_constrained_dofs = */ true);
    sparsity_pattern.copy_from(dsp);
    mass_matrix.reinit(sparsity_pattern);
    laplace_matrix.reinit(sparsity_pattern);
    system_matrix.reinit(sparsity_pattern);
    MatrixCreator::create_mass_matrix(dof_handler,
                                      QGauss<dim>(fe.degree + 1),
                                      mass_matrix);
    MatrixCreator::create_laplace_matrix(dof_handler,
                                         QGauss<dim>(fe.degree + 1),
                                         laplace_matrix);
    solution.reinit(dof_handler.n_dofs());
    old_solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }

  template <int dim>
  void HeatEquation<dim>::solve_time_step()
  {
    SolverControl            solver_control(1000, 1e-8 * system_rhs.l2_norm());
    SolverCG<Vector<double>> cg(solver_control);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.0);
    cg.solve(system_matrix, solution, system_rhs, preconditioner);
    constraints.distribute(solution);
    std::cout << "     " << solver_control.last_step() << " CG iterations."
              << std::endl;
  }

  template <int dim>
  void HeatEquation<dim>::output_results() const
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "U");
    data_out.build_patches();
    data_out.set_flags(DataOutBase::VtkFlags(current_time, timestep_number));
    const std::string filename =
      "solution-" + Utilities::int_to_string(timestep_number, 3) + ".vtk";
    std::ofstream output(filename);
    data_out.write_vtk(output);
  }

  template <int dim>
  void HeatEquation<dim>::refine_mesh(const unsigned int min_grid_level,
                                      const unsigned int max_grid_level)
  {
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
    KellyErrorEstimator<dim>::estimate(
      dof_handler,
      QGauss<dim - 1>(fe.degree + 1),
      std::map<types::boundary_id, const Function<dim> *>(),
      solution,
      estimated_error_per_cell);
    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                      estimated_error_per_cell,
                                                      0.6,
                                                      0.4);
    if (triangulation.n_levels() > max_grid_level)
      for (const auto &cell :
           triangulation.active_cell_iterators_on_level(max_grid_level))
        cell->clear_refine_flag();
    for (const auto &cell :
         triangulation.active_cell_iterators_on_level(min_grid_level))
      cell->clear_coarsen_flag();
    SolutionTransfer<dim> solution_trans(dof_handler);
    Vector<double> previous_solution;
    previous_solution = solution;
    triangulation.prepare_coarsening_and_refinement();
    solution_trans.prepare_for_coarsening_and_refinement(previous_solution);
    triangulation.execute_coarsening_and_refinement();
    setup_system();
    solution_trans.interpolate(previous_solution, solution);
    constraints.distribute(solution);
  }

  template <int dim>
  void HeatEquation<dim>::process_solution(const unsigned int cycle)
  {
    Vector<float> difference_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      ExactSolution<dim>(peclet, current_time),
                                      difference_per_cell,
                                      QGauss<dim>(fe.degree + 1),
                                      VectorTools::L2_norm);
    const double L2_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::L2_norm);
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      ExactSolution<dim>(peclet, current_time),
                                      difference_per_cell,
                                      QGauss<dim>(fe.degree + 1),
                                      VectorTools::H1_seminorm);
    const double H1_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::H1_seminorm);
    const QTrapez<1>     q_trapez;
    const QIterated<dim> q_iterated(q_trapez, fe.degree * 2 + 1);
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      ExactSolution<dim>(peclet, current_time),
                                      difference_per_cell,
                                      q_iterated,
                                      VectorTools::Linfty_norm);
    const double Linfty_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::Linfty_norm);
    const unsigned int n_active_cells = triangulation.n_active_cells();
    const unsigned int n_dofs         = dof_handler.n_dofs();
    std::cout << "Cycle " << cycle << ':' << std::endl
              << "   Number of active cells:       " << n_active_cells
              << std::endl
              << "   Number of degrees of freedom: " << n_dofs << std::endl;
    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("time_step", time_step);
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("L2", L2_error);
    convergence_table.add_value("H1", H1_error);
    convergence_table.add_value("Linfty", Linfty_error);
  }


  template <int dim>
  void HeatEquation<dim>::run()
  {
    const unsigned int initial_global_refinement       = 9;
    const unsigned int n_temporal_cycles               = 8;
    GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(initial_global_refinement);
    setup_system();
    Vector<double> tmp;
    tmp.reinit(solution.size());
    for (unsigned int i=0; i<n_temporal_cycles; ++i)
    {
      current_time = 0.0;
      time_step = (1. / 10) * std::pow(0.5, i);
      timestep_number = 0;

      VectorTools::interpolate(dof_handler,
                               ExactSolution<dim>(peclet),
                               old_solution);
      solution = old_solution;
      output_results();

      while (current_time < 1.0)
      {
        const double next_time = current_time + time_step;
        ++timestep_number;
        std::cout << "Time step " << timestep_number << " at t=" << current_time
                  << std::endl;
        mass_matrix.vmult(system_rhs, old_solution);
        laplace_matrix.vmult(tmp, old_solution);
        system_rhs.add(-(1 - theta) * time_step / peclet, tmp);
        system_matrix.copy_from(mass_matrix);
        system_matrix.add(theta * time_step / peclet, laplace_matrix);
        constraints.condense(system_matrix, system_rhs);
        {
          BoundaryValues<dim> boundary_values_function;
          boundary_values_function.set_time(next_time);
          std::map<types::global_dof_index, double> boundary_values;
          VectorTools::interpolate_boundary_values(dof_handler,
                                                   0,
                                                   boundary_values_function,
                                                   boundary_values);
          MatrixTools::apply_boundary_values(boundary_values,
                                             system_matrix,
                                             solution,
                                             system_rhs);
        }
        solve_time_step();
        if (timestep_number % 100 == 0)
          output_results();

        old_solution = solution;
        current_time = next_time;
      }

      std::cout << "Time step " << timestep_number << " at t=" << current_time
                << std::endl;
      output_results();
      process_solution(i);
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
    std::cout << std::endl;
    convergence_table.write_text(std::cout);

  }
} // namespace Step26


int main()
{
  try
    {
      using namespace Step26;
      HeatEquation<2> heat_equation_solver;
      heat_equation_solver.run();
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
