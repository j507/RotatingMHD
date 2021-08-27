#include <rotatingMHD/time_discretization.h>

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>

using namespace dealii;

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
  const double k = 2. * numbers::PI;
};


template <int dim>
ExactSolution<dim>::ExactSolution
(const double Pe,
 const double time)
:
Function<dim>(1, time),
Pe(Pe)
{}

template<>
double ExactSolution<2>::value
(const Point<2> &point,
 const unsigned int /* component */) const
{
  const double F = exp(-2.0 * k * k  / Pe * this->get_time());
  return (F *(cos(k * point[0]) * sin(k * point[1])));
}


template<>
Tensor<1, 2> ExactSolution<2>::gradient
(const Point<2> &point,
 const unsigned int /* component */) const
{
  const double F = exp(-2.0 * k * k  / Pe * this->get_time() );
  return Tensor<1, 2>({-F * k * sin(k * point[0]) * sin(k * point[1]),
                       F * k * cos(k * point[0]) * cos(k * point[1])});
}

template <int dim>
class VelocityField : public TensorFunction<1, dim>
{
public:
  VelocityField(const double time = 0);

  virtual Tensor<1, dim> value(const Point<dim>  &p) const override;

private:
  /*!
   * @brief The wave number.
   */
  const double k = 2. * numbers::PI;
};

template <int dim>
VelocityField<dim>::VelocityField
(const double time)
:
TensorFunction<1, dim>(time)
{}

template <>
Tensor<1, 2> VelocityField<2>::value
(const Point<2>  &point) const
{
  return Tensor<1, 2>({cos(k * point[0]) * cos(k * point[1]),
                       sin(k * point[0]) * sin(k * point[1])});
}


template <int dim>
class ConvectionDiffusionProblem
{
public:
  ConvectionDiffusionProblem(const RMHD::TimeDiscretization::TimeDiscretizationParameters &);

  void run();

private:

  void make_grid(const unsigned int n_global_refinements);

  void setup_dofs();

  void setup_system();

  void assemble_constant_matrices();

  void assemble_system();

  void assemble_system_rhs();

  void set_initial_condition();

  void solve_time_step();

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

  Vector<double> old_old_solution;

  Vector<double> system_rhs;

  Vector<double> auxiliary_vector;

  ConvergenceTable  convergence_table;

  RMHD::TimeDiscretization::VSIMEXMethod  discrete_time;

  const double theta;

  const double peclet;

  const types::boundary_id  left_boundary_id;

  const types::boundary_id  right_boundary_id;

  const types::boundary_id  bottom_boundary_id;

  const types::boundary_id  top_boundary_id;

};



template <int dim>
ConvectionDiffusionProblem<dim>::ConvectionDiffusionProblem(const RMHD::TimeDiscretization::TimeDiscretizationParameters &prm)
:
fe(1),
dof_handler(triangulation),
discrete_time(prm),
theta(0.5),
peclet(10.0),
left_boundary_id(0),
right_boundary_id(1),
bottom_boundary_id(2),
top_boundary_id(3)
{}



template <int dim>
void ConvectionDiffusionProblem<dim>::make_grid(const unsigned int n_global_refinements)
{
  GridGenerator::hyper_cube(this->triangulation,
                            0.0,
                            1.0,
                            true);

  std::vector<GridTools::PeriodicFacePair<typename  Triangulation<dim>::cell_iterator>>
  matched_pairs;

  GridTools::collect_periodic_faces(this->triangulation,
                                    left_boundary_id,
                                    right_boundary_id,
                                    0,
                                    matched_pairs);
  GridTools::collect_periodic_faces(this->triangulation,
                                    bottom_boundary_id,
                                    top_boundary_id,
                                    1,
                                    matched_pairs);

  this->triangulation.add_periodicity(matched_pairs);

  this->triangulation.refine_global(n_global_refinements);
}



template <int dim>
void ConvectionDiffusionProblem<dim>::setup_dofs()
{
  dof_handler.distribute_dofs(fe);

  constraints.clear();

  DoFTools::make_hanging_node_constraints(dof_handler, constraints);

  std::vector<GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator>>
  matched_pairs;

  GridTools::collect_periodic_faces(dof_handler,
                                    left_boundary_id,
                                    right_boundary_id,
                                    0,
                                    matched_pairs);
  GridTools::collect_periodic_faces(dof_handler,
                                    bottom_boundary_id,
                                    top_boundary_id,
                                    1,
                                    matched_pairs);

  DoFTools::make_periodicity_constraints<DoFHandler<dim>>(matched_pairs,
                                                          constraints);
  constraints.close();
}



template<int dim>
void ConvectionDiffusionProblem<dim>::setup_system()
{
  DynamicSparsityPattern dsp(dof_handler.n_dofs());

  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints);

  sparsity_pattern.copy_from(dsp);

  mass_matrix.reinit(sparsity_pattern);

  laplace_matrix.reinit(sparsity_pattern);

  system_matrix.reinit(sparsity_pattern);

  solution.reinit(dof_handler.n_dofs());
  old_solution.reinit(dof_handler.n_dofs());
  old_old_solution.reinit(dof_handler.n_dofs());
  system_rhs.reinit(dof_handler.n_dofs());
  auxiliary_vector.reinit(dof_handler.n_dofs());
}



template<int dim>
void ConvectionDiffusionProblem<dim>::assemble_constant_matrices()
{
  // Reset data
  mass_matrix    = 0.;
  laplace_matrix = 0.;

  const QGauss<dim>   quadrature_formula(fe.degree + 1);
  const unsigned int  n_q_points{quadrature_formula.size()};

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients | update_JxW_values);

  const unsigned int  dofs_per_cell{fe.n_dofs_per_cell()};

  FullMatrix<double>  local_mass_matrix(dofs_per_cell, dofs_per_cell);
  FullMatrix<double>  local_laplace_matrix(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double>         phi(dofs_per_cell);
  std::vector<Tensor<1,dim>>  grad_phi(dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    // Reset local data
    local_mass_matrix      = 0.;
    local_laplace_matrix = 0.;

    // Temperature's cell data
    fe_values.reinit(cell);

    // Local to global indices mapping
    cell->get_dof_indices(local_dof_indices);

    // Loop over quadrature points
    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      // Extract test function values at the quadrature points
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        phi[i]      = fe_values.shape_value(i, q);
        grad_phi[i] = fe_values.shape_grad(i, q);
      }

      // Loop over local degrees of freedom
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
        // Compute values of the lower triangular part (Symmetry)
        for (unsigned int j = 0; j <= i; ++j)
        {
          // Local matrices
          local_mass_matrix(i, j) += phi[i] * phi[j] * fe_values.JxW(q);
          local_laplace_matrix(i, j) += grad_phi[i] * grad_phi[j] * fe_values.JxW(q);
        } // Loop over local degrees of freedom

    } // Loop over quadrature points

    // Copy lower triangular part values into the upper triangular part
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      for (unsigned int j = i + 1; j < dofs_per_cell; ++j)
      {
        local_mass_matrix(i, j) = local_mass_matrix(j, i);
        local_laplace_matrix(i, j) = local_laplace_matrix(j, i);
      }

    constraints.distribute_local_to_global(local_laplace_matrix,
                                           local_dof_indices,
                                           laplace_matrix);
    constraints.distribute_local_to_global(local_mass_matrix,
                                           local_dof_indices,
                                           mass_matrix);
  }
}



template<int dim>
void ConvectionDiffusionProblem<dim>::assemble_system()
{
  if (discrete_time.get_step_number() == 0)
    assemble_constant_matrices();

  const double time_step{discrete_time.get_next_step_size()};
  const std::vector<double> alpha = discrete_time.get_alpha();
  const std::vector<double> gamma = discrete_time.get_gamma();

  system_matrix.copy_from(mass_matrix);
  system_matrix *= alpha[0];
  system_matrix.add(gamma[0] * time_step / peclet, laplace_matrix);
}



template<int dim>
void ConvectionDiffusionProblem<dim>::assemble_system_rhs()
{
  const double time_step{discrete_time.get_next_step_size()};
  const std::vector<double> alpha = discrete_time.get_alpha();
  const std::vector<double> beta = discrete_time.get_beta();
  const std::vector<double> gamma = discrete_time.get_gamma();

  VelocityField<dim>          velocity_function(peclet);

  // Reset data
  system_rhs = 0.;

  // Initiate the quadrature formula for exact numerical integration
  const QGauss<dim>   quadrature_formula(fe.degree + 1);
  const unsigned int  n_q_points{quadrature_formula.size()};

  std::vector<double>         explicit_term(n_q_points);
  std::vector<double>         convection_term(n_q_points);
  std::vector<Tensor<1,dim>>  diffusion_term(n_q_points);

  std::vector<double>         old_values(n_q_points);
  std::vector<Tensor<1,dim>>  old_gradients(n_q_points);

  std::vector<double>         old_old_values(n_q_points);
  std::vector<Tensor<1,dim>>  old_old_gradients(n_q_points);

  std::vector<Tensor<1,dim>>  old_velocity_values(n_q_points);
  std::vector<Tensor<1,dim>>  old_old_velocity_values(n_q_points);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_values | update_gradients |
                          update_JxW_values | update_quadrature_points);

  const unsigned int  dofs_per_cell{fe.n_dofs_per_cell()};

  Vector<double>      local_rhs(dofs_per_cell);
  FullMatrix<double>  local_matrix_for_inhomogeneous_bc(dofs_per_cell, dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  std::vector<double>         phi(dofs_per_cell);
  std::vector<Tensor<1,dim>>  grad_phi(dofs_per_cell);


  for (const auto &cell: dof_handler.active_cell_iterators())
  {
      // Reset local data
      local_rhs                          = 0.;
      local_matrix_for_inhomogeneous_bc  = 0.;

      // Local to global indices mapping
      cell->get_dof_indices(local_dof_indices);

      // Temperature
      fe_values.reinit(cell);

      fe_values.get_function_values(old_solution,
                                    old_values);
      fe_values.get_function_values(old_old_solution,
                                    old_old_values);

      fe_values.get_function_gradients(old_solution,
                                       old_gradients);
      fe_values.get_function_gradients(old_old_solution,
                                       old_old_gradients);

      velocity_function.set_time(discrete_time.get_previous_time());
      velocity_function.value_list(fe_values.get_quadrature_points(),
                                   old_old_velocity_values);
      velocity_function.set_time(discrete_time.get_current_time());
      velocity_function.value_list(fe_values.get_quadrature_points(),
                                   old_velocity_values);

      // Loop over quadrature points
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        // Evaluate the weak form of the right-hand side's terms at
        // the quadrature point
        explicit_term[q] = alpha[1] * old_values[q] + alpha[2] * old_old_values[q];
        diffusion_term[q] = time_step / peclet * (gamma[1] * old_gradients[q] +
                                                  gamma[2] * old_old_gradients[q]);
        convection_term[q] = time_step * (beta[1] * old_velocity_values[q] * old_gradients[q] +
                                          beta[2] * old_old_velocity_values[q] * old_old_gradients[q]);
        // Extract test function values at the quadrature points
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          phi[i]      = fe_values.shape_value(i,q);
          grad_phi[i] = fe_values.shape_grad(i,q);
        }

        // Loop over local degrees of freedom
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          // Local right hand side (Domain integrals)
          local_rhs(i) -= (grad_phi[i] * diffusion_term[q] +
                           phi[i] * convection_term[q] +
                           phi[i] * explicit_term[q] ) * fe_values.JxW(q);

          // Loop over the i-th column's rows of the local matrix
          // for the case of inhomogeneous Dirichlet boundary conditions
          if (constraints.is_inhomogeneously_constrained(local_dof_indices[i]))
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
            {
              local_matrix_for_inhomogeneous_bc(j,i) +=
                  (alpha[0] * phi[j] * phi[i]
                   + gamma[0] * time_step / peclet * grad_phi[j] * grad_phi[i]
                  ) * fe_values.JxW(q);
            } // Loop over the i-th column's rows of the local matrix

        } // Loop over local degrees of freedom

      } // Loop over quadrature points

      constraints.distribute_local_to_global(local_rhs,
                                             local_dof_indices,
                                             system_rhs,
                                             local_matrix_for_inhomogeneous_bc);
  }
} // assemble_local_rhs




template <int dim>
void ConvectionDiffusionProblem<dim>::set_initial_condition()
{
  Assert(discrete_time.get_current_time() == discrete_time.get_start_time(),
         ExcMessage("Error in set_initial_condition"));

  VectorTools::project(dof_handler,
                       constraints,
                       QGauss<dim>(fe.degree + 1),
                       ExactSolution<dim>(peclet),
                       old_solution);
  solution = old_solution;
}



template <int dim>
void ConvectionDiffusionProblem<dim>::solve_time_step()
{
  if (discrete_time.coefficients_changed())
    assemble_system();

  assemble_system_rhs();

  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, 1.0);

  SolverControl            solver_control(1000, 1e-8 * system_rhs.l2_norm());
  SolverCG<Vector<double>> cg(solver_control);
  cg.solve(system_matrix, solution, system_rhs, preconditioner);

  constraints.distribute(solution);
}



template <int dim>
void ConvectionDiffusionProblem<dim>::process_solution(const unsigned int cycle)
{
  const double current_time{discrete_time.get_current_time()};

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
  convergence_table.add_value("cycle", cycle);
  convergence_table.add_value("time_step", discrete_time.get_previous_step_size());
  convergence_table.add_value("cells", n_active_cells);
  convergence_table.add_value("dofs", n_dofs);
  convergence_table.add_value("L2", L2_error);
  convergence_table.add_value("H1", H1_error);
  convergence_table.add_value("Linfty", Linfty_error);
}



template <int dim>
void ConvectionDiffusionProblem<dim>::run()
{
  make_grid(7);

  setup_dofs();

  setup_system();

  assemble_system();

  for (unsigned int cycle = 0; cycle < 5; ++cycle)
  {
    const double time_step{discrete_time.get_maximum_step_size() * std::pow(0.5, cycle)};

    discrete_time.restart();
    discrete_time.set_desired_next_step_size(time_step);

    set_initial_condition();

    while (discrete_time.get_current_time() < discrete_time.get_end_time())
    {
      discrete_time.update_coefficients();

      // solve
      solve_time_step();

      // Advances time stepping instance to t^{k}
      discrete_time.advance_time();
      old_old_solution = old_solution;
      old_solution = solution;
    }

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
  std::cout << std::endl;
  convergence_table.write_text(std::cout);

}



int main()
{
  try
  {
    using namespace dealii;

    RMHD::TimeDiscretization::TimeDiscretizationParameters parameters;
    parameters.initial_time_step = 0.5;
    parameters.maximum_time_step = 0.5;
    parameters.minimum_time_step = 1e-9;
    parameters.final_time = 1.0;
    parameters.adaptive_time_stepping = false;

    // Crank-Nicolson scheme
    parameters.vsimex_scheme = RMHD::TimeDiscretization::VSIMEXScheme::CNAB;
    {
      ConvectionDiffusionProblem<2> simulation(parameters);
      simulation.run();
    }
    // BDF2 scheme
    parameters.vsimex_scheme = RMHD::TimeDiscretization::VSIMEXScheme::BDF2;
    {
      ConvectionDiffusionProblem<2> simulation(parameters);
      simulation.run();
    }
    // modified Crank-Nicolson scheme
    parameters.vsimex_scheme = RMHD::TimeDiscretization::VSIMEXScheme::mCNAB;
    {
      ConvectionDiffusionProblem<2> simulation(parameters);
      simulation.run();
    }
    // Crank-Nicolson leap frog scheme
    parameters.vsimex_scheme = RMHD::TimeDiscretization::VSIMEXScheme::CNLF;
    {
      ConvectionDiffusionProblem<2> simulation(parameters);
      simulation.run();
    }
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
