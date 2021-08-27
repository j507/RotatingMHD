/*
 * convection_diffusion_problem.cc
 *
 *  Created on: Aug 27, 2021
 *      Author: sg
 */
#include <deal.II/numerics/data_out.h>

#include <rotatingMHD/convection_diffusion_problem.h>
#include <rotatingMHD/data_postprocessors.h>

namespace RMHD
{

template<int dim>
ConvectionDiffusionProblem<dim>::ConvectionDiffusionProblem
(ConvectionDiffusionParameters &prm,
 const std::string  &field_name)
:
Problem<dim>(prm),
parameters(prm),
scalar_field(std::make_shared<Entities::FE_ScalarField<dim>>
            (prm.fe_degree,
             this->triangulation,
             field_name)),
time_stepping(prm.time_discretization_parameters),
solver(prm.solver_parameters,
       time_stepping,
       scalar_field,
       this->mapping,
       this->pcout,
       this->computing_timer),
cfl_number(0.0)
{
  *this->pcout << parameters << std::endl << std::endl;
}

template<int dim>
void ConvectionDiffusionProblem<dim>::run()
{
  this->make_grid();

  setup_dofs();

  this->set_boundary_conditions();

  this->set_initial_conditions();

  this->set_source_term();

  this->set_velocity_field();

  // modify member variable of RMHD::TimeDiscretization::DiscreteTime if necessary
  const double desired_end_time{parameters.time_discretization_parameters.final_time};
  if (time_stepping.get_end_time() != desired_end_time)
      time_stepping.set_end_time(desired_end_time);

  const unsigned int n_maximum_steps{parameters.time_discretization_parameters.n_maximum_steps};

  *this->pcout << "Solving the problem until t = "
               << Utilities::to_string(time_stepping.get_end_time())
               << " or until "
               << Utilities::int_to_string(n_maximum_steps)
               << " time steps are performed"
               << std::endl;

  time_loop(n_maximum_steps);

  this->save_postprocessing_results();

  if (time_stepping.get_current_time() == time_stepping.get_end_time())
    *this->pcout << "This run completed successfully!"
                 << std::endl
                 << std::endl;
  else if (time_stepping.get_step_number() >= n_maximum_steps)
    *this->pcout << "This run terminated because the maximum number of steps was "
                    "reached! The current"
                 << std::endl
                 << "time is not equal to the desired final time."
                 << std::endl
                 << std::endl;
}



template <int dim>
void ConvectionDiffusionProblem<dim>::time_loop(const unsigned int n_steps)
{
  // Copy parameters for better readability of the code
  const bool adaptive_mesh_refinement{parameters.spatial_discretization_parameters.adaptive_mesh_refinement};
  const unsigned int refinement_frequency{parameters.spatial_discretization_parameters.adaptive_mesh_refinement_frequency};
  const unsigned int postprocessing_frequency{parameters.postprocessing_frequency};
  const unsigned int terminal_output_frequency{parameters.terminal_output_frequency};
  const unsigned int graphical_output_frequency{parameters.graphical_output_frequency};

  while (time_stepping.get_current_time() < time_stepping.get_end_time())
  {
    // The VSIMEXMethod instance starts each loop at t^{k-1}

    // Compute CFL number
    //  cfl_number = solver.get_cfl_number();

    // Update the time step, i.e., sets the value of t^{k}
    const double desired_next_step_size{this->compute_next_time_step(time_stepping, cfl_number)};
    time_stepping.set_desired_next_step_size(desired_next_step_size);

    if (time_stepping.get_step_number() % terminal_output_frequency == 0 )
      *this->pcout << static_cast<TimeDiscretization::DiscreteTime &>(time_stepping) << std::endl;

    // Update the coefficients to their k-th value
    time_stepping.update_coefficients();

    // Updates the functions and the constraints to t^{k}
    scalar_field->get_boundary_conditions().set_time(time_stepping.get_next_time());
    scalar_field->update_boundary_conditions();

    // Solves the system, i.e., computes the fields at t^{k}
    solver.solve();

    // Advances the VSIMEXMethod instance to t^{k}
    update_solution_vectors();
    time_stepping.advance_time();

    // Snapshot stage
    if (postprocessing_frequency > 0)
      if (time_stepping.get_step_number() % postprocessing_frequency == 0 ||
          time_stepping.get_current_time() == time_stepping.get_end_time())
        this->postprocess_solution();

    if (adaptive_mesh_refinement)
      if (time_stepping.get_step_number() % refinement_frequency == 0)
      {
        SolutionTransferContainer<dim>  container;
        container.add_entity(*scalar_field);
        this->adaptive_mesh_refinement(container);
      }

    if (graphical_output_frequency > 0)
      if ((time_stepping.get_step_number() % graphical_output_frequency == 0) ||
          (time_stepping.get_current_time() == time_stepping.get_end_time()))
        this->output_results();

    if (time_stepping.get_step_number() >= n_steps)
      break;
  }

  *this->pcout << static_cast<TimeDiscretization::DiscreteTime &>(time_stepping) << std::endl;
}



template<int dim>
void ConvectionDiffusionProblem<dim>::resume_from_snapshot(const std::string &)
{
  AssertThrow(false, ExcNotImplemented());
}



template<int dim>
void ConvectionDiffusionProblem<dim>::clear()
{
//  solver.clear();

  time_stepping.clear();

  scalar_field->clear();

  Problem<dim>::clear();
}



template<int dim>
void ConvectionDiffusionProblem<dim>::deserialize(const std::string &)
{
  AssertThrow(false, ExcNotImplemented());
}



template<int dim>
void ConvectionDiffusionProblem<dim>::serialize(const std::string &) const
{
  AssertThrow(false, ExcNotImplemented());
}



template<int dim>
void ConvectionDiffusionProblem<dim>::setup_dofs()
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Setup - DoFs");

  scalar_field->setup_dofs();

  *this->pcout << "Number of " << scalar_field->name << " degrees of freedom = "
               << scalar_field->n_dofs()
               << std::endl << std::endl;
}



template <int dim>
void ConvectionDiffusionProblem<dim>::output_results() const
{
  TimerOutput::Scope  t(*this->computing_timer, "Problem: Graphical output");

  RMHD::PostprocessorScalarField<dim> postprocessor(scalar_field->name);

  DataOut<dim>  data_out;
  data_out.add_data_vector(scalar_field->get_dof_handler(),
                           scalar_field->solution,
                           postprocessor);
  data_out.build_patches(scalar_field->fe_degree());

  static int out_index = 0;
  data_out.write_vtu_with_pvtu_record(this->output_directory.c_str(),
                                      "solution",
                                      out_index,
                                      this->mpi_communicator,
                                      5);
  out_index++;
}

// explicit instantiations
template class ConvectionDiffusionProblem<2>;
template class ConvectionDiffusionProblem<3>;

} // namespace RMHD

