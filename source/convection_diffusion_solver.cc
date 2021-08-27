#include <rotatingMHD/convection_diffusion_solver.h>
#include <rotatingMHD/time_discretization.h>

#include <deal.II/fe/mapping_q.h>
#include <cmath>

namespace RMHD
{


template <int dim>
ConvectionDiffusionSolver<dim>::ConvectionDiffusionSolver
(const ConvectionDiffusionSolverParameters        &parameters,
 TimeDiscretization::VSIMEXMethod                 &time_stepping,
 std::shared_ptr<Entities::FE_ScalarField<dim>>   &temperature,
 const std::shared_ptr<Mapping<dim>>               external_mapping,
 const std::shared_ptr<ConditionalOStream>         external_pcout,
 const std::shared_ptr<TimerOutput>                external_timer)
:
parameters(parameters),
mpi_communicator(MPI_COMM_WORLD),
time_stepping(time_stepping),
temperature(temperature),
flag_assemble_matrices{false},
flag_setup_problem{true},
flag_update_preconditioner{false},
rhs_norm{0}
{
  Assert(temperature.get() != nullptr,
         ExcMessage("The temperature's shared pointer has not be"
                    " initialized."));

  Assert(parameters.equation_coefficient > 0.0,
         ExcLowerRangeType<double>(parameters.equation_coefficient, 0.0));
  AssertIsFinite(parameters.equation_coefficient);

  // Initiating the internal Mapping instance.
  if (external_mapping.get() != nullptr)
    mapping = external_mapping;
  else
    mapping.reset(new MappingQ<dim>(1));

  // Initiating the internal ConditionalOStream instance.
  if (external_pcout.get() != nullptr)
    pcout = external_pcout;
  else
    pcout.reset(new ConditionalOStream(
      std::cout,
      Utilities::MPI::this_mpi_process(mpi_communicator) == 0));

  // Initiating the internal TimerOutput instance.
  if (external_timer.get() != nullptr)
    computing_timer  = external_timer;
  else
    computing_timer.reset(new TimerOutput(
      *pcout,
      TimerOutput::summary,
      TimerOutput::wall_times));

  // Explicitly set pointers to zero.
  source_term_ptr = nullptr;
  velocity_function_ptr = nullptr;
  velocity = nullptr;
}



template <int dim>
void ConvectionDiffusionSolver<dim>::clear()
{
  system_matrix.clear();
  mass_matrix.clear();
  stiffness_matrix.clear();
  mass_plus_stiffness_matrix.clear();
  advection_matrix.clear();

  rhs.clear();
  rhs_norm = 0;

  preconditioner.reset();

  flag_update_preconditioner = false;
  flag_assemble_matrices = false;
  flag_setup_problem = true;
}



template <int dim>
void ConvectionDiffusionSolver<dim>::set_source_term(Function<dim> &source_term)
{
  source_term_ptr = &source_term;
}



template <int dim>
void ConvectionDiffusionSolver<dim>::set_velocity(TensorFunction<1,dim> &velocity_function)
{
  AssertThrow(velocity == nullptr,
              ExcMessage("The velocity is already specified through a finite "
                         "element field."));
  velocity_function_ptr = &velocity_function;
}



template <int dim>
void ConvectionDiffusionSolver<dim>::set_velocity
(std::shared_ptr<const Entities::FE_VectorField<dim>> &velocity_fe_field)
{
  AssertThrow(velocity_function_ptr == nullptr,
              ExcMessage("The velocity is already specified through a function."));
  velocity = velocity_fe_field;
}

}  // namespace RMHD

// explicit instantiations

template class RMHD::ConvectionDiffusionSolver<2>;
template class RMHD::ConvectionDiffusionSolver<3>;
