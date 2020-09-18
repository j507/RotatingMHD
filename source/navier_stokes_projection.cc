#include <rotatingMHD/navier_stokes_projection.h>
#include <rotatingMHD/time_discretization.h>

namespace RMHD
{

template <int dim>
NavierStokesProjection<dim>::NavierStokesProjection
(const RunTimeParameters::ParameterSet   &parameters,
 Entities::VectorEntity<dim>             &velocity,
 Entities::ScalarEntity<dim>             &pressure,
 TimeDiscretization::VSIMEXMethod        &time_stepping,
 const std::shared_ptr<ConditionalOStream>external_pcout,
 const std::shared_ptr<TimerOutput>       external_timer)
:
parameters(parameters),
mpi_communicator(velocity.mpi_communicator),
velocity(velocity),
pressure(pressure),
time_stepping(time_stepping),
flag_diffusion_matrix_assembled(false)
{
  if (external_pcout.get() != 0)
    pcout = external_pcout;
  else
    pcout.reset(new ConditionalOStream(std::cout,
                                       Utilities::MPI::this_mpi_process(mpi_communicator) == 0));

  if (external_timer.get() != 0)
      computing_timer  = external_timer;
  else
      computing_timer.reset(new TimerOutput(*pcout,
                                            TimerOutput::summary,
                                            TimerOutput::wall_times));
}

}  // namespace RMHD

// explicit instantiations
template class RMHD::NavierStokesProjection<2>;
template class RMHD::NavierStokesProjection<3>;

