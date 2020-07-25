#include <rotatingMHD/projection_solver.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/grid/grid_tools.h>

namespace Step35
{  
  template <int dim>
  void NavierStokesProjection<dim>::
  update_time_step()
  {
    dt_n_m1             = dt_n;
    const double v_max  = compute_max_velocity();

    if (v_max >= 0.01)
      dt_n = 1.0 / (1.7 * dim * std::sqrt(1. * dim )) 
            * GridTools::minimal_cell_diameter(triangulation)
            / v_max;
    else
      dt_n = 1.0 / (1.7 * dim * std::sqrt(1. * dim )) 
            * GridTools::minimal_cell_diameter(triangulation)
            / 0.01;
  }

  template <int dim>
  double NavierStokesProjection<dim>::
  compute_max_velocity()
  {
    const QIterated<dim>    quadrature_formula(QTrapez<1>(),
                                               v_fe_degree + 1);
    FEValues<dim>           fe_values(v_fe,
                                      quadrature_formula,
                                      update_values);
    const unsigned int      n_q_points    = quadrature_formula.size();
    double                  max_velocity  = 0.0;
    std::vector<Tensor<1,dim>>        velocity_values(n_q_points);

    const FEValuesExtractors::Vector  velocity(0);

    for (const auto &cell : v_dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      fe_values[velocity].get_function_values(v_n,
                                              velocity_values);
      for (unsigned int q = 0; q < n_q_points; ++q)
        max_velocity = std::max(max_velocity, velocity_values[q].norm());
    }
    return max_velocity;
  }
}

template void Step35::NavierStokesProjection<2>::update_time_step();
template void Step35::NavierStokesProjection<3>::update_time_step();
template double Step35::NavierStokesProjection<2>::compute_max_velocity();
template double Step35::NavierStokesProjection<3>::compute_max_velocity();