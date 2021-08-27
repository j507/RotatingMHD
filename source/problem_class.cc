#include <deal.II/base/quadrature_lib.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/numerics/error_estimator.h>

#include <rotatingMHD/problem_class.h>

#include <exception>
#include <string>

namespace RMHD
{

using namespace dealii;

template<int dim>
SolutionTransferContainer<dim>::SolutionTransferContainer()
:
error_vector_size(0)
{}



template<int dim>
void SolutionTransferContainer<dim>::serialize
(const std::string &file_name) const
{
  Assert(!empty(),
         ExcMessage("The entities container is empty."))

  using SolutionTransferType =
  parallel::distributed::SolutionTransfer<dim, LinearAlgebra::MPI::Vector>;

  // Store the current solutions into a std::vector
  const std::vector<TransferVectorType> transfer_vectors = get_transfer_vectors();

  // Initiate the objects responsible for the solution transfer
  std::vector<SolutionTransferType> transfer_objects = get_transfer_objects();
  // Prepare each entry for the solution transfer
  for (std::size_t i=0; i<entities.size(); ++i)
    transfer_objects[i].prepare_for_serialization(transfer_vectors[i]);

  // Cast pointer
  const parallel::distributed::Triangulation<dim> *dist_triangulation =
      dynamic_cast<const parallel::distributed::Triangulation<dim> *>(triangulation);

  AssertThrow(dist_triangulation != nullptr,
              ExcInternalError());

  // Serialize
  dist_triangulation->save(file_name);

}


template<int dim>
void SolutionTransferContainer<dim>::deserialize
(Triangulation<dim> &tria,
 const std::string  &file_name)
{
  AssertThrow(&tria == &(*triangulation),
              ExcMessage("Input triangulation is equal to that of the finite "
                         "element fields."));

  // Cast pointer
  parallel::distributed::Triangulation<dim> *dist_triangulation =
      dynamic_cast<parallel::distributed::Triangulation<dim> *>(&tria);

  AssertThrow(dist_triangulation != nullptr,
              ExcInternalError());

  try
  {
    dist_triangulation->load(file_name);
  }
  catch (...)
  {
      AssertThrow(false,
                  ExcMessage("Cannot open snapshot mesh file or read the"
                             "triangulation stored there."));
  }

  for (auto &entity: entities)
  {
    entity.first->setup_dofs();
    entity.first->setup_vectors();

    VectorType  distributed_solution(entity.first->distributed_vector);
    VectorType  distributed_old_solution(entity.first->distributed_vector);
    VectorType  distributed_old_old_solution(entity.first->distributed_vector);

    DeserializeVectorType x_solution(3);
    x_solution[0] = &distributed_solution;
    x_solution[1] = &distributed_old_solution;
    x_solution[2] = &distributed_old_old_solution;

    SolutionTransferType solution_transfer(entity.first->get_dof_handler());

    solution_transfer.deserialize(x_solution);

    entity.first->solution = distributed_solution;
    entity.first->old_solution = distributed_old_solution;
    entity.first->old_old_solution = distributed_old_old_solution;
  }
}




template<int dim>
void SolutionTransferContainer<dim>::add_entity
(Entities::FE_FieldBase<dim> &entity,
 bool flag)
{
  const Triangulation<dim>  &tria{entity.get_triangulation()};

  if (entities.empty())
    triangulation = &tria;
  else
    AssertThrow(&tria == &(*triangulation),
                ExcMessage("Entities do not share the same triangulation."));

  entities.emplace_back(std::make_pair(&entity, flag));
  if (flag)
    error_vector_size += 1;
}



template <int dim>
std::vector<typename SolutionTransferContainer<dim>::SolutionTransferType>
SolutionTransferContainer<dim>::get_transfer_objects() const
{
  std::vector<typename SolutionTransferContainer<dim>::SolutionTransferType>
  transfer_objects;

  for (const auto &field: entities)
    transfer_objects.emplace_back(field.first->get_dof_handler());

  return (transfer_objects);
}



template <int dim>
std::vector<typename SolutionTransferContainer<dim>::TransferVectorType>
SolutionTransferContainer<dim>::get_transfer_vectors() const
{
  std::vector<typename SolutionTransferContainer<dim>::TransferVectorType>
  transfer_vectors;

  for (const auto &field: entities)
  {
    typename SolutionTransferContainer<dim>::TransferVectorType
    vector(3);
    vector[0] = &(field.first->solution);
    vector[1] = &(field.first->old_solution);
    vector[2] = &(field.first->old_old_solution);

    transfer_vectors.push_back(vector);
  }

  return (transfer_vectors);
}



template<int dim>
Problem<dim>::Problem(const RunTimeParameters::ProblemBaseParameters &prm)
:
mpi_communicator(MPI_COMM_WORLD),
triangulation(mpi_communicator,
              typename Triangulation<dim>::MeshSmoothing(
              Triangulation<dim>::smoothing_on_refinement |
              Triangulation<dim>::smoothing_on_coarsening)),
mapping(std::make_shared<MappingQ<dim>>(prm.mapping_degree,
                                        prm.mapping_interior_cells)),
pcout(std::make_shared<ConditionalOStream>(std::cout,
      (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))),
computing_timer(
  std::make_shared<TimerOutput>(mpi_communicator,
                                *pcout,
                                (prm.verbose? TimerOutput::summary: TimerOutput::never),
                                TimerOutput::wall_times)),
output_directory(prm.output_directory),
cell_fraction_to_coarsen{prm.spatial_discretization_parameters.cell_fraction_to_coarsen},
cell_fraction_to_refine{prm.spatial_discretization_parameters.cell_fraction_to_refine},
n_minimum_levels{prm.spatial_discretization_parameters.n_minimum_levels},
n_maximum_levels{prm.spatial_discretization_parameters.n_maximum_levels},
adaptive_time_stepping{prm.time_discretization_parameters.adaptive_time_stepping}
{
  if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0)
  {
    if (!std::filesystem::exists(output_directory))
    {
      try
      {
        std::filesystem::create_directories(output_directory);
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
  }
}



template <int dim>
void Problem<dim>::clear()
{
  triangulation.clear();
}




template <int dim>
void Problem<dim>::set_initial_conditions
(std::shared_ptr<Entities::FE_FieldBase<dim>> entity,
 Function<dim>                              &function,
 const TimeDiscretization::VSIMEXMethod     &time_stepping,
 const bool                                 boolean)
{
  #ifdef USE_PETSC_LA
    LinearAlgebra::MPI::Vector
    tmp_old_solution(entity->get_locally_owned_dofs(), mpi_communicator);
  #else
    LinearAlgebra::MPI::Vector
    tmp_old_solution(entity->get_locally_owned_dofs());
  #endif

  function.set_time(time_stepping.get_start_time());

  if (!boolean)
  {
    VectorTools::project(entity->get_dof_handler(),
                         entity->get_constraints(),
                          QGauss<dim>(entity->fe_degree() + 2),
                          function,
                          tmp_old_solution);

    entity->old_solution = tmp_old_solution;
  }
  else
  {
    LinearAlgebra::MPI::Vector tmp_old_old_solution(tmp_old_solution);

    VectorTools::project(entity->get_dof_handler(),
                         entity->get_constraints(),
                          QGauss<dim>(entity->fe_degree() + 2),
                          function,
                          tmp_old_old_solution);

    function.advance_time(time_stepping.get_next_step_size());

    VectorTools::project(entity->get_dof_handler(),
                         entity->get_constraints(),
                          QGauss<dim>(entity->fe_degree() + 2),
                          function,
                          tmp_old_solution);

    entity->old_old_solution = tmp_old_old_solution;
    entity->old_solution     = tmp_old_solution;
  }

}

template <int dim>
double Problem<dim>::compute_next_time_step
(const TimeDiscretization::VSIMEXMethod &time_stepping,
 const double                           cfl_number,
 const double                           max_cfl_number) const
{
  if (!adaptive_time_stepping || time_stepping.get_step_number() == 0)
    return time_stepping.get_next_step_size();
  else if (cfl_number < 1e-6)
    return time_stepping.get_next_step_size();
  else
    return max_cfl_number / cfl_number * time_stepping.get_next_step_size();
}

template <int dim>
void Problem<dim>::adaptive_mesh_refinement
(const SolutionTransferContainer<dim> &container)
{
  Assert(!container.empty(),
         ExcMessage("The entities container is empty."))

  const std::vector<typename SolutionTransferContainer<dim>::FE_Field>
  entities = container.get_field_collection();

  std::vector<typename SolutionTransferContainer<dim>::SolutionTransferType>
  transfer_objects = container.get_transfer_objects();


  {
    TimerOutput::Scope t(*computing_timer,
                         "Problem: Adaptive mesh refinement Pt. 1");

    *pcout << std::endl
           << " Preparing coarsening and refining..." << std::endl;

    // Initiates the estimated error per cell of each entity, which
    // is to be considered
    std::vector<Vector<float>> estimated_errors_per_cell(
      container.get_error_vector_size(),
      Vector<float>(triangulation.n_active_cells()));

    // Initiates the estimated error per cell used in the refinement.
    // It is composed by the equally weighted sum of the estimated
    // error per cell of each entity to be considered
    Vector<float> estimated_error_per_cell(triangulation.n_active_cells());

    unsigned int j = 0;

    // Computes the estimated error per cell of all the pertinent
    // entities
    for (unsigned int i = 0; i < container.get_field_collection().size(); ++i)
    {
      if (entities[i].second)
      {
        KellyErrorEstimator<dim>::estimate(
          entities[i].first->get_dof_handler(),
          QGauss<dim-1>(entities[i].first->fe_degree() + 1),
          std::map<types::boundary_id, const Function<dim> *>(),
          entities[i].first->solution,
          estimated_errors_per_cell[j],
          ComponentMask(),
          nullptr,
          0,
          triangulation.locally_owned_subdomain());

        j += 1;
      }
      else
        continue;
    }

    // Reset the estimated error per cell and fills it with the
    // equally weighted sum of the estimated error per cell of each
    // entity to be considered
    estimated_error_per_cell = 0.;

    for (auto const &error_vector: estimated_errors_per_cell)
      estimated_error_per_cell.add(1.0 / container.get_error_vector_size(),
                                   error_vector);

    // Indicates which cells are to be refine/coarsen
    parallel::distributed::
    GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
                                                      estimated_error_per_cell,
                                                      cell_fraction_to_coarsen,
                                                      cell_fraction_to_refine);

    // Clear refinement flags if refinement level exceeds maximum
    if (triangulation.n_global_levels() > n_maximum_levels)
      for (auto cell: triangulation.active_cell_iterators_on_level(n_maximum_levels))
        cell->clear_refine_flag();

    // Clear coarsen flags if level decreases minimum
    for (auto cell: triangulation.active_cell_iterators_on_level(n_minimum_levels))
        cell->clear_coarsen_flag();

    // Count number of cells to be refined and coarsened
    unsigned int local_cell_counts[2] = {0, 0};
    for (auto cell: triangulation.active_cell_iterators())
        if (cell->is_locally_owned() && cell->refine_flag_set())
            local_cell_counts[0] += 1;
        else if (cell->is_locally_owned() && cell->coarsen_flag_set())
            local_cell_counts[1] += 1;

    unsigned int global_cell_counts[2];
    Utilities::MPI::sum(local_cell_counts, mpi_communicator, global_cell_counts);

    *pcout << "   Number of cells set for refinement: " << global_cell_counts[0] << std::endl
           << "   Number of cells set for coarsening: " << global_cell_counts[1] << std::endl;

    // Stores the current solutions into std::vector declared below
    // and prepare each entry for the solution transfer
    const std::vector<typename SolutionTransferContainer<dim>::TransferVectorType>
    transfer_vectors = container.get_transfer_vectors();

    AssertDimension(transfer_objects.size(), transfer_vectors.size());

    triangulation.prepare_coarsening_and_refinement();
    for (size_t i = 0; i < transfer_objects.size(); ++i)
      transfer_objects[i].prepare_for_coarsening_and_refinement(transfer_vectors[i]);

    // Execute the mesh refinement/coarsening
    *pcout << " Executing coarsening and refining..." << std::endl;
    triangulation.execute_coarsening_and_refinement();
  }

  *pcout << "   Number of global active cells:      "
         << triangulation.n_global_active_cells() << std::endl;

  std::vector<types::global_cell_index> locally_active_cells(triangulation.n_global_levels());
  for (unsigned int level = 0; level < triangulation.n_levels(); ++level)
    for (auto cell: triangulation.active_cell_iterators_on_level(level))
      if (cell->is_locally_owned())
        locally_active_cells[level] += 1;
  *pcout << "   Number of cells on each (level):    ";
  for (unsigned int level=0; level < triangulation.n_global_levels(); ++level)
  {
      *pcout << Utilities::MPI::sum(locally_active_cells[level], mpi_communicator)
             << " (" << level << ")" << ", ";
  }
  *pcout << "\b\b \n" << std::endl;

  int n_total_dofs = 0;

  // Reinitiate the entities to accomodate to the new mesh
  for (auto &entity: entities)
  {
    (entity.first)->setup_dofs();

    if (!entity.first->is_child_entity())
    {
      *pcout << std::setw(60)
             << (" Number of degrees of freedom of the \""
                + (entity.first)->name + "\" entity")
             << " = "
             << (entity.first)->n_dofs()
             << std::endl;

      n_total_dofs += entity.first->n_dofs();
    }

    (entity.first)->apply_boundary_conditions();
    (entity.first)->setup_vectors();
  }

  *pcout << std::setw(60)
         << " Number of total degrees of freedom"
         << " = "
         << n_total_dofs << std::endl << std::endl;

  {
    TimerOutput::Scope t(*computing_timer,
                         "Problem: Adaptive mesh refinement Pt. 2");

    for (size_t i = 0; i<entities.size(); ++i)
    {
      // Temporary vectors to extract the interpolated solutions back
      // into the entities
      LinearAlgebra::MPI::Vector  distributed_tmp_solution;
      LinearAlgebra::MPI::Vector  distributed_tmp_old_solution;
      LinearAlgebra::MPI::Vector  distributed_tmp_old_old_solution;

      distributed_tmp_solution.reinit(entities[i].first->distributed_vector);
      distributed_tmp_old_solution.reinit(distributed_tmp_solution);
      distributed_tmp_old_old_solution.reinit(distributed_tmp_solution);

      std::vector<LinearAlgebra::MPI::Vector *>  tmp(3);
      tmp[0] = &(distributed_tmp_solution);
      tmp[1] = &(distributed_tmp_old_solution);
      tmp[2] = &(distributed_tmp_old_old_solution);

      // Interpolate and apply constraints to the temporary vectors
      transfer_objects[i].interpolate(tmp);

      const AffineConstraints<LinearAlgebra::MPI::Vector::value_type>
      &current_constraints = entities[i].first->get_constraints();

      current_constraints.distribute(distributed_tmp_solution);
      current_constraints.distribute(distributed_tmp_old_solution);
      current_constraints.distribute(distributed_tmp_old_old_solution);

      // Pass the interpolated vectors to the fields' vector instances
      (entities[i].first)->solution          = distributed_tmp_solution;
      (entities[i].first)->old_solution      = distributed_tmp_old_solution;
      (entities[i].first)->old_old_solution  = distributed_tmp_old_old_solution;
    }
  }
}


} // namespace RMHD

template struct RMHD::SolutionTransferContainer<2>;
template struct RMHD::SolutionTransferContainer<3>;

template class RMHD::Problem<2>;
template class RMHD::Problem<3>;
