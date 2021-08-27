#ifndef INCLUDE_ROTATINGMHD_CONVERGENCE_TEST_H_
#define INCLUDE_ROTATINGMHD_CONVERGENCE_TEST_H_

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <string>

namespace RMHD
{

namespace ConvergenceTest
{

using namespace dealii;

/*!
 * @brief Enumeration for convergence test type.
 */
enum class Type
{
  /*!
   * @brief Spatial convergence test.
   *
   * @details Test to study the spatial discretization dependence of
   * convergence for a given problem.
   *
   * @note Spatial convergence tests should be performed with a fine
   * time discretization, *i. e.*, a small enough time step.
   */
  spatial = 0x0001,

  /*!
   * @brief Temporal convergence test.
   *
   * @details Test to study the temporal discretization dependence of
   * convergence for a given problem.
   *
   * @note Temporal convergence tests should be performed with a fine
   * spatial discretization, *i. e.*, a triangulation with small enough cells.
   */
  temporal = 0x0002,

  /*!
   * @brief Spatio-temporal convergence test.
   *
   * @details Test to study the dependency on the temporal and the spatial
   * discretization for a given problem.
   *
   * @note Convergence rates are not computed.
   */
  spatio_temporal = spatial|temporal
};

/*!
 * @struct ConvergenceTestParameters
 *
 * @brief @ref ConvergenceTestParameters contains parameters which are
 * related to convergence tests.
 */
struct Parameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  Parameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  Parameters(const std::string &parameter_filename);

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters from the ParameterHandler
   * object @p prm.
   */
  void parse_parameters(ParameterHandler &prm);

  /*!
   * @brief Method forwarding parameters to a stream object.
   *
   * @details This method does not add a `std::endl` to the stream at the end.
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &, const Parameters &);

  /*!
   * @brief The type of convergence test (spatial or temporal).
   */
  Type          type;

  /*!
   * Number of spatial convergence cycles.
   */
  unsigned int  n_spatial_cycles;

  /*!
   * @brief Factor \f$ s \f$ of the reduction of the timestep between two
   * subsequent levels, *i. e.*, \f$ \Delta t_{l+1} = s \Delta t_l\f$.
   *
   * @details The factor \f$ s \f$ must be positive and less than unity.
   */
  double        step_size_reduction_factor;

  /*!
   * @brief Number of temporal convergence cycles.
   */
  unsigned int  n_temporal_cycles;
};

/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 */
template<typename Stream>
Stream& operator<<(Stream &, const Parameters &);


/*!
 * @class ConvergenceResults
 *
 * @brief @ref ConvergenceResults is a book-keeping class for the errors of a
 * convergence test.
 *
 */
class ConvergenceResults
{

public:
  using NormType = VectorTools::NormType;

  /*
   * @brief Default constructor specifying the type of the convergence test.
   */
  ConvergenceResults(const Type type = Type::temporal);

  /*!
   * @brief Add errors in @p error_map to the convergence table. This variant adds
   * the number of DoFs, the cell diameter and the size of the timestep to the
   * convergence table.
   */
  template <int dim, int spacedim>
  void update
  (const std::map<NormType, double>  &error_map,
   const DoFHandler<dim, spacedim>   &dof_handler,
   const double                       time_step = std::numeric_limits<double>::lowest());

  /*!
   * @brief Add errors in @p error_map to the convergence table. This variant adds
   * the size of the timestep and optionally the spatial discretization parameters.
   */
  void update
  (const std::map<NormType, double> &error_map,
   const double                     time_step = std::numeric_limits<double>::lowest(),
   const types::global_dof_index    n_dofs = numbers::invalid_dof_index,
   const types::global_cell_index   n_cells = numbers::invalid_coarse_cell_id,
   unsigned int                     n_levels = static_cast<unsigned int>(-1),
   const double                     h_max = std::numeric_limits<double>::max());

  /*!
   * @brief Add errors in @p error_map to the convergence table. This variant
   * optionally adds the spatial discretization parameters and
   * the size of the timestep.
   */
  void update
  (const std::map<NormType, double> &error_map,
   const types::global_dof_index    n_dofs = numbers::invalid_dof_index,
   const types::global_cell_index   n_cells = numbers::invalid_coarse_cell_id,
   unsigned int                     n_levels = static_cast<unsigned int>(-1),
   const double                     h_max = std::numeric_limits<double>::max(),
   const double                     time_step = std::numeric_limits<double>::lowest());

  Type  get_type() const;

  /*!
   * @brief Output of the convergence table to a stream object,
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &, ConvergenceResults &);

  /*!
   * @brief Save results of convergence test to a text file using Org-mode formatting.
   */
  void  write_text(std::ostream &file);

private:
  /*!
   * @brief Method which formats the columns of the convergence table.
   *
   * @details The column are printed in scientific notation with a precision of
   * two digits. The convergence rates are also evaluated.
   *
   */
  void format_columns();

  /*!
   * @brief Type of convergence test which is performed.
   *
   * @details The column for computing the convergence rates is selected according
   * to this variable.
   */
  const Type type;

  /*!
   * @brief Number of rows added to the convergence table..
   */
  unsigned int n_rows;

  /*!
   * @brief Convergence table which stores the error norms and related data.
   */
  ConvergenceTable  table;

  /*!
   * @brief Flag indicating whether the number of dofs was specified in
   * the last cycle.
   */
  bool n_dofs_specified{false};

  /*!
   * @brief Flag indicating whether the number of cells was specified in
   * the last cycle.
   */
  bool n_cells_specified{false};

  /*!
   * @brief Flag indicating whether the number of refinements was specified in
   * the last cycle.
   */
  bool n_levels_specified{false};

  /*!
   * @brief Flag indicating whether the characteristic cell diameter was specified
   * in the last cycle.
   */
  bool h_max_specified{false};

  /*!
   * @brief Flag indicating whether the size of the timestep was specified in
   * the last cycle.
   */
  bool time_step_specified{false};

  /*!
   * @brief Flag indicating whether the L2 error norm was specified in the last
   * cycle.
   */
  bool L2_error_specified{false};

  /*!
   * @brief Flag indicating whether the H1 error norm was specified in the last
   * cycle.
   */
  bool H1_error_specified{false};

  /*!
   * @brief Flag indicating whether the infinity error norm was specified in the
   * last cycle.
   */
  bool Linfty_error_specified{false};

};

inline Type ConvergenceResults::get_type() const
{
  return (type);
}

}  // namespace ConvergenceTest

} // namespace RMHD

#endif /*INCLUDE_ROTATINGMHD_CONVERGENCE_TEST_H_*/
