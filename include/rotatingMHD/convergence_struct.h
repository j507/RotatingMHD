#ifndef INCLUDE_ROTATINGMHD_CONVERGENCE_STRUCT_H_
#define INCLUDE_ROTATINGMHD_CONVERGENCE_STRUCT_H_

#include <deal.II/base/convergence_table.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/dofs/dof_handler.h>
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
enum class ConvergenceTestType
{
  /*!
   * @brief Spatial convergence test.
   * @details Test to study the spatial discretization dependence of
   * convergence for a given problem.
   * @note Spatial convergence tests should be performed with a fine
   * time discretization, *i. e.*, a small enough time step.
   */
  spatial = 0x0001,

  /*!
   * @brief Temporal convergence test.
   * @details Test to study the temporal discretization dependence of
   * convergence for a given problem.
   * @note Temporal convergence tests should be performed with a fine
   * spatial discretization, *i. e.*, a triangulation with small enough cells.
   */
  temporal = 0x0002,

  spatio_temporal = spatial|temporal
};


/*!
 * @struct ConvergenceTestParameters
 *
 * @brief @ref ConvergenceTestParameters contains parameters which are
 * related to convergence tests.
 */
struct ConvergenceTestParameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  ConvergenceTestParameters();

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
  friend Stream& operator<<(Stream &stream,
                            const ConvergenceTestParameters &prm);

  /*!
   * @brief The type of convergence test (spatial or temporal).
   */
  ConvergenceTestType test_type;

  /*!
   * Number of spatial convergence cycles.
   */
  unsigned int        n_spatial_cycles;

  /*!
   * @brief Factor \f$ s \f$ of the reduction of the timestep between two
   * subsequent levels, *i. e.*, \f$ \Delta t_{l+1} = s \Delta t_l\f$.
   *
   * @details The factor \f$ s \f$ must be positive and less than unity.
   */
  double              step_size_reduction_factor;

  /*!
   * @brief Number of temporal convergence cycles.
   */
  unsigned int        n_temporal_cycles;
};

/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const ConvergenceTestParameters &prm);


class ConvergenceTestData
{

public:

  ConvergenceTestData(const ConvergenceTestType &type = ConvergenceTestType::temporal);

  template <int dim, int spacedim>
  void update_table
  (const DoFHandler<dim, spacedim>  &dof_handler,
   const double           time_step,
   const std::map<typename VectorTools::NormType, const double> &error_map);

  template <int dim, int spacedim>
  void update_table
  (const DoFHandler<dim, spacedim>  &dof_handler,
   const std::map<typename VectorTools::NormType,const double> &error_map);

  void update_table
  (const double time_step,
   const std::map<typename VectorTools::NormType, const double> &error_map);

  /*!
   * @brief Output of the convergence table to a stream object,
   */
  template<typename Stream>
  void print_data(Stream &stream);

  bool save(const std::string &file_name);

private:

  const ConvergenceTestType type;

  unsigned int level;

  ConvergenceTable  table;

  void format_columns();
};

} // namespace ConvergenceTest

} // namespace RMHD

#endif /*INCLUDE_ROTATINGMHD_CONVERGENCE_STRUCT_H_*/
