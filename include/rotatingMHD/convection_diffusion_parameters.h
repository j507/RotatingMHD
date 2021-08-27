#ifndef INCLUDE_ROTATINGMHD_CONVECTION_DIFFUSION_PARAMETERS_H_
#define INCLUDE_ROTATINGMHD_CONVECTION_DIFFUSION_PARAMETERS_H_

#include <deal.II/base/parameter_handler.h>

#include <rotatingMHD/basic_parameters.h>
#include <rotatingMHD/linear_solver_parameters.h>

#include <fstream>

namespace RMHD
{
/*!
 * @struct ConvectionDiffusionSolverParameters
 *
 * @brief A structure containing all the parameters of the heat
 * equation solver.
 */
struct ConvectionDiffusionSolverParameters
{
  /*!
   * Constructor which sets up the parameters with default values.
   */
  ConvectionDiffusionSolverParameters();

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
   *
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &stream, const ConvectionDiffusionSolverParameters &prm);

  /*!
   * @brief Enumerator controlling which time discretization of the
   * convective term is to be implemented
   */
  RunTimeParameters::ConvectiveTermTimeDiscretization  time_discretization;

    /*!
   * @brief The factor multiplying the diffusion term.
   */
  double  equation_coefficient;

  /*!
   * @brief The parameters for the linear solver.
   */
  RunTimeParameters::LinearSolverParameters linear_solver;

  /*!
   * @brief Specifies the frequency of the update of the solver's
   * preconditioner.
   */
  unsigned int  preconditioner_update_frequency;

  /*!
   * @brief Boolean flag to enable verbose output on the terminal.
   */
  bool  verbose;
};



/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 *
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const ConvectionDiffusionSolverParameters &prm);



/*!
 * @struct ConvectionDiffusionParameters
 */
struct ConvectionDiffusionParameters
    : public RunTimeParameters::ProblemBaseParameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  ConvectionDiffusionParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  ConvectionDiffusionParameters(const std::string &parameter_filename);

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
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &stream,
                            const ConvectionDiffusionParameters &prm);

  /*!
   * @brief The polynomial degree of the finite element.
   */
  unsigned int  fe_degree;

  /*!
   * @brief The Peclet number of the problem.
   */
  double        peclet_number;

  /*!
   * @brief Parameters of the convection diffusion solver.
   */
  ConvectionDiffusionSolverParameters solver_parameters;
};



/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const ConvectionDiffusionParameters &prm);

}  // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_CONVECTION_DIFFUSION_PARAMETERS_H_ */
