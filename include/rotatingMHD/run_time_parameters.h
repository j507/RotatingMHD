#ifndef INCLUDE_ROTATINGMHD_RUN_TIME_PARAMETERS_H_
#define INCLUDE_ROTATINGMHD_RUN_TIME_PARAMETERS_H_

#include <deal.II/base/parameter_handler.h>

#include <rotatingMHD/basic_parameters.h>
#include <rotatingMHD/convergence_test.h>
#include <rotatingMHD/linear_solver_parameters.h>
#include <rotatingMHD/time_discretization.h>
#include <rotatingMHD/convection_diffusion_parameters.h>

#include <memory>

namespace RMHD
{

/*!
 * @brief Namespace containing all the structs and enum classes related
 * to the run time parameters.
 */
namespace RunTimeParameters
{

/*!
 * @struct DimensionlessNumbers
 *
 * @brief @ref DimensionlessNumbers contains all the dimensionless
 * numbers relevant to the different problem types.
 */
struct DimensionlessNumbers
{
  /*!
   * Constructor which sets up the parameters with default values.
   */
  DimensionlessNumbers();

  /*!
   * @brief Static method which declares the all dimensionless numbers to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(ParameterHandler &prm);

  /*!
   * @brief Static method which declares the dimensionless numbers associated
   * with the ProblemType @problem_type to the ParameterHandler object @p prm.
   */
  static void declare_parameters(ParameterHandler &prm,
                                 const ProblemType problem_type);

  /*!
   * @brief Method which parses the dimensionless numbers from
   * the ParameterHandler object @p prm.
   */
  void parse_parameters(ParameterHandler &prm);

  /*!
   * @brief Method forwarding parameters to a stream object.
   *
   * @details This method does not add a `std::endl` to the stream at the end.
   *
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &stream, const DimensionlessNumbers &prm);

  /*!
   * @brief The Reynolds number
   * @details Defined as
   * \f[
   * \Reynolds = \frac{u D}{\nu},
   * \f]
   * where \f$ u \f$ is the characteristic velocity, \f$ D \f$ the
   * characteristic length and \f$ \nu \f$ the kinematic viscosity.
   */
  double  Re;

  /*!
   * @brief The Prandtl number
   * @details Defined as
   * \f[
   * \Prandtl = \frac{\nu}{\kappa},
   * \f]
   * where \f$ \nu \f$ is the kinematic viscosity and \f$ \kappa \f$ the
   * thermal diffusivity.
   */
  double  Pr;

  /*!
   * @brief The Peclet number
   * @details Defined as
   * \f[
   * \Peclet = \frac{u D}{\kappa},
   * \f]
   * where \f$ u \f$ is the characteristic velocity, \f$ D \f$ the
   * characteristic length and \f$ \kappa \f$ the thermal diffusivity.
   */
  double  Pe;

  /*!
   * @brief The Rayleigh number
   * @details Defined as
   * \f[
   * \Rayleigh = \frac{\alpha g D^3}{\nu \kappa},
   * \f]
   * where \f$ \alpha \f$ is the thermal expansion coefficients,
   * \f$ g \f$ the reference gravity magnitude,
   * \f$ D \f$ the characteristic length,
   * \f$ \nu \f$ the kinematic viscosity and
   * \f$ \kappa \f$ the thermal diffusivity.
   */
  double  Ra;

  /*!
   * @brief The Ekman number
   * @details Defined as
   * \f[
   * \Ekman = \frac{\nu}{\Omega D^2},
   * \f]
   * where \f$ nu \f$ is the kinematic viscosity, \f$ \Omega \f$ the
   * characteristic angular velocity and \f$ D \f$ the characteristic
   * length.
   */
  double  Ek;

  /*!
   * @brief The magnetic Prandtl number
   * @details Defined as
   * \f[
   * \magPrandtl = \frac{\nu}{\eta},
   * \f]
   * where \f$ nu \f$ is the kinematic viscosity and
   * \f$ \eta \f$ the magnetic diffusivity.
   */
  double  Pm;

private:
  /*!
   * @brief The problem type
   * @details Its inclusion in @ref DimensionlessNumbers is for the
   * sole purpose to control the stream output, effectively printing only
   * the relevant numbers.
   */
  ProblemType problem_type;
};



/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 *
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const DimensionlessNumbers &prm);



/*!
 * @struct NavierStokesParameters
 *
 * @brief A structure containing all the parameters of the Navier-Stokes
 * solver.
 */
struct NavierStokesParameters
{
  /*!
   * Constructor which sets up the parameters with default values.
   */
  NavierStokesParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  NavierStokesParameters(const std::string &parameter_filename);

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
  friend Stream& operator<<(Stream &stream, const NavierStokesParameters &prm);


  /*!
   * @brief Enumerator controlling the incremental pressure-correction
   * scheme is to be implemented.
   */
  PressureCorrectionScheme          pressure_correction_scheme;

  /*!
   * @brief Enumerator controlling which weak form of the convective
   * term is to be implemented
   */
  ConvectiveTermWeakForm            convective_term_weak_form;

  /*!
   * @brief Enumerator controlling which time discretization of the
   * convective term is to be implemented
   */
  ConvectiveTermTimeDiscretization  convective_term_time_discretization;

  /*!
   * @brief The factor multiplying the Coriolis acceleration.
   */
  double                            C1;

    /*!
   * @brief The factor multiplying the velocity's laplacian.
   */
  double                            C2;

    /*!
   * @brief The factor multiplying the buoyancy term.
   */
  double                            C3;

    /*!
   * @brief The factor multiplying the electromagnetic force.
   */
  double                            C5;

    /*!
   * @brief The factor multiplying the pressure gradient.
   * @attention This factor is only introduced to replicate
   * Christensen's benchmark. They use a different scaling.
   */
  double                            C6;

  /*!
   * @brief The parameters for the linear solver used in the
   * diffusion step.
   */
  LinearSolverParameters            diffusion_step_solver_parameters;

  /*!
   * @brief The parameters for the linear solver used in the
   * diffusion step.
   */
  LinearSolverParameters            projection_step_solver_parameters;

  /*!
   * @brief The parameters for the linear solver used in the
   * diffusion step.
   */
  LinearSolverParameters            correction_step_solver_parameters;

  /*!
   * @brief The parameters for the linear solver used in the
   * poisson pre-step.
   */
  LinearSolverParameters            poisson_prestep_solver_parameters;

  /*!
   * @brief Specifies the frequency of the update of the diffusion
   * preconditioner.
   */
  unsigned int                      preconditioner_update_frequency;

  /*!
   * @brief Boolean flag to enable verbose output on the terminal.
   */
  bool                              verbose;
};



/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 *
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const NavierStokesParameters &prm);







/*!
 * @struct HydrodynamicProblemParameters
 */
struct HydrodynamicProblemParameters
    : public ProblemBaseParameters,
      public DimensionlessNumbers
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  HydrodynamicProblemParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  HydrodynamicProblemParameters(const std::string &parameter_filename);

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
                            const HydrodynamicProblemParameters &prm);

  /*!
   * @brief Problem type.
   */
  const ProblemType                           problem_type;

  /*!
   * @brief The polynomial degree of the pressure's finite element.
   */
  unsigned int                                fe_degree_pressure;

  /*!
   * @brief The polynomial degree of the velocity's finite element.
   */
  unsigned int                                fe_degree_velocity;

  /*!
   * @brief Parameters of the Navier-Stokes solver.
   */
  NavierStokesParameters                      navier_stokes_parameters;
};



/*!
 * @brief Method forwarding parameters to a stream object.
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const HydrodynamicProblemParameters &prm);



/*!
 * @struct BoussinesqProblemParameters
 */
struct BoussinesqProblemParameters
    : public ProblemBaseParameters,
      public DimensionlessNumbers
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  BoussinesqProblemParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   */
  BoussinesqProblemParameters(const std::string &parameter_filename);

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
                            const HydrodynamicProblemParameters &prm);

  /*!
   * @brief Problem type.
   */
  const ProblemType                           problem_type;

  /*!
   * @brief The polynomial degree of the pressure's finite element.
   */
  unsigned int                                fe_degree_pressure;

  /*!
   * @brief The polynomial degree of the velocity's finite element.
   */
  unsigned int                                fe_degree_velocity;

  /*!
   * @brief The polynomial degree of the temperature's finite element.
   */
  unsigned int                                fe_degree_temperature;

  /*!
   * @brief Parameters of the Navier-Stokes solver.
   */
  NavierStokesParameters                      navier_stokes_parameters;

  /*!
   * @brief Parameters of the heat equation solver.
   */
  ConvectionDiffusionSolverParameters         heat_equation_parameters;

};



/*!
 * @brief Method forwarding parameters to a stream object.
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const BoussinesqProblemParameters &prm);


/*!
 * @struct ProblemParameters
 */
struct ProblemParameters
    : public ProblemBaseParameters,
      public DimensionlessNumbers
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  ProblemParameters();

  /*!
   * @brief Constructor which sets up the parameters as specified in the
   * parameter file with the filename @p parameter_filename.
   * @attention I do not like the flag. It was my quick fix in order to
   * print either the refinement parameters or the convergence test
   * parameters. As when one is needed the other is not.
   */
  ProblemParameters(const std::string &parameter_filename,
                    const bool        flag = false);

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
  friend Stream& operator<<(Stream &stream, const ProblemParameters &prm);

  /*!
   * @brief Problem type.
   */
  ProblemType                                 problem_type;

  /*!
   * @brief Spatial dimension of the problem.
   */
  unsigned int                                dim;

  /*!
   * @brief Polynomial degree of the mapping.
   */
  unsigned int                                mapping_degree;

  /*!
   * @brief Boolean indicating if the mapping is to be assigned to
   * the interior cells too.
   */
  bool                                        mapping_interior_cells;

  /*!
   * @brief The polynomial degree of the pressure's finite element.
   */
  unsigned int                                fe_degree_pressure;

  /*!
   * @brief The polynomial degree of the velocity's finite element.
   */
  unsigned int                                fe_degree_velocity;

  /*!
   * @brief The polynomial degree of the temperature's finite element.
   */
  unsigned int                                fe_degree_temperature;

  /*!
   * @brief Boolean flag to enable verbose output on the terminal.
   */
  bool                                        verbose;

  /*!
   * @brief Parameters of the convergence test.
   */
  ConvergenceTest::ConvergenceTestParameters  convergence_test_parameters;

  /*!
   * @brief Parameters of the Navier-Stokes solver.
   */
  NavierStokesParameters                      navier_stokes_parameters;

  /*!
   * @brief Parameters of the heat equation solver.
   */
  ConvectionDiffusionSolverParameters         heat_equation_parameters;

private:

  bool                                        flag_convergence_test;
};

/*!
 * @brief Method forwarding parameters to a stream object.
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const ProblemParameters &prm);

} // namespace RunTimeParameters

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_RUN_TIME_PARAMETERS_H_ */
