#ifndef INCLUDE_ROTATINGMHD_LINEAR_SOLVER_PARAMETERS_H_
#define INCLUDE_ROTATINGMHD_LINEAR_SOLVER_PARAMETERS_H_

#include <deal.II/base/parameter_handler.h>

#include <rotatingMHD/basic_parameters.h>

namespace RMHD
{

/*!
 * @brief Namespace containing all the structs and enum classes related
 * to the run time parameters.
 */
namespace RunTimeParameters
{

using namespace dealii;

/*!
 * @struct PreconditionerParametersBase
 *
 * @brief A structure from which all other preconditioner parameters structures
 * are derived.
 */
struct PreconditionBaseParameters
{
  /*!
   * Constructor which sets up the parameters with default values.
   */
  PreconditionBaseParameters(const std::string        &name = "ILU",
                             const PreconditionerType &type = PreconditionerType::ILU);

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(ParameterHandler &prm);

  /*!
   * @brief Method which parses the only the type parameters of
   * ::PreconditionBaseParameters from the ParameterHandler object @p prm.
   */
  static PreconditionerType parse_preconditioner_type(const ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters of ::PreconditionBaseParameters
   * from the ParameterHandler object @p prm.
   */
  void parse_parameters(const ParameterHandler &prm);

  /*!
   * @brief The type of the preconditioner used for solving the linear system.
   */
  PreconditionerType preconditioner_type;

  /*!
   * @brief The name of the preconditioner.
   */
  std::string       preconditioner_name;
};


/*!
 * @struct PreconditionRelaxationParameters
 *
 * @brief A structure containing all parameters relevant for Jacobi
 * preconditioning.
 */
struct PreconditionRelaxationParameters : PreconditionBaseParameters
{
  /*!
   * Constructor which sets up the parameters with default values.
   */
  PreconditionRelaxationParameters();

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters of the ILU preconditioner from
   * the ParameterHandler object @p prm.
   */
  void parse_parameters(const ParameterHandler &prm);

  /*!
   * @brief Method forwarding parameters to a stream object.
   *
   * @details This method does not add a `std::endl` to the stream at the end.
   *
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &stream, const PreconditionRelaxationParameters &prm);

  /*!
   * @brief The relaxation parameter in the preconditioner.
   *
   * @attention For a Jacobi preconditioner, this parameter is only meaningful if the library is
   * compiled using the Trilinos linear algebra package.
   *
   */
  double        omega;

  /*!
   * @brief A parameter specifying how large the overlap of the local matrix
   * portions on each processor in a parallel application should be.
   *
   * @attention This parameter is only meaningful if the library is compiled
   * using the Trilinos linear algebra package and if an SOR or SSOR
   * preconditioner is used.
   *
   */
  unsigned int  overlap;


  /*!
   * @brief A parameter specifying how many times the given preconditioner
   * should be applied at each call.
   *
   * @attention This parameter is only meaningful if the library is compiled
   * using the Trilinos linear algebra package.
   *
   */
  unsigned int  n_sweeps;
};

using PreconditionJacobiParameters = PreconditionRelaxationParameters;

using PreconditionSSORParameters = PreconditionRelaxationParameters;

/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const PreconditionRelaxationParameters &prm);

/*!
 * @struct PreconditionILUParameters
 *
 * @brief A structure containing all parameters relevant for the computation of
 * an incomplete LU preconditioner.
 *
 */
struct PreconditionILUParameters : PreconditionBaseParameters
{
  /*!
   * Constructor which sets up the parameters with default values.
   */
  PreconditionILUParameters();

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters of the ILU preconditioner from
   * the ParameterHandler object @p prm.
   */
  void parse_parameters(const ParameterHandler &prm);

  /*!
   * @brief Method forwarding parameters to a stream object.
   *
   * @details This method does not add a `std::endl` to the stream at the end.
   *
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &stream, const PreconditionILUParameters &prm);

  /*!
   * @brief Relative tolerance used for the perturbation of the diagonal entries
   * of the diagonal matrix which meet yield a better preconditioning. For the
   * relative tolerance \f$\beta\f$ the condition \f$\beta\geq 1\f$ must hold.
   * The suggested value is \f$\beta\approx 1.01\f$.
   *
   * @attention This parameter is only meaningful if the library is compiled
   * using the Trilinos linear algebra package.
   *
   */
  double        relative_tolerance;

  /*!
   * @brief Absolute tolerance used for the perturbation of the diagonal entries
   * of the diagonal matrix which meet yield a better preconditioning. For the
   * relative tolerance \f$\alpha\f$ the condition \f$\alpha>0\f$ must hold. The
   * suggested value is \f$10^{-5}\alpha\leq 10^{-2}\f$.
   *
   * @attention This parameter is only meaningful if the library is compiled
   * using the Trilinos linear algebra package.
   */
  double        absolute_tolerance;

  /*!
   * @brief Fill-in level of the ILU. The ILU uses only the entries of the
   * sparsity pattern of the matrix \f$ A^k\f$ where \f$ k\f$ is the fill-in
   * level.
   */
  unsigned int  fill;

  /*!
   * @brief Overlap between the different MPI processes used for computing
   * the ILU.
   *
   * @attention This parameter is only meaningful if the library is compiled
   * using the Trilinos linear algebra package.
   */
  unsigned int  overlap;
};

/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const PreconditionILUParameters &prm);


/*!
 * @struct PreconditionAMGParameters
 *
 * @brief A structure containing the common parameters used in AMG
 * preconditioning.
 */
struct PreconditionAMGParameters : PreconditionBaseParameters
{
  /*!
   * Constructor which sets up the parameters with default values.
   */
  PreconditionAMGParameters();

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters of the AMG preconditioner from
   * the ParameterHandler object @p prm.
   */
  void parse_parameters(const ParameterHandler &prm);

  /*!
   * @brief Method forwarding parameters to a stream object.
   *
   * @details This method does not add a `std::endl` to the stream at the end.
   *
   */
  template<typename Stream>
  friend Stream& operator<<(Stream &stream, const PreconditionAMGParameters &prm);

  /*
   * @brief A control parameter for AMG preconditioning using the PETSc library.
   *
   * @attention This parameter is only meaningful if the library is compiled
   * using the PETSc linear algebra package.
   *
   */
  double strong_threshold;

  /*
   * @brief A control parameter to control whether smoothed aggregation or
   * non-smoothed aggregration should be applied.
   *
   * @attention This parameter is only meaningful if the library is compiled
   * using the Trilinos linear algebra package.
   *
   */
  bool  elliptic;

  /*
   * @brief A control parameter to control whether higher order finite elements
   * are used.
   *
   * @details The parameters cannot be specified through the parameter file but
   * is set according to the polynomial degree of the finite element.
   *
   * @attention This parameter is only meaningful if the library is compiled
   * using the Trilinos linear algebra package.
   *
   */
  bool  higher_order_elements;

  /*
   * @brief A control parameter to control how many multigrid cycles should be
   * performed.
   *
   * @attention This parameter is only meaningful if the library is compiled
   * using the Trilinos linear algebra package.
   *
   */
  unsigned int  n_cycles;

  /*
   * @brief A control parameter for AMG preconditioning using the Trilinos
   * library to control how the coarsening should be performed.
   *
   * @attention This parameter is only meaningful if the library is compiled
   * using the Trilinos linear algebra package.
   *
   */
  double  aggregation_threshold;

};


/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const PreconditionAMGParameters &prm);


/*!
 * @struct LinearSolverParameters
 *
 * @brief A structure containing all parameters relevant for the solution of
 * linear systems using a Krylov subspace method.
 *
 * @todo Proper initiation of the solver_name string without constructor
 * ambiguity. The string is included in order for the stream to print
 * also the names of the solvers to the terminal. As of now one can not
 * tell one from the other (On the prm file they are properly differentiated)
 */
struct LinearSolverParameters
{
  /*!
   * Constructor which sets up the parameters with default values.
   */
  LinearSolverParameters(const std::string &name = "default");

  /*!
   * @brief Static method which declares the associated parameter to the
   * ParameterHandler object @p prm.
   */
  static void declare_parameters(ParameterHandler &prm);

  /*!
   * @brief Method which parses the parameters of the linear solver from
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
  friend Stream& operator<<(Stream &stream, const LinearSolverParameters &prm);

  /*!
   * @brief Relative tolerance.
   *
   * @details This parameter specifies the desired relative reduction of the
   * initial residual of the linear system. The initial residual is given by
   *
   * \f[
   *    r_0=|\!| \mathbf{A}\cdot\mathbf{x}_0-\mathbf{b}|\!|_2\,.
   * \f]
   *
   * If the relative tolerance is applied, the final solution of the linear
   * system is such that
   *
   * \f[
   *    |\!| \mathbf{A}\cdot\mathbf{x}-\mathbf{b}|\!|_2\leq\mathrm{tol}\,r_0\,.
   * \f]
   *
   */
  double        relative_tolerance;

  /*!
   * @brief Absolute tolerance.
   *
   * @details This parameter specifies the desired absolute value of the
   * residual of the linear system.
   *
   * If the absolute tolerance is applied, the final solution of the linear
   * system is such that
   *
   * \f[
   *    |\!| \mathbf{A}\cdot\mathbf{x}-\mathbf{b}|\!|_2\leq\mathrm{tol}\,.
   * \f]
   *
   */
  double        absolute_tolerance;

  /*!
   * @brief Maximum number of iterations to be performed by the linear solver.
   *
   * @details If the tolerance is fulfilled within the maximum number of
   * iterations, an error is thrown and the simulation is aborted. In this case,
   * the preconditioner might require some optimization.
   *
   */
  unsigned int  n_maximum_iterations;

  /*!
   * @brief Pointer to the parameter of the preconditioners
   */
  std::shared_ptr<PreconditionBaseParameters> preconditioner_parameters_ptr;

private:

  /*!
   * @brief The name of the solver.
   */
  std::string   solver_name;
};



/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 *
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const LinearSolverParameters &prm);

} // namespace RunTimeParameters

} // namespace RMHD


#endif /* INCLUDE_ROTATINGMHD_LINEAR_SOLVER_PARAMETERS_H_ */
