#ifndef INCLUDE_ROTATINGMHD_BASIC_PARAMETERS_H_
#define INCLUDE_ROTATINGMHD_BASIC_PARAMETERS_H_

#include <deal.II/base/parameter_handler.h>

#include <rotatingMHD/time_discretization.h>

namespace RMHD
{

using namespace dealii;

/*!
 * @brief Namespace containing all the structs and enum classes related
 * to the run time parameters.
 */
namespace RunTimeParameters
{

namespace internal
{
  /*!
   * @brief Prints a table row consisting of a single column with a fixed width
   * and `|` delimiters.
   */
  template<typename Stream, typename A>
  void add_line(Stream  &stream, const A line);

  /*!
   * @brief Prints a table row consisting of a two columns with a fixed width
   * and `|` delimiters.
   */
  template<typename Stream, typename A, typename B>
  void add_line(Stream  &stream, const A first_column, const B second_column);

  /*!
   * @brief Prints a table header with a fixed width and `|` delimiters.
   */
  template<typename Stream>
  void add_header(Stream  &stream);

} // internal


/*!
 * @brief Enumeration for the different types of problems.
 *
 * @details The different problems derive from the stencil
  * \f[
  * \begin{gather*}
    \nabla \cdot \bs{v}
    = 0, \quad
      \pd{\bs{v}}{t} + \bs{v} \cdot (\nabla \otimes \bs{v})
      + C_1 \bs{\Omega} \times \bs{v}
    = -C_6 \nabla \bar{p}
    + C_ 2\nabla^2 \bs{v}
    - C_3 \vartheta \bs{g}
    + C_5 (\nabla \times \bs{B}) \times \bs{B},
    \\
    \pd{\vartheta}{t} + \bs{v} \cdot \nabla \vartheta
    =
    C_4 \nabla^2 \vartheta, \\
    \nabla \cdot \bs{B} = 0, \quad
    \pd{\bs{B}}{t} = \nabla \times ( \bs{v} \times \bs{B} ) + C_5 \nabla \times ( \nabla \times \bs{B} ),    \qquad \qquad \forall (\bs{x}, t) \in \Omega \times [0, T],
    \end{gather*}
  * \f]
  * where \f$ \bs{v} \f$ is the velocity,
  * \f$ \bar{p} \f$ the modified pressure,
  * \f$ \vartheta \f$ the temperature,
  * \f$ \bs{B} \f$ the magnetic field and
  * \f$ {C_i} \f$ the dimensionless coefficients. These coefficients
  * are given by a combination of dimensionless numbers depending of
  * the problem type.
  * @note For a definition of the pertinent dimensionless numbers see
  * @ref DimensionlessNumbers
 */
enum class ProblemType
{
  /*!
   * @brief The Navier-Stokes equations in an intertial frame of reference.
   * @details Given by
  * \f[
  * \begin{gather*}
    \nabla \cdot \bs{v}
    = 0, \quad
      \pd{\bs{v}}{t} + \bs{v} \cdot (\nabla \otimes \bs{v})
    = - \nabla \bar{p}
    + \frac{1}{\Reynolds} \nabla^2 \bs{v}
    + \bs{f},
    \qquad \qquad \forall (\bs{x}, t) \in \Omega \times [0, T],
    \end{gather*}
  * \f]
  * where \f$ \bs{v} \f$ is the velocity,
  * \f$ \rho_0 \f$ the density of the reference state,
  * \f$ \bar{p} \f$ the modified pressure,
  * \f$ \Reynolds \f$ the Reynolds number and
  * \f$ \bs{f} \f$ the body force,
  */
  hydrodynamic,

  /*!
   * @brief The heat convection-difussion equation.
   * @details Given by
  * \f[
  * \begin{gather*}
    \pd{\vartheta}{t} + \bs{v} \cdot \nabla \vartheta
    =
    \frac{1}{\Peclet} \nabla^2 \vartheta + r, \qquad \qquad \forall (\bs{x}, t) \in \Omega \times [0, T],
    \end{gather*}
  * \f]
  * where \f$ \bs{v} \f$ is the velocity,
  * \f$ \vartheta \f$ the temperature,
  * \f$ \Peclet \f$ the Peclet number and
  * \f$ r \f$ the internal heat generation.
  */
  heat_convection_diffusion,

  /*!
   * @brief Boussinesq-approximation in an interial frame of reference.
   * @details Given by
  * \f[
  * \begin{gather*}
    \nabla \cdot \bs{v}
    = 0, \quad
      \pd{\bs{v}}{t} + \bs{v} \cdot (\nabla \otimes \bs{v})
    = - \nabla \bar{p}
    + \sqrt{\frac{\Prandtl}{\Rayleigh}} \nabla^2 \bs{v}
    - \vartheta \bs{g},
    \\
    \pd{\vartheta}{t} + \bs{v} \cdot \nabla \vartheta
    =
    \frac{1}{\sqrt{\Rayleigh \Prandtl}} \nabla^2 \vartheta, \qquad \qquad \forall (\bs{x}, t) \in \Omega \times [0, T],
    \end{gather*}
  * \f]
  * where \f$ \bs{v} \f$ is the velocity,
  * \f$ \bar{p} \f$ the modified pressure,
  * \f$ \Prandtl \f$ the Prandtl number,
  * \f$ \Rayleigh \f$ the Rayleigh number and
  * \f$ \vartheta \f$ the temperature.
  * @note If specifified, body forces and internal heat generation are
  * also considered.
  */
  boussinesq,

  /*!
   * @brief Boussinesq-approximation in an rotating frame of reference.
   * @details Given by
  * \f[
  * \begin{gather*}
    \nabla \cdot \bs{v}
    = 0, \quad
      \pd{\bs{v}}{t} + \bs{v} \cdot (\nabla \otimes \bs{v})
      + \frac{2}{\Ekman} \bs{\Omega} \times \bs{v}
    = -\dfrac{1}{\Ekman} \nabla \bar{p}
    + \nabla^2 \bs{v}
    - \frac{\Rayleigh}{\Prandtl} \vartheta \bs{g},
    \\
    \pd{\vartheta}{t} + \bs{v} \cdot \nabla \vartheta
    =
    \frac{1}{\Prandtl} \nabla^2 \vartheta, \qquad \qquad \forall (\bs{x}, t) \in \Omega \times [0, T],
    \end{gather*}
  * \f]
  * where \f$ \bs{v} \f$ is the velocity,
  * \f$ \bs{\Omega} \f$ the angular velocity,
  * \f$ \bar{p} \f$ the modified pressure,
  * \f$ \Ekman \f$ the Ekman number,
  * \f$ \Rayleigh \f$ the Rayleigh number,
  * \f$ \Prandtl \f$ the Prandtl number,
  * \f$ \vartheta \f$ the temperature.
  * @note If specifified, body forces and internal heat generation are
  * also considered.
  */
  rotating_boussinesq,

  /*!
   * @brief Boussinesq-approximation with electromagnetic forces in an rotating frame of reference.
   * @details Given by
  * \f[
  * \begin{gather*}
    \nabla \cdot \bs{v}
    = 0, \quad
      \pd{\bs{v}}{t} + \bs{v} \cdot (\nabla \otimes \bs{v})
      + \frac{2}{\Ekman} \bs{\Omega} \times \bs{v}
    = -\dfrac{1}{\Ekman} \nabla \bar{p}
    + \nabla^2 \bs{v}
    - \frac{\Rayleigh}{\Prandtl} \vartheta \bs{g}
    + \frac{1}{\magPrandtl}(\nabla \times \bs{B}) \times \bs{B},
    \\
    \pd{\vartheta}{t} + \bs{v} \cdot \nabla \vartheta
    =
    \frac{1}{\Prandtl} \nabla^2 \vartheta, \\
    \nabla \cdot \bs{B} = 0, \quad
    \pd{\bs{B}}{t} = \nabla \times ( \bs{v} \times \bs{B} ) + \frac{1}{\magPrandtl} \nabla \times ( \nabla \times \bs{B} ),    \qquad \qquad \forall (\bs{x}, t) \in \Omega \times [0, T],
    \end{gather*}
  * \f]
  * where \f$ \bs{v} \f$ is the velocity,
  * \f$ \bs{\Omega} \f$ the angular velocity,
  * \f$ \Ekman \f$ the Ekman number,
  * \f$ \bar{p} \f$ the modified pressure,
  * \f$ \Rayleigh \f$ the Rayleigh number,
  * \f$ \Prandtl \f$ the Prandtl number,
  * \f$ \vartheta \f$ the temperature,
  * \f$ \bs{B} \f$ the magnetic field and
  * \f$ \magPrandtl \f$ the magnetic Prandtl number.
  * @note If specifified, body forces and internal heat generation are
  * also considered.
  */
  rotating_magnetohydrodynamic
};



/*!
 * @brief Enumeration for incremental pressure-correction scheme types.
 */
enum class PressureCorrectionScheme
{
  /*!
   * @brief Standard incremental pressure-correction scheme.
   * @details The pressure update is given by
   * \f[
   * p^{k} = p^{k-1} + \phi^{k}
   * \f]
   * @note See @ref NavierStokesProjection for a complete explanation
   * of the incremental pressure-correction scheme.
   */
  standard,

  /*!
   * @brief Rotational incremental pressure-correction scheme.
   * @details The pressure update is given by
   * \f[
   * p^{k} = p^{k-1} + \phi^{k} - \nu \nabla \cdot \bs{u}^k
   * \f]
   * @note See @ref NavierStokesProjection for a complete explanation
   * of the incremental pressure-correction scheme.
   */
  rotational
};



/*!
 * @brief Enumeration for the weak form of the non-linear convective term.
 * @attention These definitions are the ones I see the most in the literature.
 * Nonetheless Volker John and Helene Dallmann define the skew-symmetric
 * and the divergence form differently.
 */
enum class ConvectiveTermWeakForm
{
  /*!
   * @brief The standard form.
   * @details Given by
   * \f[
   * \int_\Omega \bs{w} \cdot [ \bs{v} \cdot ( \nabla \otimes \bs{v})] \dint{v}
   * \f]
   * where \f$ \bs{v} \f$ and \f$ \bs{w} \f$ are the velocity and the
   * test function respectively.
   */
  standard,

  /*!
   * @brief The skew-symmetric form.
   * @details Given by
   * \f[
   * \int_\Omega \bs{w} \cdot [ \bs{v} \cdot ( \nabla \otimes \bs{v}) +
   * (\nabla \cdot \bs{v}) \bs{v}] \dint{v}
   * \f]
   * where \f$ \bs{v} \f$ and \f$ \bs{w} \f$ are the velocity and the
   * test function respectively.
   */
  skewsymmetric,

  /*!
   * @brief The divergence form.
   * @details Given by
   * \f[
   * \int_\Omega \bs{w} \cdot [ \bs{v} \cdot ( \nabla \otimes \bs{v}) +
   * \frac{1}{2}(\nabla \cdot \bs{v}) \bs{v}] \dint{v}
   * \f]
   * where \f$ \bs{v} \f$ and \f$ \bs{w} \f$ are the velocity and the
   * test function respectively.
   */
  divergence,

  /*!
   * @brief The rotational form.
   * @details Given by
   * \f[
   * \int_\Omega \bs{w} \cdot [ ( \nabla \times\bs{v}) \times \bs{v}] \dint{v}
   * \f]
   * where \f$ \bs{v} \f$ and \f$ \bs{w} \f$ are the velocity and the
   * test function respectively.
   * @note This form modifies the pressure, *i. e.*,
   * \f[
   * \bar{p} = p + \frac{1}{2} \bs{v} \cdot \bs{v}.
   * \f]
   */
  rotational
};



/*!
 * @brief Enumeration for the type of the temporal discretization of the
 * non-linear convective term.
 */
enum class ConvectiveTermTimeDiscretization
{
  /*!
   * @brief Semi-implicit treatment in time of the convective term, *i. e.*,
   * \f[
   * \bs{v}^{\star,k} \cdot
   * (\nabla \otimes\bs{\phi}^{k}),
   * \f]
   * where \f$\bs{v}^{\star, k}\f$ is the second order Taylor extrapolation
   * of the velocity at \f$t^k \f$ and \f$ \bs{\phi}^{k} \f$ is the
   * field being transported by the velocity.
   * @note The semi-implicit treatment for non-standard weak forms of
   * the convective terms is given
   * by
   * \f[
   * \bs{v}^{k} \cdot ( \nabla \otimes \bs{v}^{\star,k}) +
   * \frac{1}{2}(\nabla \cdot \bs{v}^{\star,k}) \bs{v}^{k},
   * \qquad
   * \bs{v}^{k} \cdot ( \nabla \otimes \bs{v}^{\star,k}) +
   * (\nabla \cdot \bs{v}^{\star,k}) \bs{v}^{k},
   * \qquad
   * ( \nabla \times\bs{v}^{\star,k}) \times \bs{v}^{k},
   * \f]
   * for the skew-symmetric, divergence and rotational form respectively.
   */
  semi_implicit,

  /*!
   * @brief Explicit treatment in time of the convective term, *i. e.*,
   * \f[
   * \sum_{j = 1} \beta_j
   * \bs{v}^{k-j} \cdot
   * (\nabla \otimes\bs{\phi}^{k-j}),
   * \f]
   * where \f$ \beta_j \f$ are time discretization coefficientes,
   * \f$\bs{v}^{k-j}\f$ the previous values of the velocity field and
   * \f$ \bs{\phi}^{k-j} \f$ the previous values of the
   * field being transported by the velocity.
   * @attention Why fully_explicit and not just explicit?
   */
  fully_explicit
};

/*!
 * @brief Enumeration for the type of the preconditioner to be used.
 */
enum class PreconditionerType
{
  /*!
   * @brief Incomplete LU decomposition preconditioning.
   */
  ILU,

  /*!
   * @brief Geometric multigrid preconditioning.
   *
   * @attention This is not implemented yet but may be helpful once the
   * MatrixFree framework is used.
   */
  GMG,

  /*!
   * @brief Algebraic multigrid preconditioning.
   */
  AMG,

  /*!
   * @brief Jacobi preconditioning.
   */
  Jacobi,

  /*!
   * @brief Symmetric Sucessive overrelaxation preconditioning. The system
   * matrix must be symmetric to apply this preconditioner.
   */
  SSOR

};


/*!
 * @struct OutputControlParameters
 * @brief @ref OutputControlParameters contains parameters which are
 * related to the graphical and terminal output.
 */
struct OutputControlParameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  OutputControlParameters();

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
                            const OutputControlParameters &prm);

  /*!
   * @brief The frequency at which a graphical output file (vtk output) is
   * written to the file system.
   */
  unsigned int  graphical_output_frequency;

  /*!
   * @brief The frequency at which diagnostics are written the terminal.
   */
  unsigned int  terminal_output_frequency;

  /*!
   * @brief The frequency at which postprocessing of the results is performed.
   */
  unsigned int  postprocessing_frequency;

  /*!
   * @brief Directory where the output should be written.
   */
  std::string   output_directory;
};

/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const OutputControlParameters &prm);


/*!
 * @struct SpatialDiscretizationParameters
 *
 * @brief @ref SpatialDiscretizationParameters contains parameters which are
 * related to the adaptive refinement of the mesh.
 */
struct SpatialDiscretizationParameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  SpatialDiscretizationParameters();

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
                            const SpatialDiscretizationParameters &prm);

  /*!
   * @brief Boolean flag to enable or disable adaptive mesh refinement.
   */
  bool          adaptive_mesh_refinement;

  /*!
   * @brief The frequency at which an adaptive mesh refinement is performed.
   */
  unsigned int  adaptive_mesh_refinement_frequency;

  /*!
   * @brief The upper fraction of the total number of cells set to
   * coarsen.
   */
  double        cell_fraction_to_coarsen;

  /*!
   * @brief The lower fraction of the total number of cells set to
   * refine.
   */
  double        cell_fraction_to_refine;

  /*!
   * @brief The number of maximum levels of the mesh. This parameter prohibits
   * a further refinement of the mesh.
   */
  unsigned int  n_maximum_levels;

  /*!
   * @brief The number of minimum levels of the mesh. This parameter prohibits
   * a further coarsening of the mesh.
   */
  unsigned int  n_minimum_levels;

  /*!
   * @brief The number of adaptive initial refinement steps. This parameter
   * determines the number of refinements which are based on the spatial structure
   * of the initial condition.
   */
  unsigned int  n_initial_adaptive_refinements;

  /*!
   * @brief The number of global initial refinement steps.
   */
  unsigned int  n_initial_global_refinements;

   /*!
   * @brief The number of initial refinement steps of cells at the boundary.
   */
  unsigned int  n_initial_boundary_refinements;
};



/*!
 * @brief Method forwarding parameters to a stream object.
 *
 * @details This method does not add a `std::endl` to the stream at the end.
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const SpatialDiscretizationParameters &prm);





/*!
 * @struct ProblemBaseParameters
 */
struct ProblemBaseParameters : public OutputControlParameters
{
  /*!
   * @brief Constructor which sets up the parameters with default values.
   */
  ProblemBaseParameters();

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
                            const ProblemBaseParameters &prm);

  /*!
   * @brief Spatial dimension of the problem.
   */
  unsigned int                                dim;

  /*!
   * @brief Polynomial degree of the mapping.
   */
  unsigned int                                mapping_degree;

  /*!
   * @brief Boolean indicating whether the mapping should be applied for
   * the interior cells as well.
   */
  bool                                        mapping_interior_cells;

  /*!
   * @brief Boolean flag to enable verbose output on the terminal.
   */
  bool                                        verbose;

  /*!
   * @brief Parameters of the adaptive mesh refinement.
   */
  SpatialDiscretizationParameters             spatial_discretization_parameters;

  /*!
   * @brief Parameters of the time stepping scheme.
   */
  TimeDiscretization::TimeDiscretizationParameters  time_discretization_parameters;
};

/*!
 * @brief Method forwarding parameters to a stream object.
 */
template<typename Stream>
Stream& operator<<(Stream &stream, const ProblemBaseParameters &prm);


} // namespace RunTimeParameters

} // namespace RMHD

#endif /* INCLUDE_ROTATINGMHD_BASIC_PARAMETERS_H_ */
