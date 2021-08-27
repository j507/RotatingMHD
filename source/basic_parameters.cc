#include <rotatingMHD/basic_parameters.h>

#include <deal.II/base/conditional_ostream.h>

#include <iomanip>
#include <iostream>

namespace RMHD
{

/*!
 * @brief Namespace containing all the structs and enum classes related
 * to the run time parameters.
 */
namespace RunTimeParameters
{

namespace internal
{

constexpr char header[] = "+------------------------------------------+"
                      "----------------------+";

constexpr size_t column_width[2] ={ 40, 20 };

constexpr size_t line_width = 63;

template<typename Stream, typename A>
void add_line(Stream  &stream, const A line)
{
  stream << "| "
         << std::setw(line_width)
         << line
         << " |"
         << std::endl;
}

template<typename Stream, typename A, typename B>
void add_line(Stream  &stream, const A first_column, const B second_column)
{
  stream << "| "
         << std::setw(column_width[0]) << first_column
         << " | "
         << std::setw(column_width[1]) << second_column
         << " |"
         << std::endl;
}

template<typename Stream>
void add_header(Stream  &stream)
{
  stream << std::left << header << std::endl;
}

} // internal

SpatialDiscretizationParameters::SpatialDiscretizationParameters()
:
adaptive_mesh_refinement(false),
adaptive_mesh_refinement_frequency(100),
cell_fraction_to_coarsen(0.30),
cell_fraction_to_refine(0.03),
n_maximum_levels(5),
n_minimum_levels(1),
n_initial_adaptive_refinements(0),
n_initial_global_refinements(0),
n_initial_boundary_refinements(0)
{}



void SpatialDiscretizationParameters::declare_parameters(ParameterHandler &prm)
{

  prm.enter_subsection("Refinement control parameters");
  {
    prm.declare_entry("Adaptive mesh refinement",
                      "false",
                      Patterns::Bool());

    prm.declare_entry("Adaptive mesh refinement frequency",
                      "100",
                      Patterns::Integer(1));

    prm.declare_entry("Fraction of cells set to coarsen",
                      "0.3",
                      Patterns::Double(0));

    prm.declare_entry("Fraction of cells set to refine",
                      "0.03",
                      Patterns::Double(0));

    prm.declare_entry("Maximum number of levels",
                      "5",
                      Patterns::Integer(1));

    prm.declare_entry("Minimum number of levels",
                      "0",
                      Patterns::Integer(0));

    prm.declare_entry("Number of initial global refinements",
                      "0",
                      Patterns::Integer(0));

    prm.declare_entry("Number of initial adaptive refinements",
                      "0",
                      Patterns::Integer(0));

    prm.declare_entry("Number of initial boundary refinements",
                      "0",
                      Patterns::Integer(0));
  }
  prm.leave_subsection();
}



void SpatialDiscretizationParameters::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Refinement control parameters");
  {
    n_maximum_levels = prm.get_integer("Maximum number of levels");

    adaptive_mesh_refinement = prm.get_bool("Adaptive mesh refinement");

    n_initial_global_refinements = prm.get_integer("Number of initial global refinements");

    n_initial_adaptive_refinements = prm.get_integer("Number of initial adaptive refinements");

    n_initial_boundary_refinements = prm.get_integer("Number of initial boundary refinements");

    if (adaptive_mesh_refinement)
    {
      n_minimum_levels = prm.get_integer("Minimum number of levels");
      AssertThrow(n_minimum_levels > 0,
                  ExcMessage("Minimum number of levels must be larger than zero."));
      AssertThrow(n_minimum_levels <= n_maximum_levels ,
                  ExcMessage("Maximum number of levels must be larger equal "
                             "than the minimum number of levels."));

      adaptive_mesh_refinement_frequency = prm.get_integer("Adaptive mesh refinement frequency");

      cell_fraction_to_coarsen = prm.get_double("Fraction of cells set to coarsen");

      cell_fraction_to_refine = prm.get_double("Fraction of cells set to refine");

      const double total_cell_fraction_to_modify =
        cell_fraction_to_coarsen + cell_fraction_to_refine;

      AssertThrow(cell_fraction_to_coarsen >= 0.0,
                  ExcLowerRangeType<double>(cell_fraction_to_coarsen, 0));

      AssertThrow(cell_fraction_to_refine >= 0.0,
                  ExcLowerRangeType<double>(cell_fraction_to_refine, 0));

      AssertThrow(1.0 > total_cell_fraction_to_modify,
                  ExcMessage("The sum of the top and bottom fractions to "
                             "coarsen and refine may not exceed 1.0"));
    }

    n_initial_adaptive_refinements = prm.get_integer("Number of initial adaptive refinements");

    n_initial_boundary_refinements = prm.get_integer("Number of initial boundary refinements");

    const unsigned int n_initial_refinements
    = n_initial_global_refinements + n_initial_adaptive_refinements
    + n_initial_boundary_refinements;

    if (adaptive_mesh_refinement)
      AssertThrow(n_minimum_levels <= n_initial_refinements,
                  ExcMessage("Number of initial refinements must be larger "
                             "equal than the minimum number of levels."));
  }
  prm.leave_subsection();
}



template<typename Stream>
Stream& operator<<(Stream &stream, const SpatialDiscretizationParameters &prm)
{
  internal::add_header(stream);
  internal::add_line(stream, "Refinement control parameters");
  internal::add_header(stream);

  internal::add_line(stream,
                     "Adaptive mesh refinement",
                     (prm.adaptive_mesh_refinement ? "True": "False"));
  if (prm.adaptive_mesh_refinement)
  {
    internal::add_line(stream,
                       "Adapt. mesh refinement frequency",
                       prm.adaptive_mesh_refinement_frequency);
    internal::add_line(stream,
                       "Fraction of cells set to coarsen", prm.cell_fraction_to_coarsen);
    internal::add_line(stream,
                       "Fraction of cells set to refine",
                       prm.cell_fraction_to_refine);
    internal::add_line(stream,
                       "Maximum number of levels", prm.n_maximum_levels);
    internal::add_line(stream,
                       "Minimum number of levels", prm.n_minimum_levels);
  }
  internal::add_line(stream,
                     "Number of initial adapt. refinements",
                     prm.n_initial_adaptive_refinements);
  internal::add_line(stream,
                     "Number of initial global refinements",
                     prm.n_initial_global_refinements);
  internal::add_line(stream,
                     "Number of initial boundary refinements",
                     prm.n_initial_boundary_refinements);

  internal::add_header(stream);

  return (stream);
}



OutputControlParameters::OutputControlParameters()
:
graphical_output_frequency(100),
terminal_output_frequency(100),
output_directory("./")
{}



void OutputControlParameters::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Output control parameters");
  {
    prm.declare_entry("Graphical output frequency",
                      "100",
                      Patterns::Integer(1));

    prm.declare_entry("Terminal output frequency",
                      "100",
                      Patterns::Integer(1));

    prm.declare_entry("Graphical output directory",
                      "./",
                      Patterns::DirectoryName());
  }
  prm.leave_subsection();
}



void OutputControlParameters::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Output control parameters");
  {
    graphical_output_frequency = prm.get_integer("Graphical output frequency");
    Assert(graphical_output_frequency > 0,
           ExcMessage("The graphical output frequency must larger than zero."));


    terminal_output_frequency = prm.get_integer("Terminal output frequency");
    Assert(terminal_output_frequency > 0,
           ExcMessage("The terminal output frequency must larger than zero."));

    output_directory = prm.get("Graphical output directory");
  }
  prm.leave_subsection();
}



template<typename Stream>
Stream& operator<<(Stream &stream, const OutputControlParameters &prm)
{
  internal::add_header(stream);
  internal::add_line(stream, "Output control parameters");
  internal::add_header(stream);

  internal::add_line(stream,
                     "Graphical output frequency",
                     prm.graphical_output_frequency);
  internal::add_line(stream,
                     "Terminal output frequency",
                     prm.terminal_output_frequency);
  internal::add_line(stream,
                     "Graphical output directory",
                     prm.output_directory);

  internal::add_header(stream);

  return (stream);
}



ProblemBaseParameters::ProblemBaseParameters()
:
OutputControlParameters(),
dim(2),
mapping_degree(1),
mapping_interior_cells(false),
verbose(false),
spatial_discretization_parameters(),
time_discretization_parameters()
{}



void ProblemBaseParameters::declare_parameters(ParameterHandler &prm)
{
  prm.declare_entry("Spatial dimension",
                    "2",
                    Patterns::Integer(1));

  prm.declare_entry("Mapping - Polynomial degree",
                    "1",
                    Patterns::Integer(1));

  prm.declare_entry("Mapping - Apply to interior cells",
                    "false",
                    Patterns::Bool());

  prm.declare_entry("Verbose",
                    "false",
                    Patterns::Bool());

  OutputControlParameters::declare_parameters(prm);

  SpatialDiscretizationParameters::declare_parameters(prm);

  TimeDiscretization::
  TimeDiscretizationParameters::declare_parameters(prm);

}



void ProblemBaseParameters::parse_parameters(ParameterHandler &prm)
{
  dim = prm.get_integer("Spatial dimension");
  AssertThrow(dim > 0, ExcLowerRange(dim, 0) );
  AssertThrow(dim <= 3, ExcMessage("The spatial dimension is larger than three.") );

  mapping_degree = prm.get_integer("Mapping - Polynomial degree");
  AssertThrow(mapping_degree > 0, ExcLowerRange(mapping_degree, 0) );

  mapping_interior_cells = prm.get_bool("Mapping - Apply to interior cells");

  verbose = prm.get_bool("Verbose");

  OutputControlParameters::parse_parameters(prm);

  spatial_discretization_parameters.parse_parameters(prm);

  time_discretization_parameters.parse_parameters(prm);
}



template<typename Stream>
Stream& operator<<(Stream &stream, const ProblemBaseParameters &prm)
{
  internal::add_header(stream);
  internal::add_line(stream, "Problem parameters");
  internal::add_header(stream);

  internal::add_line(stream, "Spatial dimension", prm.dim);

  {
    std::stringstream strstream;

    strstream << "MappingQ<" << std::to_string(prm.dim) << ">"
              << "(" << std::to_string(prm.mapping_degree) << ")";
    internal::add_line(stream, "Mapping", strstream.str().c_str());
  }

  internal::add_line(stream,
                     "Mapping - Apply to interior cells",
                     (prm.mapping_interior_cells ? "true" : "false"));


  internal::add_line(stream, "Verbose", (prm.verbose? "true": "false"));

  stream << static_cast<const OutputControlParameters &>(prm);

  stream << "\r";

  stream << prm.spatial_discretization_parameters;

  stream << "\r";

  stream << prm.time_discretization_parameters;

  stream << "\r";

  return (stream);
}



} // namespace RunTimeParameters

} // namespace RMHD

template void RMHD::RunTimeParameters::internal::add_header
(std::ostream &);
template void RMHD::RunTimeParameters::internal::add_header
(dealii::ConditionalOStream &);

template void RMHD::RunTimeParameters::internal::add_line
(std::ostream &, const char[]);
template void RMHD::RunTimeParameters::internal::add_line
(std::ostream &, std::string);

template void RMHD::RunTimeParameters::internal::add_line
(dealii::ConditionalOStream &, const char[]);
template void RMHD::RunTimeParameters::internal::add_line
(dealii::ConditionalOStream &, const std::string);

template void RMHD::RunTimeParameters::internal::add_line
(std::ostream &, const char[], const double);
template void RMHD::RunTimeParameters::internal::add_line
(std::ostream &, const char[], const unsigned int);
template void RMHD::RunTimeParameters::internal::add_line
(std::ostream &, const char[], const int);
template void RMHD::RunTimeParameters::internal::add_line
(std::ostream &, const char[], const std::string);
template void RMHD::RunTimeParameters::internal::add_line
(std::ostream &, const char[], const char[]);

template void RMHD::RunTimeParameters::internal::add_line
(std::ostream &, const std::string, const double);
template void RMHD::RunTimeParameters::internal::add_line
(std::ostream &, const std::string, const unsigned int);
template void RMHD::RunTimeParameters::internal::add_line
(std::ostream &, const std::string, const int);
template void RMHD::RunTimeParameters::internal::add_line
(std::ostream &, const std::string, const std::string);
template void RMHD::RunTimeParameters::internal::add_line
(std::ostream &, const std::string, const char[]);

template void RMHD::RunTimeParameters::internal::add_line
(dealii::ConditionalOStream &, const char[], const double);
template void RMHD::RunTimeParameters::internal::add_line
(dealii::ConditionalOStream &, const char[], const unsigned int);
template void RMHD::RunTimeParameters::internal::add_line
(dealii::ConditionalOStream &, const char[], const int);
template void RMHD::RunTimeParameters::internal::add_line
(dealii::ConditionalOStream &, const char[], const std::string);
template void RMHD::RunTimeParameters::internal::add_line
(dealii::ConditionalOStream &, const char[], const char[]);

template void RMHD::RunTimeParameters::internal::add_line
(dealii::ConditionalOStream &, const std::string, const double);
template void RMHD::RunTimeParameters::internal::add_line
(dealii::ConditionalOStream &, const std::string, const unsigned int);
template void RMHD::RunTimeParameters::internal::add_line
(dealii::ConditionalOStream &, const std::string, const int);
template void RMHD::RunTimeParameters::internal::add_line
(dealii::ConditionalOStream &, const std::string, const std::string);
template void RMHD::RunTimeParameters::internal::add_line
(dealii::ConditionalOStream &, const std::string, const char[]);

template std::ostream & RMHD::RunTimeParameters::operator<<
(std::ostream &, const RMHD::RunTimeParameters::SpatialDiscretizationParameters &);
template dealii::ConditionalOStream  & RMHD::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const RMHD::RunTimeParameters::SpatialDiscretizationParameters &);

template std::ostream & RMHD::RunTimeParameters::operator<<
(std::ostream &, const RMHD::RunTimeParameters::OutputControlParameters &);
template dealii::ConditionalOStream  & RMHD::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const RMHD::RunTimeParameters::OutputControlParameters &);

template std::ostream & RMHD::RunTimeParameters::operator<<
(std::ostream &, const RMHD::RunTimeParameters::ProblemBaseParameters &);
template dealii::ConditionalOStream & RMHD::RunTimeParameters::operator<<
(dealii::ConditionalOStream &, const RMHD::RunTimeParameters::ProblemBaseParameters &);

