#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/vector.h>

#include <rotatingMHD/convergence_test.h>

#include <string.h>

namespace RMHD
{

using namespace dealii;

namespace ConvergenceTest
{

namespace internal
{
  constexpr char header[] = "+------------------------------------------+"
                        "----------------------+";

  constexpr size_t column_width[2] = {40, 20};

  constexpr size_t line_width = 63;

  template<typename Stream, typename A>
  void add_line(Stream  &stream,
                const A &line)
  {
    stream << "| "
           << std::setw(line_width)
           << line
           << " |"
           << std::endl;
  }

  template<typename Stream, typename A, typename B>
  void add_line(Stream  &stream,
                const A &first_column,
                const B &second_column)
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


Parameters::Parameters()
:
type(Type::temporal),
n_spatial_cycles(2),
step_size_reduction_factor(0.5),
n_temporal_cycles(2)
{}


Parameters::Parameters
(const std::string &parameter_filename)
:
Parameters()
{
  ParameterHandler prm;

  declare_parameters(prm);

  std::ifstream parameter_file(parameter_filename.c_str());

  if (!parameter_file)
  {
    parameter_file.close();

    std::ostringstream message;
    message << "Input parameter file <"
            << parameter_filename << "> not found. Creating a"
            << std::endl
            << "template file of the same name."
            << std::endl;

    std::ofstream parameter_out(parameter_filename.c_str());
    prm.print_parameters(parameter_out,
                         ParameterHandler::OutputStyle::Text);

    AssertThrow(false, ExcMessage(message.str().c_str()));
  }

  prm.parse_input(parameter_file);

  parse_parameters(prm);
}


void Parameters::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Convergence test parameters");
  {
    prm.declare_entry("Convergence test type",
                      "temporal",
                      Patterns::Selection("spatial|temporal|both"));

    prm.declare_entry("Number of spatial convergence cycles",
                      "2",
                      Patterns::Integer(1));

    prm.declare_entry("Time-step reduction factor",
                      "0.5",
                      Patterns::Double());

    prm.declare_entry("Number of temporal convergence cycles",
                      "2",
                      Patterns::Integer(1));
  }
  prm.leave_subsection();
}



void Parameters::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Convergence test parameters");
  {
    if (prm.get("Convergence test type") == std::string("spatial"))
    {
      type = Type::spatial;

      n_spatial_cycles = prm.get_integer("Number of spatial convergence cycles");
      AssertThrow(n_spatial_cycles > 0,
                  ExcLowerRange(n_spatial_cycles, 0));
    }
    else if (prm.get("Convergence test type") == std::string("temporal"))
    {
      type = Type::temporal;

      step_size_reduction_factor = prm.get_double("Time-step reduction factor");
      AssertThrow(step_size_reduction_factor > 0.0,
                  ExcLowerRangeType<double>(step_size_reduction_factor, 0.0));
      AssertThrow(step_size_reduction_factor < 1.0,
                  ExcLowerRangeType<double>(1.0, step_size_reduction_factor));

      n_temporal_cycles = prm.get_integer("Number of temporal convergence cycles");
      AssertThrow(n_temporal_cycles > 0,
                  ExcLowerRange(n_temporal_cycles, 0));
    }
    else if (prm.get("Convergence test type") == std::string("both"))
    {
        type = Type::spatio_temporal;

        step_size_reduction_factor = prm.get_double("Time-step reduction factor");
        AssertThrow(step_size_reduction_factor > 0.0,
                    ExcLowerRangeType<double>(step_size_reduction_factor, 0.0));
        AssertThrow(step_size_reduction_factor < 1.0,
                    ExcLowerRangeType<double>(1.0, step_size_reduction_factor));

        n_temporal_cycles = prm.get_integer("Number of temporal convergence cycles");
        AssertThrow(n_temporal_cycles > 0,
                    ExcLowerRange(n_temporal_cycles, 0));

        n_spatial_cycles = prm.get_integer("Number of spatial convergence cycles");
        AssertThrow(n_spatial_cycles > 0,
                    ExcLowerRange(n_spatial_cycles, 0));

        AssertThrow(false, ExcNotImplemented());
    }
    else
      AssertThrow(false,
                  ExcMessage("Unexpected identifier for the type of"
                             " of convergence test."));
  }
  prm.leave_subsection();
}



template<typename Stream>
Stream& operator<<(Stream &stream, const Parameters &prm)
{
  internal::add_header(stream);
  internal::add_line(stream, "Convergence test parameters");
  internal::add_header(stream);

  switch (prm.type)
  {
    case Type::spatial:
      internal::add_line(stream, "Convergence test type", "spatial");
      break;
    case Type::temporal:
      internal::add_line(stream, "Convergence test type", "temporal");
      break;
    default:
      AssertThrow(false, ExcMessage("Unexpected identifier for the type of"
                               " of convergence test."));
      break;
  }

  internal::add_line(stream,
                     "Number of spatial convergence cycles",
                     prm.n_spatial_cycles);

  internal::add_line(stream,
                     "Number of temporal convergence cycles",
                     prm.n_temporal_cycles);

  internal::add_line(stream,
                     "Time-step reduction factor",
                     prm.step_size_reduction_factor);

  internal::add_header(stream);

  return (stream);
}


ConvergenceResults::ConvergenceResults(const Type type)
:
type(type),
n_rows(0)
{}

template<int dim, int spacedim>
void ConvergenceResults::update
(const std::map<typename ConvergenceResults::NormType, double> &error_map,
 const DoFHandler<dim, spacedim> &dof_handler,
 const double step_size)
{
  const types::global_dof_index n_dofs{dof_handler.n_dofs()};

  const Triangulation<dim, spacedim> &tria{dof_handler.get_triangulation()};
  const types::global_cell_index  n_cells{tria.n_global_active_cells()};
  const unsigned int n_levels{tria.n_global_levels()};

  const double h_max{GridTools::maximal_cell_diameter(tria)};

  update(error_map, n_dofs, n_cells, n_levels, h_max, step_size);
}

void ConvergenceResults::update
(const std::map<typename ConvergenceResults::NormType, double>&error_map,
 const types::global_dof_index    n_dofs,
 const types::global_cell_index   n_cells,
 unsigned int                     n_levels,
 const double                     h_max,
 const double                     time_step)
{

  n_rows += 1;
  table.add_value("level", n_rows);

  if (n_dofs < numbers::invalid_dof_index)
  {
    table.add_value("dofs", n_dofs);
    n_dofs_specified = true;
  }
  if (n_cells < numbers::invalid_coarse_cell_id)
  {
    table.add_value("cells", n_cells);
    n_cells_specified = true;
  }
  if (n_levels < static_cast<unsigned int>(-1))
  {
    table.add_value("refinements", n_levels);
    n_levels_specified = true;
  }
  if (h_max < std::numeric_limits<double>::max())
  {
    table.add_value("h", h_max);
    h_max_specified = true;
  }
  if (time_step > std::numeric_limits<double>::lowest())
  {
    table.add_value("time step", time_step);
    time_step_specified = true;
  }

  std::map<typename VectorTools::NormType, double>::const_iterator it;

  it = error_map.find(VectorTools::NormType::L2_norm);
  if (it != error_map.end())
  {
    table.add_value("L2", it->second);
    L2_error_specified = true;
  }

  it = error_map.find(VectorTools::NormType::Linfty_norm);
  if (it != error_map.end())
  {
    table.add_value("Linfty", it->second);
    Linfty_error_specified = true;
  }

  it = error_map.find(VectorTools::NormType::H1_norm);
  if (it != error_map.end())
  {
    table.add_value("H1", it->second);
    H1_error_specified = true;
  }
}



void ConvergenceResults::update
(const std::map<typename VectorTools::NormType, double> &error_map,
 const double                     time_step,
 const types::global_dof_index    n_dofs,
 const types::global_cell_index   n_cells,
 unsigned int                     n_levels,
 const double                     h_max)
{
  update(error_map, n_dofs, n_cells, n_levels, h_max, time_step);
}

void ConvergenceResults::format_columns()
{
  constexpr unsigned int precision{6};

  // set column formatting
  if (time_step_specified)
  {
      table.set_scientific("time step", true);
      table.set_precision("time step", 2);
  }
  if (h_max_specified)
  {
    table.set_scientific("h", true);
    table.set_precision("h", 2);
  }
  if (L2_error_specified)
  {
    table.set_scientific("L2", true);
    table.set_precision("L2", precision);
  }
  if (H1_error_specified)
  {
    table.set_scientific("H1", true);
    table.set_precision("H1", precision);
  }
  if (Linfty_error_specified)
  {
    table.set_scientific("Linfty", true);
    table.set_precision("Linfty", precision);
  }

  // evaluate convergence rates
  std::string column;
  bool evaluate_convergence{false};

  switch (type)
  {
    case Type::spatial:
      column.assign("h");
      evaluate_convergence = true;
      break;
    case Type::temporal:
      column.assign("time step");
      evaluate_convergence = true;
      break;
    case Type::spatio_temporal:
      // Do not evaluate convergence rates for a spatio-temporal test,
      // leaving switch-case to avoid throwing an exception.
      break;
    default:
      Assert(false,
             ExcMessage("Unexpected value for ConvergenceTest::Type."));
      break;
  }

  if (evaluate_convergence)
	{
    if (L2_error_specified)
      table.evaluate_convergence_rates("L2",
                                       column,
                                       ConvergenceTable::reduction_rate_log2,
                                       1);
    if (Linfty_error_specified)
      table.evaluate_convergence_rates("Linfty",
                                       column,
                                       ConvergenceTable::reduction_rate_log2,
                                       1);
    if (H1_error_specified)
      table.evaluate_convergence_rates("H1",
                                       column,
                                       ConvergenceTable::reduction_rate_log2,
                                       1);
	}
}

template<typename Stream>
Stream& operator<<(Stream &stream, ConvergenceResults &data)
{
  if (data.n_rows == 0)
    return (stream);

  data.format_columns();

  data.table.write_text(stream);

  return (stream);
}

template <>
ConditionalOStream& operator<<(ConditionalOStream &stream, ConvergenceResults &data)
{
  if (data.n_rows == 0)
    return (stream);

  data.format_columns();

  if (stream.is_active())
  	data.table.write_text(stream.get_stream());

  return (stream);
}


void ConvergenceResults::write_text(std::ostream  &file)
{
  if (n_rows == 0)
    return;

  format_columns();

  table.write_text(file, TableHandler::TextOutputFormat::org_mode_table);
}

} // namespace ConvergenceTest

} // namespace RMHD


// explicit instantiations
template std::ostream& RMHD::ConvergenceTest::operator<<
(std::ostream &, const RMHD::ConvergenceTest::Parameters &);
template dealii::ConditionalOStream& RMHD::ConvergenceTest::operator<<
(dealii::ConditionalOStream &, const RMHD::ConvergenceTest::Parameters &);

template
void RMHD::ConvergenceTest::ConvergenceResults::update
(const std::map<RMHD::ConvergenceTest::ConvergenceResults::NormType, double> &,
 const dealii::DoFHandler<1, 1> &,
 const double);

template
void RMHD::ConvergenceTest::ConvergenceResults::update
(const std::map<RMHD::ConvergenceTest::ConvergenceResults::NormType, double> &,
 const dealii::DoFHandler<1, 2> &,
 const double);

template
void RMHD::ConvergenceTest::ConvergenceResults::update
(const std::map<RMHD::ConvergenceTest::ConvergenceResults::NormType, double> &,
 const dealii::DoFHandler<2, 2> &,
 const double);

template
void RMHD::ConvergenceTest::ConvergenceResults::update
(const std::map<RMHD::ConvergenceTest::ConvergenceResults::NormType, double> &,
 const dealii::DoFHandler<1, 3> &,
 const double);

template
void RMHD::ConvergenceTest::ConvergenceResults::update
(const std::map<RMHD::ConvergenceTest::ConvergenceResults::NormType, double> &,
 const dealii::DoFHandler<2, 3> &,
 const double);

template
void RMHD::ConvergenceTest::ConvergenceResults::update
(const std::map<RMHD::ConvergenceTest::ConvergenceResults::NormType, double> &,
 const dealii::DoFHandler<3, 3> &,
 const double);

template std::ostream& RMHD::ConvergenceTest::operator<<
(std::ostream &, RMHD::ConvergenceTest::ConvergenceResults &);
