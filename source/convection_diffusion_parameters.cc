/*
 * convection_diffusion_parameters.cc
 *
 *  Created on: Aug 27, 2021
 *      Author: sg
 */

#include <rotatingMHD/convection_diffusion_parameters.h>

namespace RMHD
{


ConvectionDiffusionSolverParameters::ConvectionDiffusionSolverParameters()
:
time_discretization(RunTimeParameters::ConvectiveTermTimeDiscretization::semi_implicit),
equation_coefficient(1.0),
linear_solver("Convection-diffusion equation"),
preconditioner_update_frequency(10),
verbose(false)
{}



void ConvectionDiffusionSolverParameters::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Convection-diffusion solver parameters");
  {
    prm.declare_entry("Convective term time discretization",
                      "semi-implicit",
                      Patterns::Selection("semi-implicit|explicit"));

    prm.declare_entry("Preconditioner update frequency",
                      "10",
                      Patterns::Integer(1));

    prm.declare_entry("Verbose",
                      "false",
                      Patterns::Bool());

    prm.enter_subsection("Linear solver parameters");
    {
      RunTimeParameters::LinearSolverParameters::declare_parameters(prm);
    }
    prm.leave_subsection();
  }
  prm.leave_subsection();
}



void ConvectionDiffusionSolverParameters::parse_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Convection-diffusion solver parameters");
  {
    const std::string str_convective_term_time_discretization(prm.get("Convective term time discretization"));

    if (str_convective_term_time_discretization == std::string("semi-implicit"))
      time_discretization
      = RunTimeParameters::ConvectiveTermTimeDiscretization::semi_implicit;
    else if (str_convective_term_time_discretization == std::string("explicit"))
      time_discretization
      = RunTimeParameters::ConvectiveTermTimeDiscretization::fully_explicit;
    else
      AssertThrow(false,
                  ExcMessage("Unexpected identifier for the time discretization "
                             "of the convective term."));

    preconditioner_update_frequency = prm.get_integer("Preconditioner update frequency");
    AssertThrow(preconditioner_update_frequency > 0,
                ExcLowerRange(preconditioner_update_frequency, 0));

    verbose = prm.get_bool("Verbose");

    prm.enter_subsection("Linear solver parameters");
    {
      linear_solver.parse_parameters(prm);
    }
    prm.leave_subsection();
  }
  prm.leave_subsection();
}



template<typename Stream>
Stream& operator<<(Stream &stream, const ConvectionDiffusionSolverParameters &prm)
{
  using namespace RunTimeParameters::internal;

  add_header(stream);
  add_line(stream, "Convection-diffusion solver parameters");
  add_header(stream);

  switch (prm.time_discretization)
  {
    case RunTimeParameters::ConvectiveTermTimeDiscretization::semi_implicit:
      add_line(stream, "Time discretization", "semi-implicit");
      break;
    case RunTimeParameters::ConvectiveTermTimeDiscretization::fully_explicit:
      add_line(stream, "Time discretization", "explicit");
      break;
    default:
      AssertThrow(false, ExcMessage("Unexpected type identifier for the "
                                    "time discretization of the convective term."));
      break;
  }

  add_line(stream, "Preconditioner update frequency", prm.preconditioner_update_frequency);

  stream << prm.linear_solver;

  return (stream);
}



ConvectionDiffusionParameters::ConvectionDiffusionParameters()
:
ProblemBaseParameters(),
fe_degree(1),
peclet_number(1.0),
solver_parameters()
{}



ConvectionDiffusionParameters::ConvectionDiffusionParameters
(const std::string &parameter_filename)
:
ConvectionDiffusionParameters()
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



void ConvectionDiffusionParameters::declare_parameters(ParameterHandler &prm)
{
  prm.declare_entry("FE's polynomial degree",
                    "1",
                    Patterns::Integer(1));

  prm.declare_entry("Peclet number",
                    "1.0",
                    Patterns::Double());

  ProblemBaseParameters::declare_parameters(prm);

  ConvectionDiffusionParameters::declare_parameters(prm);
}



void ConvectionDiffusionParameters::parse_parameters(ParameterHandler &prm)
{
  ProblemBaseParameters::parse_parameters(prm);


  fe_degree = prm.get_integer("FE's polynomial degree");
  AssertThrow(fe_degree > 0, ExcLowerRange(fe_degree, 0));

  peclet_number = prm.get_double("Peclet number");
  Assert(peclet_number > 0.0, ExcLowerRangeType<double>(peclet_number, 0.0));
  AssertIsFinite(peclet_number)
  AssertIsFinite(1.0 / peclet_number);

  solver_parameters.equation_coefficient = 1.0 / peclet_number;

  solver_parameters.parse_parameters(prm);
}



template<typename Stream>
Stream& operator<<(Stream &stream, const ConvectionDiffusionParameters &prm)
{
  using namespace RunTimeParameters::internal;

  add_header(stream);
  add_line(stream, "Convection-diffusion problem parameters");
  add_header(stream);

  {
    std::string fe = "FE_Q<" + std::to_string(prm.dim) + ">(" + std::to_string(prm.fe_degree) + ")";
    add_line(stream, "Finite Element", fe);
  }

  add_line(stream, "Peclet number", prm.peclet_number);

  stream << static_cast<const RunTimeParameters::ProblemBaseParameters &>(prm);

  stream << prm.solver_parameters;

  add_header(stream);

  return (stream);
}

// explicit instantiations
template std::ostream & operator<<
(std::ostream &, const ConvectionDiffusionParameters &);
template dealii::ConditionalOStream & operator<<
(ConditionalOStream &, const ConvectionDiffusionParameters &);


}  // namespace RMHD



