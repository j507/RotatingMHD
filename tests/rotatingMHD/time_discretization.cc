#include <rotatingMHD/time_discretization.h>

#include <iostream>
#include <sstream>

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

void test(RMHD::TimeDiscretization::VSIMEXScheme scheme)
{
  std::stringstream ss;

  using namespace RMHD::TimeDiscretization;

  TimeDiscretizationParameters  parameters;
  parameters.vsimex_scheme = scheme;
  parameters.minimum_time_step = 0.01;
  parameters.maximum_time_step = 1.0;
  parameters.initial_time_step = 0.1;

  std::cout << parameters << std::endl;

  {
    VSIMEXMethod  timestepping(parameters);

    while (!timestepping.is_at_end())
    {
      timestepping.set_desired_next_step_size(0.1);

      std::cout << timestepping << std::endl;

      timestepping.update_coefficients();
      if (timestepping.coefficients_changed())
        timestepping.print_coefficients(std::cout, "    ");

      timestepping.advance_time();
    }
    std::cout << timestepping << std::endl;

    boost::archive::text_oarchive oa{ss};
    oa << timestepping;
  }
  {
    // deserialize
    boost::archive::text_iarchive ia{ss};
    VSIMEXMethod  timestepping;
    ia >> timestepping;

    timestepping.set_end_time(2.0);

    while (!timestepping.is_at_end())
    {
      timestepping.set_desired_next_step_size(0.2);

      std::cout << timestepping << std::endl;

      timestepping.update_coefficients();
      if (timestepping.coefficients_changed())
        timestepping.print_coefficients(std::cout, "    ");

      timestepping.advance_time();
    }
    std::cout << timestepping << std::endl;
  }
  return;
}



int main(void)
{
  try
  {
    test(RMHD::TimeDiscretization::VSIMEXScheme::BDF2);
    test(RMHD::TimeDiscretization::VSIMEXScheme::CNAB);
    test(RMHD::TimeDiscretization::VSIMEXScheme::mCNAB);
    test(RMHD::TimeDiscretization::VSIMEXScheme::CNLF);
  }
  catch(std::exception & exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }
  catch(...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}
