#include "Logger.h"

#include <string>
#include <sstream>

#include <pybind11/pybind11.h>
#include <stan/callbacks/logger.hpp>

namespace slimp
{

Logger
::Logger()
{
    pybind11::gil_scoped_acquire acquire_gil;
    
    auto logging = pybind11::module::import("logging");
    
    this->_loggers.resize(1+int(Level::Max));
    this->_loggers[int(Level::Debug)] = logging.attr("debug");
    this->_loggers[int(Level::Info)] = logging.attr("info");
    this->_loggers[int(Level::Warn)] = logging.attr("warning");
    this->_loggers[int(Level::Error)] = logging.attr("error");
    this->_loggers[int(Level::Fatal)] = logging.attr("critical");
}

void
Logger
::debug(std::string const & message)
{
    this->_log(Level::Debug, message);
}

void
Logger
::debug(std::stringstream const & message)
{
    this->debug(message.str());
}

void
Logger
::info(std::string const & message)
{
    this->_log(Level::Info, message);
}

void
Logger
::info(std::stringstream const & message)
{
    this->info(message.str());
}

void
Logger
::warn(std::string const & message)
{
    this->_log(Level::Warn, message);
}

void
Logger
::warn(std::stringstream const & message)
{
    this->warn(message.str());
}

void
Logger
::error(std::string const & message)
{
    this->_log(Level::Error, message);
}

void
Logger
::error(std::stringstream const & message)
{
    this->error(message.str());
}

void
Logger
::fatal(std::string const & message)
{
    this->_log(Level::Fatal, message);
}

void
Logger
::fatal(std::stringstream const & message)
{
    this->fatal(message.str());
}

void
Logger
::_log(Level level, std::string const & message) const
{
    if(message.empty())
    {
        return;
    }
    
    pybind11::gil_scoped_acquire acquire_gil;
    this->_loggers[int(level)](message);
}

}
