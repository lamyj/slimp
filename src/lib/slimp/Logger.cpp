#include "Logger.h"

#include <map>
#include <mutex>
#include <string>
#include <sstream>
#include <pybind11/pybind11.h>
#include <stan/callbacks/logger.hpp>

#include <iostream>

Logger
::Logger()
{
    auto logging = pybind11::module::import("logging");
    for(auto && level: {"debug", "info", "warning", "error", "critical"})
    {
        this->_loggers[level] = pybind11::getattr(logging, level);
    }
}

void
Logger
::debug(std::string const & message)
{
    this->_log("debug", message);
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
    this->_log("info", message);
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
    this->_log("warning", message);
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
    this->_log("error", message);
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
    this->_log("critical", message);
}

void
Logger
::fatal(std::stringstream const & message)
{
    this->fatal(message.str());
}

void
Logger
::_log(std::string const & level, std::string const & message) const
{
    if(message.empty())
    {
        return;
    }
    
    std::lock_guard<std::mutex> guard(this->_mutex);
    
    // auto logger = pybind11::getattr(this->_logging, level.c_str());
    // logger(message);
    this->_loggers.at(level)(message);
}
