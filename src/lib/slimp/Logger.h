#ifndef _4ba5e886_d129_44ef_a3fc_e2a79869388e
#define _4ba5e886_d129_44ef_a3fc_e2a79869388e

#include <map>
#include <mutex>
#include <string>
#include <sstream>
#include <pybind11/pybind11.h>
#include <stan/callbacks/logger.hpp>

#include "slimp/api.h"

class SLIMP_API Logger: public stan::callbacks::logger
{
public:
    Logger();
    ~Logger() = default;
    
    void debug(std::string const & message) override;
    void debug(std::stringstream const & message) override;
    
    void info(std::string const & message) override;
    void info(std::stringstream const & message) override;
    
    void warn(std::string const & message) override;
    void warn(std::stringstream const & message) override;
    
    void error(std::string const & message) override;
    void error(std::stringstream const & message) override;
    
    void fatal(std::string const & message) override;
    void fatal(std::stringstream const & message) override;

private:
    mutable std::mutex _mutex;
    std::map<std::string, pybind11::object> _loggers;
    void _log(std::string const & level, std::string const & message) const;
};

#endif // _4ba5e886_d129_44ef_a3fc_e2a79869388e
