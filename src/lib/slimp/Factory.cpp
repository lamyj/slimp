#include "Factory.h"

#include <functional>
#include <map>
#include <ostream>
#include <stdexcept>
#include <string>

#include <stan/model/model_header.hpp>

Factory * Factory::_instance = nullptr;

Factory &
Factory
::instance()
{
    static Factory instance;
    return instance;
}

void
Factory
::register_(std::string const & name, Creator creator)
{
    this->_creators.insert({name, creator});
}

stan::model::model_base &
Factory
::get(
    std::string const & name, stan::io::var_context & context,
    unsigned int seed, std::ostream * stream) const
{
    auto const iterator = this->_creators.find(name);
    if(iterator == this->_creators.end())
    {
        throw std::runtime_error("No such program: "+name);
    }
    
    auto && creator = iterator->second;
    return creator(context, seed, stream);
}
