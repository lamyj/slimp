#include "Factory.h"

#include <functional>
#include <map>
#include <ostream>
#include <string>

#include <stan/model/model_header.hpp>

Factory * Factory::_instance = nullptr;

Factory &
Factory
::instance()
{
    if(Factory::_instance == nullptr)
    {
        Factory::_instance = new Factory();
    }
    
    return *Factory::_instance;
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
    auto && creator = this->_creators.at(name);
    return creator(context, seed, stream);
}
