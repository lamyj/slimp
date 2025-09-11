#include "VarContext.h"

#include <stdint.h>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <vector>

#include <stan/io/validate_dims.hpp>
#include <stan/io/var_context.hpp>

namespace slimp
{

void
VarContext
::set(std::string const & key, int x)
{
    this->_vals_i[key] = {x};
    this->_dims_i[key] = {};
}

void
VarContext
::set(std::string const & key, double x)
{
    this->_vals_r[key] = {x};
    this->_dims_r[key] = {};
}

bool
VarContext
::contains_r(std::string const & name) const
{
    return
        this->_vals_r.find(name) != this->_vals_r.end()
        || this->contains_i(name);
}

std::vector<double>
VarContext
::vals_r(std::string const & name) const
{
    auto const iterator = this->_vals_r.find(name);
    if(iterator != this->_vals_r.end())
    {
        return iterator->second;
    }
    else
    {
        auto const vals_i = this->vals_i(name);
        return {vals_i.begin(), vals_i.end()};
    }
}

std::vector<std::complex<double>>
VarContext
::vals_c(std::string const & name) const
{
    throw std::runtime_error("Not implemented");
}

std::vector<size_t>
VarContext
::dims_r(std::string const & name) const
{
    auto const iterator = this->_dims_r.find(name);
    if(iterator != this->_dims_r.end())
    {
        return iterator->second;
    }
    else
    {
        return this->dims_i(name);
    }
}

bool
VarContext
::contains_i(std::string const & name) const
{
    return this->_vals_i.find(name) != this->_vals_i.end();
}

std::vector<int>
VarContext
::vals_i(std::string const & name) const
{
    auto const iterator = this->_vals_i.find(name);
    if(iterator != this->_vals_i.end())
    {
        return iterator->second;
    }
    else
    {
        return {};
    }
}

std::vector<size_t>
VarContext
::dims_i(std::string const & name) const
{
    auto const iterator = this->_dims_i.find(name);
    if(iterator != this->_dims_i.end())
    {
        return iterator->second;
    }
    else
    {
        return {};
    }
}

void
VarContext
::names_r(std::vector<std::string> & names) const
{
    names.clear();
    names.reserve(this->_vals_r.size());
    for(auto && item: this->_vals_r)
    {
        names.push_back(item.first);
    }
}

void
VarContext
::names_i(std::vector<std::string> & names) const
{
    names.clear();
    names.reserve(this->_vals_i.size());
    for(auto && item: this->_vals_i)
    {
        names.push_back(item.first);
    }
}

void
VarContext
::validate_dims(
    std::string const& stage, std::string const & name,
    std::string const& base_type,
    std::vector<size_t> const & dims_declared) const
{
    size_t num_elts = 1;
    for(auto && d: dims_declared)
    {
        num_elts *= d;
    }
    if(num_elts == 0)
    {
        return;
    }
    stan::io::validate_dims(*this, stage, name, base_type, dims_declared);
}

}
