#include "VarContext.h"

#include <stdint.h>
#include <unordered_map>
#include <stdexcept>
#include <string>
#include <vector>

#include <numpy/ndarrayobject.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stan/io/validate_dims.hpp>
#include <stan/io/var_context.hpp>

VarContext
::VarContext(pybind11::dict dictionary)
{
    for(auto && item: dictionary)
    {
        auto const & key = item.first.cast<std::string>();
        auto const & value = item.second;
        if(pybind11::isinstance<pybind11::int_>(value))
        {
            this->_vals_i.insert({key, {value.cast<int>()}});
            this->_dims_i.insert({key, {}});
        }
        else if(pybind11::isinstance<pybind11::float_>(value))
        {
            this->_vals_r.insert({key, {value.cast<double>()}});
            this->_dims_r.insert({key, {}});
        }
        else
        {
            auto array = value.cast<pybind11::array>();
            if(!(array.flags() & NPY_ARRAY_F_CONTIGUOUS))
            {
                throw std::runtime_error("Arrays must be F-contiguous");
            }
            
            auto dtype = array.dtype();
            std::vector<size_t> const shape(
                array.shape(), array.shape()+array.ndim());
            
            // https://numpy.org/doc/stable/reference/arrays.scalars.html#arrays-scalars-built-in
            
            // Signed integer type
            if(dtype.char_() == 'b')
            {
                auto & v = this->_vals_i.insert({key, {}}).first->second;
                v.resize(array.size());
                auto data = reinterpret_cast<int8_t const *>(array.data());
                std::copy(data, data+array.size(), v.begin());
                this->_dims_i.insert({key, shape});
            }
            else if(dtype.char_() == 'h')
            {
                auto & v = this->_vals_i.insert({key, {}}).first->second;
                v.resize(array.size());
                auto data = reinterpret_cast<int16_t const *>(array.data());
                std::copy(data, data+array.size(), v.begin());
                this->_dims_i.insert({key, shape});
            }
            else if(dtype.char_() == 'i')
            {
                auto & v = this->_vals_i.insert({key, {}}).first->second;
                v.resize(array.size());
                auto data = reinterpret_cast<int32_t const *>(array.data());
                std::copy(data, data+array.size(), v.begin());
                this->_dims_i.insert({key, shape});
            }
            else if(dtype.char_() == 'l')
            {
                auto & v = this->_vals_i.insert({key, {}}).first->second;
                v.resize(array.size());
                auto data = reinterpret_cast<int64_t const *>(array.data());
                std::copy(data, data+array.size(), v.begin());
                this->_dims_i.insert({key, shape});
            }
            // Unsigned integer types
            else if(dtype.char_() == 'B')
            {
                auto & v = this->_vals_i.insert({key, {}}).first->second;
                v.resize(array.size());
                auto data = reinterpret_cast<uint8_t const *>(array.data());
                std::copy(data, data+array.size(), v.begin());
                this->_dims_i.insert({key, shape});
            }
            else if(dtype.char_() == 'H')
            {
                auto & v = this->_vals_i.insert({key, {}}).first->second;
                v.resize(array.size());
                auto data = reinterpret_cast<uint16_t const *>(array.data());
                std::copy(data, data+array.size(), v.begin());
                this->_dims_i.insert({key, shape});
            }
            else if(dtype.char_() == 'I')
            {
                auto & v = this->_vals_i.insert({key, {}}).first->second;
                v.resize(array.size());
                auto data = reinterpret_cast<uint32_t const *>(array.data());
                std::copy(data, data+array.size(), v.begin());
                this->_dims_i.insert({key, shape});
            }
            else if(dtype.char_() == 'L')
            {
                auto & v = this->_vals_i.insert({key, {}}).first->second;
                v.resize(array.size());
                auto data = reinterpret_cast<uint64_t const *>(array.data());
                std::copy(data, data+array.size(), v.begin());
                this->_dims_i.insert({key, shape});
            }
            // Floating-point types
            else if(dtype.char_() == 'f')
            {
                auto & v = this->_vals_r.insert({key, {}}).first->second;
                v.resize(array.size());
                auto data = reinterpret_cast<float const *>(array.data());
                std::copy(data, data+array.size(), v.begin());
                this->_dims_r.insert({key, shape});
            }
            else if(dtype.char_() == 'd')
            {
                auto data = reinterpret_cast<double const *>(array.data());
                this->_vals_r.insert({key, {data, data+array.size()}});
                this->_dims_r.insert({key, shape});
            }
            // Unsupported type
            else
            {
                throw std::runtime_error(
                    std::string("Array type not handled: ")+ dtype.char_());
            }
        }
    }
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
