#include "ArrayWriter.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// WARNING: Stan must be included before Eigen so that the plugin system is
// active. https://discourse.mc-stan.org/t/includes-in-user-header/26093
#include <stan/math.hpp>

#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <stan/callbacks/writer.hpp>

ArrayWriter
::ArrayWriter(Array & array, size_t chain, size_t offset, size_t skip)
: _array(array), _chain(chain), _offset(offset), _skip(skip), _draw(0), _names()
{
    // Nothing else
}

void
ArrayWriter
::operator()(std::vector<std::string> const & names)
{
    // NOTE: names are informative, don't check their size    
    this->_names = names;
}

void
ArrayWriter
::operator()(std::vector<double> const & state)
{
    if(state.size()-this->_skip != this->_array.shape(2)-this->_offset)
    {
        throw std::runtime_error(
            "Shape mismatch (state): expected "
            + std::to_string(this->_array.shape(2)-this->_offset)
            + " got " + std::to_string(state.size()-this->_skip));
    }
    
    auto source = state.begin()+this->_skip;
    auto destination = this->_array.mutable_unchecked().mutable_data(
        this->_chain, this->_draw, this->_offset);
    auto const stride = this->_array.strides()[2]/this->_array.itemsize();
    while(source != state.end())
    {
        *destination = *source;
        ++source;
        destination += stride;
    }
    ++this->_draw;
}

void
ArrayWriter
::operator()(std::string const & message)
{
    this->_messages[this->_draw].push_back(message);
}

void
ArrayWriter
::operator()(Eigen::Ref<Eigen::Matrix<double, -1, -1>> const & values)
{
    if(values.rows()-this->_skip != this->_array.shape(2)-this->_offset)
    {
        throw std::runtime_error(
            "Shape mismatch (values): expected "
            + std::to_string(this->_array.shape(2)-this->_offset)
            + " got " + std::to_string(values.rows()-this->_skip));
    }
    
    auto source = values(this->_skip, 0);
    auto destination = this->_array.mutable_unchecked().mutable_data(
        this->_chain, this->_draw, this->_offset);
    auto const stride = this->_array.strides()[2]/this->_array.itemsize();
    for(size_t j=0; j!=values.cols(); ++j)
    {
        for(size_t i=this->_skip; i!=values.rows(); ++i)
        {
            *destination = values(i,j);
            destination += stride;
        }
        ++this->_draw;
        destination = this->_array.mutable_unchecked().mutable_data(
            this->_chain, this->_draw, this->_offset);
    }
}

std::vector<std::string> const &
ArrayWriter
::names() const
{
    return this->_names;
}
