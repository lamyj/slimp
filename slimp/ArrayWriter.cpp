#include "ArrayWriter.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

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
            + " got " + std::to_string(state.size()));
    }
    
    std::copy(
        state.begin()+this->_skip, state.end(),
        this->_array.mutable_unchecked().mutable_data(
            this->_chain, this->_draw, this->_offset));
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
    
    for(size_t j=0; j!=values.cols(); ++j)
    {
        for(size_t i=this->_skip; i!=values.rows(); ++i)
        {
            *this->_array.mutable_unchecked().mutable_data(
                    this->_chain, this->_draw, i+this->_offset-this->_skip
                ) = values(i, j);
        }
        ++this->_draw;
    }
}

std::vector<std::string> const &
ArrayWriter
::names() const
{
    return this->_names;
}
