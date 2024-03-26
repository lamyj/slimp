#include "ArrayWriter.h"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <stan/callbacks/writer.hpp>

ArrayWriter
::ArrayWriter(Array & array, size_t chain, size_t skip)
: _array(array), _chain(chain), _skip(skip), _draw(0), _names()
{
    // Nothing else
}

void
ArrayWriter
::operator()(std::vector<std::string> const & names)
{
    if(names.size() != this->_array.shape(2))
    {
        throw std::runtime_error(
            "Shape mismatch (names): expected "
            + std::to_string(this->_array.shape(2))
            + " got " + std::to_string(names.size()));
    }
    
    this->_names = names;
}

void
ArrayWriter
::operator()(std::vector<double> const & state)
{
    if(state.size() != this->_array.shape(2))
    {
        throw std::runtime_error(
            "Shape mismatch (state): expected "
            + std::to_string(this->_array.shape(2))
            + " got " + std::to_string(state.size()));
    }
    std::copy(
        state.begin(), state.end(),
        this->_array.mutable_unchecked().mutable_data(
            this->_chain, this->_draw));
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
    for(size_t j=0; j!=values.cols(); ++j)
    {
        for(size_t i=this->_skip; i!=values.rows(); ++i)
        {
            *this->_array.mutable_unchecked().mutable_data(
                    this->_chain, this->_draw, i-this->_skip
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
