#ifndef _b1fa63f0_8b1e_42c6_bc2d_f511b5f400a4
#define _b1fa63f0_8b1e_42c6_bc2d_f511b5f400a4

#include "VarContext.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <stan/io/var_context.hpp>
#include <xtensor/xarray.hpp>

#include "slimp/api.h"

namespace slimp
{

template<typename T, std::enable_if_t<std::is_integral<T>::value, bool>>
void
VarContext
::set(std::string const & key, xt::xarray<T> const & array)
{
    this->_vals_i[key] = {
        array.template begin<xt::layout_type::column_major>(),
        array.template end<xt::layout_type::column_major>()};
    
    auto const shape = array.shape();
    this->_dims_i[key] = {shape.begin(), shape.end()};
}

template<typename T, std::enable_if_t<std::is_floating_point<T>::value, bool>>
void
VarContext
::set(std::string const & key, xt::xarray<T> const & array)
{
    this->_vals_r[key] = {
        array.template begin<xt::layout_type::column_major>(),
        array.template end<xt::layout_type::column_major>()};
    
    auto const shape = array.shape();
    this->_dims_r[key] = {shape.begin(), shape.end()};
}

}

#endif // _b1fa63f0_8b1e_42c6_bc2d_f511b5f400a4
