#ifndef _817f0186_0979_4426_a1a6_6c6131e66ef6
#define _817f0186_0979_4426_a1a6_6c6131e66ef6

#include <cstdint>

#if __has_include(<xtensor/xtensor.hpp>)
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#else
#include <xtensor/containers/xarray.hpp>
#include <xtensor/containers/xtensor.hpp>
#endif

namespace slimp
{

using Arrayi8 = xt::xarray<int8_t>;
using Arrayi16 = xt::xarray<int16_t>;
using Arrayi32 = xt::xarray<int32_t>;
using Arrayi64 = xt::xarray<int64_t>;

using Arrayui8 = xt::xarray<uint8_t>;
using Arrayui16 = xt::xarray<uint16_t>;
using Arrayui32 = xt::xarray<uint32_t>;
using Arrayui64 = xt::xarray<uint64_t>;

using Arrayf = xt::xarray<float>;
using Arrayd = xt::xarray<double>;

template<typename T>
using Array = xt::xarray<T>;

using Tensor1d = xt::xtensor<double, 1>;
using Tensor2d = xt::xtensor<double, 2>;
using Tensor3d = xt::xtensor<double, 3>;
using Tensor4d = xt::xtensor<double, 4>;

}

#endif // _817f0186_0979_4426_a1a6_6c6131e66ef6
