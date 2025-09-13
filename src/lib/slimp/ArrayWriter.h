#ifndef _f5319195_814d_49c2_8186_b46578694468
#define _f5319195_814d_49c2_8186_b46578694468

#include <cstdint>
#include <string>
#include <vector>

// WARNING: Stan must be included before Eigen so that the plugin system is
// active. https://discourse.mc-stan.org/t/includes-in-user-header/26093
#include <stan/math.hpp>

#include <Eigen/Dense>
#include <stan/callbacks/writer.hpp>
#include <stan/version.hpp>

#if __has_include(<xtensor/xtensor.hpp>)
#include <xtensor/xadapt.hpp>
#include <xtensor/xslice.hpp>
#include <xtensor/xview.hpp>
#else
#include <xtensor/containers/xadapt.hpp>
#include <xtensor/views/xslice.hpp>
#include <xtensor/views/xview.hpp>
#endif


#include "slimp/api.h"
#include "slimp/misc.h"

namespace slimp
{

/// @brief Stan writer to an array of shape parameters x chains x draws
class SLIMP_API ArrayWriter: public stan::callbacks::writer
{
public:
    using Array = Tensor3d;
    
    ArrayWriter() = delete;
    ArrayWriter(ArrayWriter const &) = delete;
    ArrayWriter(ArrayWriter &&) = default;
    ~ArrayWriter() = default;
    ArrayWriter & operator=(ArrayWriter const &) = delete;
    
    /**
     * @brief Create a writer to given array.
     * @param array destination array
     * @param chain 0-based index of chain this writer uses
     * @param offset offset in the destination array for the start of the write
     * @param skip number of parameters at the head of written data which are
     *             skipped (used e.g. for generated quantities)
     */
    ArrayWriter(Array & array, size_t chain, size_t offset=0, size_t skip=0);
    
    /// @addtogroup writer_Interface Interface of std::callbacks::writer
    /// @{
    void operator()(std::vector<std::string> const & names) override;
    void operator()(std::vector<double> const & state) override;
    void operator()(std::string const & message) override;

#if STAN_MAJOR < 2 || STAN_MAJOR == 2 && STAN_MINOR <= 36
    void operator()(
        Eigen::Ref<Eigen::Matrix<double, -1, -1>> const & values) override;
#else
    void operator()(Eigen::Matrix<double, -1, -1> const & values) override;
    void operator()(Eigen::Matrix<double, -1, 1> const & values) override;
    void operator()(Eigen::Matrix<double, 1, -1> const & values) override;
#endif
    /// @}
    
    std::vector<std::string> const & names() const;
    
private:
    Array & _array;
    size_t _chain, _offset, _skip, _draw;
    std::vector<std::string> _names;
    std::map<size_t, std::vector<std::string>> _messages;
    
    template<typename T>
    void _write_1d_container(T const & values)
    {
        using namespace xt::placeholders;
        
        std::size_t const size = values.size()-this->_skip;
        auto const source = xt::adapt(
            values.data()+this->_skip, size, xt::no_ownership(),
            std::vector<std::size_t>{size});
        auto target = xt::view(
            this->_array, xt::range(this->_offset, _), this->_chain, this->_draw);
        target = source;
        
        ++this->_draw;
    }
};

}

#endif // _f5319195_814d_49c2_8186_b46578694468
