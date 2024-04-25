#ifndef _f5319195_814d_49c2_8186_b46578694468
#define _f5319195_814d_49c2_8186_b46578694468

#include <cstdint>
#include <string>
#include <vector>

// WARNING: Stan must be included before Eigen so that the plugin system is
// active. https://discourse.mc-stan.org/t/includes-in-user-header/26093
#include <stan/math.hpp>

#include <Eigen/Dense>
#include <pybind11/numpy.h>
#include <stan/callbacks/writer.hpp>

#include "slimp/api.h"

/**
 * @brief Stan writer to a user-provided numpy array.
 *
 * The array must be a F-style, 3D array of doubles with shape chains × draws ×
 * parameters.
 */
class SLIMP_API ArrayWriter: public stan::callbacks::writer
{
public:
    using Array = pybind11::array_t<double, pybind11::array::f_style>;
    
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
    void operator()(Eigen::Ref<Eigen::Matrix<double, -1, -1>> const & values) override;
    /// @}
    
    std::vector<std::string> const & names() const;
    
private:
    Array & _array;
    size_t _chain, _offset, _skip, _draw;
    std::vector<std::string> _names;
    std::map<size_t, std::vector<std::string>> _messages;
};

#endif // _f5319195_814d_49c2_8186_b46578694468
