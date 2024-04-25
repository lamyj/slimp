#ifndef _5eca19ab_3261_414f_8dd3_ce485c9e547d
#define _5eca19ab_3261_414f_8dd3_ce485c9e547d

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>
#include <stan/io/var_context.hpp>

#include "slimp/api.h"

/// @brief Input data based on a Python dictionary
class SLIMP_API VarContext: public stan::io::var_context
{
public:
    VarContext() = default;
    VarContext(VarContext const &) = default;
    VarContext(VarContext &&) = default;
    ~VarContext() = default;
    VarContext & operator=(VarContext const &) = default;
    VarContext & operator=(VarContext &&) = default;
    
    VarContext(pybind11::dict dictionary);
    
    /// @addtogroup var_context_Interface Interface of std::io::var_context
    /// @{
    bool contains_r(std::string const & name) const override;
    std::vector<double> vals_r(std::string const & name) const override;
    std::vector<std::complex<double>> vals_c(
        std::string const & name) const override;
    std::vector<size_t> dims_r(std::string const & name) const override;
    bool contains_i(std::string const & name) const override;
    std::vector<int> vals_i(std::string const & name) const override;
    std::vector<size_t> dims_i(std::string const & name) const override;
    void names_r(std::vector<std::string> & names) const override;
    void names_i(std::vector<std::string> & names) const override;
    void validate_dims(
        std::string const& stage, std::string const & name,
        std::string const& base_type,
        std::vector<size_t> const & dims_declared) const override;
    /// @}

private:
    std::unordered_map<std::string, std::vector<int>> _vals_i;
    std::unordered_map<std::string, std::vector<double>> _vals_r;
    std::unordered_map<std::string, std::vector<size_t>> _dims_i, _dims_r;
};

#endif // _5eca19ab_3261_414f_8dd3_ce485c9e547d
