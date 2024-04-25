#ifndef _f8884fab_b495_4452_ae61_fc546f8758fb
#define _f8884fab_b495_4452_ae61_fc546f8758fb

#include <functional>
#include <map>
#include <ostream>
#include <string>

#include <stan/model/model_header.hpp>

#include "slimp/api.h"

/// @brief Register and create models.
class SLIMP_API Factory
{
public:
    using Creator = std::function<
        stan::model::model_base & (
            stan::io::var_context &, unsigned int, std::ostream *)>;
    
    static Factory & instance();
    
    ~Factory() = default;
    Factory(Factory const &) = delete;
    Factory(Factory &&) = delete;
    Factory & operator=(Factory const &) = delete;
    Factory & operator=(Factory &&) = delete;
    
    void register_(std::string const & name, Creator creator);
    stan::model::model_base & get(
        std::string const & name, stan::io::var_context & context,
        unsigned int seed, std::ostream * stream) const;
    
private:
    static Factory * _instance;
    std::map<std::string, Creator> _creators;
    
    Factory() = default;
};

#endif // _f8884fab_b495_4452_ae61_fc546f8758fb
