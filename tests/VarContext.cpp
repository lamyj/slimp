#define BOOST_TEST_MODULE VarContext
#include <boost/test/unit_test.hpp>

#include "slimp/VarContext.h"

BOOST_AUTO_TEST_CASE(Empty)
{
    slimp::VarContext const context;
    std::vector<std::string> names;
    
    context.names_i(names);
    BOOST_TEST(names.empty());
    
    context.names_r(names);
    BOOST_TEST(names.empty());
}

BOOST_AUTO_TEST_CASE(Int)
{
    slimp::VarContext context;
    context.set("int_key", 42);
    
    std::vector<std::string> names;
    context.names_i(names);
    BOOST_TEST(names == std::vector<std::string>{"int_key"});
    
    BOOST_TEST(context.contains_i("int_key"));
    BOOST_TEST(context.contains_r("int_key"));
    
    BOOST_TEST(context.dims_i("int_key") == std::vector<size_t>());
    
    BOOST_TEST(context.vals_i("int_key") == std::vector<int>{42});
    BOOST_TEST(context.vals_r("int_key") == std::vector<double>{42.});
}

BOOST_AUTO_TEST_CASE(Double)
{
    slimp::VarContext context;
    context.set("double_key", 42.);
    
    std::vector<std::string> names;
    context.names_r(names);
    BOOST_TEST(names == std::vector<std::string>{"double_key"});
    
    BOOST_TEST(context.contains_r("double_key"));
    
    BOOST_TEST(context.dims_r("double_key") == std::vector<size_t>());
    
    BOOST_TEST(context.vals_r("double_key") == std::vector<double>{42.});
}

BOOST_AUTO_TEST_CASE(IntArray)
{
    auto const array = xt::xarray<uint8_t, xt::layout_type::row_major>{
        {1, 2, 3}, {4, 5, 6}};
    slimp::VarContext context;
    context.set("int_key", array);
    
    std::vector<std::string> names;
    context.names_i(names);
    BOOST_TEST(names == std::vector<std::string>{"int_key"});
    
    BOOST_TEST(context.contains_i("int_key"));
    BOOST_TEST(context.contains_r("int_key"));
    
    BOOST_TEST((context.dims_i("int_key") == std::vector<size_t>{2, 3}));
    
    BOOST_TEST((
        context.vals_i("int_key") == std::vector<int>{1, 4, 2, 5, 3, 6}));
    BOOST_TEST((
        context.vals_r("int_key")
        == std::vector<double>{1., 4., 2., 5., 3., 6.}));
}

BOOST_AUTO_TEST_CASE(DoubleArray)
{
    auto const array = xt::xarray<float, xt::layout_type::row_major>{
        {1.f, 2.f, 3.f}, {4.f, 5.f, 6.f}};
    slimp::VarContext context;
    context.set("double_key", array);
    
    std::vector<std::string> names;
    context.names_r(names);
    BOOST_TEST(names == std::vector<std::string>{"double_key"});
    
    BOOST_TEST(context.contains_r("double_key"));
    
    BOOST_TEST((context.dims_r("double_key") == std::vector<size_t>{2, 3}));
    
    BOOST_TEST((
        context.vals_r("double_key")
        == std::vector<double>{1., 4., 2., 5., 3., 6.}));
}

