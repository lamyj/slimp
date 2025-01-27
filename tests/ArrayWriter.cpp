#define BOOST_TEST_MODULE ArrayWriter
#include <boost/test/unit_test.hpp>

#include "slimp/ArrayWriter.h"

#include <xtensor/xview.hpp>

BOOST_AUTO_TEST_CASE(Names)
{
    xt::xarray<double> array({2, 3, 5}, 0.);
    
    slimp::ArrayWriter writer(array, 1);
    writer(std::vector<std::string>{"foo", "bar"});
    BOOST_TEST((writer.names() == std::vector<std::string>{"foo", "bar"}));
}

BOOST_AUTO_TEST_CASE(Array)
{
    xt::xarray<double> array({3, 2, 5}, 0.);
    
    slimp::ArrayWriter writer(array, 1, 1, 2);
    
    writer(std::vector<double>{42, 43, 1, 2});
    BOOST_TEST((
        xt::view(array, xt::all(), 0) == xt::zeros<double>({3, 5})));
    BOOST_TEST((
        xt::view(array, xt::all(), 1)
        == xt::xarray<double>{{0, 0, 0, 0, 0}, {1, 0, 0, 0, 0}, {2, 0, 0, 0, 0}}));
    
    writer(std::vector<double>{44, 45, 3, 4});
    BOOST_TEST((
        xt::view(array, xt::all(), 0) == xt::zeros<double>({3, 5})));
    BOOST_TEST((
        xt::view(array, xt::all(), 1)
        == xt::xarray<double>{{0, 0, 0, 0, 0}, {1, 3, 0, 0, 0}, {2, 4, 0, 0, 0}}));
}

BOOST_AUTO_TEST_CASE(Matrix1)
{
    xt::xarray<double> array({3, 2, 4}, 0.);
    
    slimp::ArrayWriter writer(array, 1, 1, 2);
    
    Eigen::MatrixXd matrix(4, 3);
    matrix(0, 0) = 42; matrix(0, 1) = 42; matrix(0, 2) = 42;
    matrix(1, 0) = 43; matrix(1, 1) = 43; matrix(1, 2) = 43;
    matrix(2, 0) = 1; matrix(2, 1) = 2; matrix(2, 2) = 3;
    matrix(3, 0) = 4; matrix(3, 1) = 5; matrix(3, 2) = 6;
    
    // NOTE: this test and the next one is based on the CSV-like output (i.e.
    // one row containing all the parameters per sample) of the following:
    // std::stringstream stream;
    // stan::callbacks::unique_stream_writer<std::stringstream> stan_writer{
    //     std::unique_ptr<std::stringstream>(&stream)};
    // stan_writer(matrix);
    // std::cout << "unique_stream_writer: \n" << stream.str() << std::endl;
    
    writer(matrix);
    
    BOOST_TEST((
        xt::view(array, xt::all(), 0) == xt::zeros<double>({3, 4})));
    BOOST_TEST((
        xt::view(array, xt::all(), 1)
        == xt::xarray<double>{
            {0, 0, 0, 0},
            {1, 2, 3, 0},
            {4, 5, 6, 0}}));
    
    writer(std::vector<double>{42, 43, 7, 8});
    
    BOOST_TEST((
        xt::view(array, xt::all(), 0) == xt::zeros<double>({3, 4})));
    BOOST_TEST((
        xt::view(array, xt::all(), 1)
        == xt::xarray<double>{
            {0, 0, 0, 0},
            {1, 2, 3, 7},
            {4, 5, 6, 8}}));
}

BOOST_AUTO_TEST_CASE(Matrix2)
{
    xt::xarray<double> array({3, 2, 4}, 0.);
    
    slimp::ArrayWriter writer(array, 1, 1, 2);
    
    writer(std::vector<double>{42, 43, 1, 4});
    
    Eigen::MatrixXd matrix(4, 3);
    matrix(0, 0) = 42; matrix(0, 1) = 42; matrix(0, 2) = 42;
    matrix(1, 0) = 43; matrix(1, 1) = 43; matrix(1, 2) = 43;
    matrix(2, 0) = 2; matrix(2, 1) = 3; matrix(2, 2) = 7;
    matrix(3, 0) = 5; matrix(3, 1) = 6; matrix(3, 2) = 8;
    
    writer(matrix);
    
    BOOST_TEST((
        xt::view(array, xt::all(), 0) == xt::zeros<double>({3, 4})));
    BOOST_TEST((
        xt::view(array, xt::all(), 1)
        == xt::xarray<double>{
            {0, 0, 0, 0},
            {1, 2, 3, 7},
            {4, 5, 6, 8}}));
}
