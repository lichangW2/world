//
// Created by CJ on 13/04/2019.
//

#define BOOST_TEST_MODULE IPCV_TEST
#include <boost/test/unit_test.hpp>
#include <iostream>
#include "IPConv.h"

//================================

BOOST_AUTO_TEST_SUITE(FailTest)
BOOST_AUTO_TEST_CASE(FailTest)
{
    unsigned int result;
    char const *ipstr="172.168.5.01";
    int ret=IPConv(ipstr,&result);
    BOOST_CHECK_EQUAL(-1, ret);

    char *tmp_str;
    ret=IPConv(tmp_str,&result);
    BOOST_CHECK_EQUAL(-1, ret);

    unsigned int *tmp_int;
    ret=IPConv(ipstr,tmp_int);
    BOOST_CHECK_EQUAL(-1, ret);

    ipstr="172.16  8.5.1";
    ret=IPConv(ipstr,&result);
    BOOST_CHECK_EQUAL(-1, ret);

    ipstr="  172.168.5.1";
    ret=IPConv(ipstr,&result);
    BOOST_CHECK_EQUAL(-1, ret);

    ipstr="172.168.5.1  ";
    ret=IPConv(ipstr,&result);
    BOOST_CHECK_EQUAL(-1, ret);

    ipstr="172.9168.5.1";
    ret=IPConv(ipstr,&result);
    BOOST_CHECK_EQUAL(-1, ret);

    ipstr="1c72.168.5.1";
    ret=IPConv(ipstr,&result);
    BOOST_CHECK_EQUAL(-1, ret);

    ipstr=".168.5.1";
    ret=IPConv(ipstr,&result);
    BOOST_CHECK_EQUAL(-1, ret);

    ipstr="172.168.5.";
    ret=IPConv(ipstr,&result);
    BOOST_CHECK_EQUAL(-1, ret);

    ipstr="172.168.5";
    ret=IPConv(ipstr,&result);
    BOOST_CHECK_EQUAL(-1, ret);

    ipstr="0.0.0.000";
    ret=IPConv(ipstr,&result);
    BOOST_CHECK_EQUAL(-1, ret);
}
BOOST_AUTO_TEST_SUITE_END()

//====================================

BOOST_AUTO_TEST_SUITE(PassTest)
BOOST_AUTO_TEST_CASE(PassTest)
{
    unsigned int result;
    char const *ipstr="172.168.5.1";
    BOOST_CHECK_EQUAL(0, IPConv(ipstr,&result));
    BOOST_CHECK_EQUAL(result,2896692481);

    ipstr="172.168.0.1";
    BOOST_CHECK_EQUAL(0, IPConv(ipstr,&result));
    BOOST_CHECK_EQUAL(result,2896691201);

    ipstr="172  . 168.       5.1";
    BOOST_CHECK_EQUAL(0, IPConv(ipstr,&result));
    BOOST_CHECK_EQUAL(result,2896692481);

    ipstr="0.0.0.0";
    BOOST_CHECK_EQUAL(0, IPConv(ipstr,&result));
    BOOST_CHECK_EQUAL(result,0);

    ipstr="255.255.255.255";
    BOOST_CHECK_EQUAL(0, IPConv(ipstr,&result));
    BOOST_CHECK_EQUAL(result,4294967295);
}
BOOST_AUTO_TEST_SUITE_END()