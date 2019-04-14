#!/bin/bash

rm -rf build
mkdir build
cd build

cmake ..
make
make test

echo -e "\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>test details:"
cat Testing/Temporary/LastTest.log