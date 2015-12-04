#!/bin/bash
cd ../mxnet/amalgamation/
rm mxnet_predict-all.cc
make libmxnet_predict.js MIN=1
cp libmxnet_predict.js* ../../mxnetjs/
cd -
