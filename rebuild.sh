#!/bin/bash
# This script syncs libmxnet_predict.js from mxnet repo
# This should rarely be ran, as long as the predictor works
echo "Rebuild libmxnet_predict.js from MXNet with emscripten"
cd mxnet/amalgamation/
rm mxnet_predict-all.cc
make libmxnet_predict.js MIN=1
cd -
cp mxnet/amalgamation/libmxnet_predict.js* .


