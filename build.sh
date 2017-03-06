
THIS_DIR=$(cd `dirname $0`; pwd)

# MXNet checkout directory
MXNET=mxnet

# Build
echo "Rebuild libmxnet_predict.js from MXNet with emscripten"
rm -rf libmxnet_predict.js*
cd ${MXNET}
git pull
git submodule update --init --recursive
cd amalgamation/
make clean libmxnet_predict.js MIN=1 EMCC="docker run -v ${PWD}:/src apiaryio/emcc emcc"
cp libmxnet_predict.js* ../..
cd ${THIS_DIR}
