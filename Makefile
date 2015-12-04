ifndef MXNET
	MXNET=mxnet
endif

.PHONY: rebuild

# This script syncs libmxnet_predict.js from mxnet repo
# This should rarely be ran, as long as the predictor works
# type make rebuild
rebuild:
	echo "Rebuild libmxnet_predict.js from MXNet with emscripten"
	cd $(MXNET)/amalgamation/; rm -f mxnet_predict-all.cc; make libmxnet_predict.js MIN=1; cd -
	cp $(MXNET)/amalgamation/libmxnet_predict.js* .



