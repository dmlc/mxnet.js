// To run:
//   docker run -v "$PWD":/usr/src/app -w /usr/src/app node:4 node test_on_node.js

// External Includes
var mx = require("./mxnet_predict.js");

function runModel(modelJson, cat) {
    console.log("  ");
    console.log("  ");
    console.log("Running model %s: ", modelJson);
    console.log("  ");

    var model = require(modelJson);
    pred = new mx.Predictor(model, {'data': [1, 3, 224, 224]});
    pred.setinput('data', cat);
    var nleft = 1;

    var start = new Date().getTime();
    var end   = new Date().getTime();
    var time  = (end - start) / 1000;

    for (var step = 0; nleft != 0; ++step) {
      nleft = pred.partialforward(step);
      end = new Date().getTime();
      time = (end - start) / 1000;
      console.log("    progress " + (step+1) + "/" + (nleft+step+1) + "  Time=" + time + "s");
    }
    out = pred.output(0);

    out = pred.output(0);
    var index = new Array();
    for (var i=0;i<out.data.length;i++) {
        index[i] = i;
    }
    index.sort(function(a,b) {return out.data[b]-out.data[a];});

    max_output = 10;
    console.log("Finished. Top %d predictions: ", max_output);
    for (var i = 0; i < max_output; i++) {
        console.log("    [%d]: %s, PROB=%d%", (i+1), model.synset[index[i]], out.data[index[i]]*100);
    }
    pred.destroy();
}


//
// Prepare input data
//
var cat_encoded = require("./data/cat.base64.json");
var decode = mx.base64Decode(cat_encoded);
var decoded = new Float32Array(decode.buffer);
var cat = mx.ndarray(decoded, [1, 3, 224, 224]);


//
// Models
//
// -- run "./models/prepare_models.sh -all" first to run the other two
var modelJSONs = [ "./model/inception-bn-model.json",
                   "./model/squeezenet-model.json",
                   "./model/resnet-model.json",
                   "./model/nin-model.json"]

//
// Run all models
//
for (var i=0; i<modelJSONs.length-2; i++) {
    runModel(modelJSONs[i], cat);
}
