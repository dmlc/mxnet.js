var mx = require("./mxnet_predict.js");
var model = require("./model/fastpoor.json");
var cat_encoded = require("./data/cat.base64.json");
var decode = mx.base64Decode(cat_encoded);
var decoded = new Float32Array(decode.buffer);
var cat = mx.ndarray(decoded, [1, 3, 224, 224]);

pred = new mx.Predictor(model, {'data': [1, 3, 224, 224]});
pred.setinput('data', cat);
console.log("Here");
var nleft = 1;
for (var step = 0; nleft != 0; ++step) {
  nleft = pred.partialforward(step);
  console.log("progress " + (step+1) + "/" + (nleft+step+1));
}
out = pred.output(0);
max_index = 0;
for (var i = 0; i < out.data.length; ++i) {
  if (out.data[max_index] < out.data[i]) max_index = i;
}
console.log('Top-1: ' + model.synset[max_index] + ', value=' + out.data[max_index]);
pred.destroy();

