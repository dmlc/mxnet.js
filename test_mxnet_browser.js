// adapted from convnetjs
function logEvent(str) {
  console.log(str);
  var d = document.createElement('div');
  d.innerHTML = str;
  document.getElementById('result').appendChild(d);
}

function start() {
   $.getJSON("../model/inception-bn-model.json", function(model) {
     $.getJSON("../data/cat.base64.json", function(catb64) {
       logEvent("start...");
       // data was converted from json which was preprocessed.
       // TODO: add preprocessing pipeline.
       var buf = base64Decode(catb64);
       var decoded = new Float32Array(buf.buffer);
       var cat = ndarray(decoded, [1, 3, 224, 224]);
       pred = new Predictor(model, {'data': [1, 3, 224, 224]});
       pred.setinput('data', cat);
       logEvent("start... prediction... this can take a while");
       // delay 1sec before running prediction, so the log event renders on webpage.
       setTimeout(function(){
         pred.forward();
         logEvent("finished prediction...");
         out = pred.output(0);
         max_index = 0;
         for (var i = 0; i < out.data.length; ++i) {
           if (out.data[max_index] < out.data[i]) max_index = i;
         }
         logEvent('Top-1: ' + model.synset[max_index] + ', value=' + out.data[max_index]);
         pred.destroy();
       }, 1000);
     });
   });
}