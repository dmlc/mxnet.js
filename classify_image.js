function logEvent(str) {
  console.log(str);
  var d = document.createElement('div');
  d.innerHTML = str;
  document.getElementById('result').appendChild(d);
}

function preproc(url, targetLen, meanimg, callback) {
  var canvas = document.getElementById('myCanvas');
  var context = canvas.getContext('2d');
  var image = new Image();
  var targetLen = 224;
  image.setAttribute('crossOrigin', 'anonymous');
  image.onload = function() {
    var sourceWidth = this.width;
    var sourceHeight = this.height;
    var shortEdge = Math.min(this.width, this.height);
    var yy = Math.floor((sourceHeight - shortEdge) / 2);
    var xx = Math.floor((sourceWidth - shortEdge) / 2);
    context.drawImage(image,
                      yy, xx,
                      shortEdge, shortEdge,
                      0, 0, targetLen, targetLen);
    var imgdata = context.getImageData(0, 0, targetLen, targetLen);
    var data = new Float32Array(targetLen * targetLen * 3);
    var stride = targetLen * targetLen;
    for (var i = 0; i < stride; ++i) {
      data[stride * 0 + i] = imgdata.data[i * 4 + 0];
      data[stride * 1 + i] = imgdata.data[i * 4 + 1];
      data[stride * 2 + i] = imgdata.data[i * 4 + 2];
    }
    for (var i = 0; i < data.length; ++i) {
      data[i] = data[i] - meanimg.data[i];
    }
    var nd = ndarray(data, [1, 3, targetLen, targetLen]);
    callback(nd);
  };
  image.src = url;
}

function start() {
   $.getJSON("../model/inception-bn-model.json", function(model) {
       var url = document.getElementById("imageURL").value;
       pred = new Predictor(model, {'data': [1, 3, 224, 224]});

       preproc(url, 224, pred.meanimg,  function(nd) {
           pred.setinput('data', nd);
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