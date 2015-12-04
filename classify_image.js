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
    var sourceWidth = image.width;
    var sourceHeight = image.height;
    var shortEdge = Math.min(this.width, this.height);
    var yy = Math.floor((sourceHeight - shortEdge) / 2);
    var xx = Math.floor((sourceWidth - shortEdge) / 2);
    logEvent("shortEdge=" + shortEdge);

    context.drawImage(image,
                      xx, yy,
                      shortEdge, shortEdge,
                      0, 0,
                      targetLen, targetLen);
    canvas.height = targetLen;
    canvas.width = targetLen;

    var imgdata = context.getImageData(0, 0, targetLen, targetLen);
    var data = new Float32Array(targetLen * targetLen * 3);
    var stride = targetLen * targetLen;
    for (var i = 0; i < stride; ++i) {
      data[stride * 0 + i] = imgdata.data[i * 4 + 0];
      data[stride * 1 + i] = imgdata.data[i * 4 + 1];
      data[stride * 2 + i] = imgdata.data[i * 4 + 2];
    }
    if (typeof meanimg !== 'undefined') {
      for (var i = 0; i < data.length; ++i) {
        data[i] = data[i] - meanimg.data[i];
      }
    } else {
      // use 117 as mean by default.
      for (var i = 0; i < data.length; ++i) {
        data[i] = data[i] - 117;
      }
    }
    var nd = ndarray(data, [1, 3, targetLen, targetLen]);
    callback(nd);
  };
  image.src = url;
}

function start() {
   $.getJSON("../model/fastpoor.json", function(model) {
       var url = document.getElementById("imageURL").value;
       pred = new Predictor(model, {'data': [1, 3, 224, 224]});
       preproc(url, 224, pred.meanimg,  function(nd) {
           pred.setinput('data', nd);
           logEvent("start... prediction... this can take a while");
           // delay 1sec before running prediction, so the log event renders on webpage.
           setTimeout(function(){
               var start = new Date().getTime();
               pred.forward();
               logEvent("finished prediction...");
               out = pred.output(0);
               max_index = 0;
               for (var i = 0; i < out.data.length; ++i) {
                   if (out.data[max_index] < out.data[i]) max_index = i;
               }
               var end = new Date().getTime();
               var time = (end - start) / 1000;
               logEvent('Top-1: ' + model.synset[max_index] + ', value=' + out.data[max_index] + ', time-cost=' + time + 'secs');
               pred.destroy();
           }, 1000);
       });
   });
}