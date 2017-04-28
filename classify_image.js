
function logProgress(progress) {
  $('#myProgress')
        .css('width', progress+'%')
        .attr('aria-valuenow', progress);
}

function resetProgress() {
  $('#myProgress')
        .attr('class', 'progress-bar')
        .css('width', '0%')
        .attr('aria-valuenow', '0')
        .html('');
}

function logEvent(str) {
  console.log(str);
  var d = document.createElement('div');
  d.innerHTML = str;
  document.getElementById('result').appendChild(d);
}

function logError(message) {
  $('#myProgress')
        .attr('class', 'progress-bar progress-bar-danger')
        .css('width', '100%')
        .attr('aria-valuenow', 100).html(message);
  logEvent(message);
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
                      xx, yy,
                      shortEdge, shortEdge,
                      0, 0,
                      targetLen, targetLen);
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
  $(image).bind('error', function (event) {
    logError("Opps.. Failed to load image " + url);
  });
  image.src = url;
}

function start() {
   $.getJSON("./model/squeezenet-model.json", function(model) {
       var url = document.getElementById("imageURL").value;
       pred = new Predictor(model, {'data': [1, 3, 224, 224]});
       preproc(url, 224, pred.meanimg,  function(nd) {
           pred.setinput('data', nd);
           logEvent("start... prediction... this can take a while");
           // delay 1sec before running prediction, so the log event renders on webpage.
           var start = new Date().getTime();
           // print every 10%
           var print_step = 10;
           // reset progress bar
           resetProgress();

           function trainloop(step, nleft, next_goal, finish_callback) {
               if (nleft == 0) {
                 finish_callback(); return;
               }
               nleft = pred.partialforward(step);
               progress = (step + 1) / (nleft + step + 1) * 100;
               if (progress >= next_goal || progress == 100) {
                   logProgress(progress);
                   setTimeout(function() {
                       trainloop(step + 1, nleft, next_goal + print_step, finish_callback);
                   }, 1);
               } else {
                   setTimeout(function() {
                       trainloop(step + 1, nleft, next_goal, finish_callback);
                   }, 0);
               }
           }
           trainloop(0, 1, 0, function() {
              logEvent("finished prediction....");
              out = pred.output(0);
              var index = new Array();
              for (var i=0;i<out.data.length;i++) {
                index[i] = i;
              }
              max_output = Number(document.getElementById("max-output").value);
              logEvent("Max output = " + max_output);
              index.sort(function(a,b) {return out.data[b]-out.data[a];});
              var end = new Date().getTime();
              var time = (end - start) / 1000;
              logEvent("time-cost=" + time + " sec");
              for (var i = 0; i < max_output; i++) {
                logEvent('Top-' + (i+1) + ':' + model.synset[index[i]] + ', value=' + out.data[index[i]]);
              }
              pred.destroy();
           });
       });
   });
}
