/**
 * Image preprocessing module in NodeJS
 */
var fs = require('fs')
    , gm = require('gm').subClass({imageMagick: true});


function LoadAndPreproc(imageurl, min_size, meanimg, callback) {
  img = gm(imageurl)
      .resize(min_size, min_size)
      .toBuffer('');
  return img;
}

module.exports.LoadAndPreproc = LoadAndPreproc;