/**
 * MXNet Javascript Library
 */
// Utility function section.
//-------------------------------------------
var IS_NODEJS = (typeof module !== 'undefined' && module.exports);
if (IS_NODEJS) {
  var Module = require("./libmxnet_predict.js");
}
// constants
var SIZEOF_POINTER = 4;
var SIZEOF_UINT = 4;
var SIZEOF_FLOAT = 4;

/**
 * Decode a base64 string into string.
 * @param b64 base64 encoded string
 * @return The decoded binary string.
 */
function base64Decode(b64) {
  if (IS_NODEJS) {
    var buf =  new Buffer(b64, "base64");
    var ret = new Uint8Array(buf.length);
    for (var i = 0; i < buf.length; ++i) {
      ret[i] = buf[i];
    }
    return ret;
  } else {
    var buf = window.atob(b64);
    var ret = new Uint8Array(buf.length);
    for (var i = 0; i < buf.length; ++i) {
      ret[i] = buf.charCodeAt(i);
    }
    return ret;
  }
}

/**
 * Create a NDArray representation in javascript.
 * @param data Float32Array The data array in the ndarray.
 * @param shape Uint32Array The shape of the array.
 * @return The constructed NDArray object.
 */
function ndarray(data, shape) {
  var data = Float32Array.from(data);
  var shape = Uint32Array.from(shape);
  var size = shape.reduce(function(a, b) { return a * b; }, 1);
  if (data.length != size) {
    throw "Size and shape mismatch";
  }
  return {'data': data, 'shape': shape};
}

/**
 * Create a Uint32Array from pointer space.
 * @param ptr The pointer address of the source data from C API.
 * @param length Length of the array.
 * @return The created new Uint32Array
 */
function Uint32ArrayFromPtr(ptr, length) {
  var srcbuf = new Uint32Array(Module.HEAPU32.buffer, ptr, length).slice(0);
  return new Uint32Array(srcbuf);
}

/**
 * Create a Float32Array from pointer space.
 * @param ptr The pointer address of the source data from C API.
 * @param length Length of the array.
 * @return The created new Float32Array
 */
function Float32ArrayFromPtr(ptr, length) {
  var srcbuf = new Float32Array(Module.HEAPF32.buffer, ptr, length).slice(0);
  return new Float32Array(srcbuf);
}

/**
 * Create a JS string from C pointer.
 * @param ptr The pointer address of the source data from C API.
 * @param length Length of the array.
 * @return The created new JS string.
 */
function CStringFromPtr(ptr) {
  var ret = []
  var ch = 1;
  while (ch != 0) {
    ch = Module.getValue(ptr, 'i8');
    if (ch != 0) {
      ret.push(String.fromCharCode(ch));
    }
    ++ptr;
  }
  return ret.join('');
}

// Library code that is runtime invariant
// --------------------------------------
_CWRAP_MXPredCreate = Module.cwrap
    ('MXPredCreate',
     'number',
     ['string', // const char* symbol_json_str
      'number', // const char* param_bytes(raw memory, not ascii string)
      'number', // size_t param_size
      'number', // int dev_type
      'number', // int dev_id
      'number', // mx_uint num_input_nodes,
      'number', // const char** input_keys,
      'number', // const mx_uint* input_shape_indptr
      'number', // const mx_uint* input_shape_data,
      'number'  // PredictorHandle* out
      ]);

_CWRAP_MXPredGetOutputShape = Module.cwrap
    ('MXPredGetOutputShape',
     'number',
     ['number', // PredictorHandle handle
      'number', // mx_uint index
      'number', // mx_uint** shape_data
      'number'] // mx_uint* shape_ndim
     );

_CWRAP_MXPredSetInput = Module.cwrap
    ('MXPredSetInput',
     'number',
     ['number', // PredictorHandle handle
      'string', // const char* key
      'number', // const mx_float* data
      'number'] // size_t size
     );

_CWRAP_MXPredForward = Module.cwrap
    ('MXPredForward',
     'number',
     ['number'] // PredictorHandle handle
     );

_CWRAP_MXPredPartialForward = Module.cwrap
    ('MXPredPartialForward',
     'number',
     ['number', // PredictorHandle handle
      'number', // int step
      'number'] // int* step_left
     );

_CWRAP_MXPredGetOutput = Module.cwrap
    ('MXPredGetOutput',
     'number',
     ['number', // PredictorHandle handle
      'number', // mx_uint index
      'number', // float* data
      'number'] // mx_uint size
     );

_CWRAP_MXPredFree = Module.cwrap
    ('MXPredFree',
     'number',
     ['number'] // PredictorHandle handle
     );

_CWRAP_MXNDListCreate = Module.cwrap
    ('MXNDListCreate',
     'number',
     ['number', // const char* nd_file_bytes (raw memory, not ascii string)
      'number', // size_t nd_file_size
      'number', // NDListHandle *out
      'number']  // int out_length
     );

_CWRAP_MXNDListGet = Module.cwrap
    ('MXNDListGet',
     'number',
     ['number', // NDListHandle handle
      'number', // mx_uint index
      'number', // const char** out_key,
      'number', // const float** out_data,
      'number', // const float** out_shape,
      'number'] // mx_uint* out_dim
     );

_CWRAP_MXNDListFree = Module.cwrap
    ('MXNDListGet',
     'number',
     ['number']  // NDListHandle handl
     );
// Implementations of Javascript API
//----------------------------------

/**
 * Load NDList from binary blob.
 *
 * @param binarr Uint8Array the binary format of ndarray list parameter.
 * @return Loaded object of NDArrays
 */
function NDListLoad(binarr) {
  // load handle
  var ptr_handle_out = Module._malloc(SIZEOF_POINTER);
  var ptr_data_bytes = Module._malloc(binarr.length);
  var ptr_out_length = Module._malloc(SIZEOF_UINT);
  Module.HEAPU8.set(binarr, ptr_data_bytes);
  _CWRAP_MXNDListCreate(ptr_data_bytes,
                        binarr.length,
                        ptr_handle_out,
                        ptr_out_length);
  var out_length = Module.getValue(ptr_out_length, 'i32');
  var handle = Module.getValue(ptr_handle_out, '*');
  Module._free(ptr_handle_out);
  Module._free(ptr_data_bytes);
  Module._free(ptr_out_length);
  // get data
  var ret = {};
  var ptr_out_key = Module._malloc(SIZEOF_POINTER);
  var ptr_out_data = Module._malloc(SIZEOF_POINTER);
  var ptr_out_shape = Module._malloc(SIZEOF_POINTER);
  var ptr_out_dim = Module._malloc(SIZEOF_UINT);

  for (var i = 0 ; i < out_length; ++i) {
    _CWRAP_MXNDListGet(handle, i, ptr_out_key,
                       ptr_out_data, ptr_out_shape,
                       ptr_out_dim);
    var out_key = CStringFromPtr(Module.getValue(ptr_out_key, '*'));
    var out_dim = Module.getValue(ptr_out_dim, 'i32');
    var out_shape = Uint32ArrayFromPtr(Module.getValue(ptr_out_shape, '*'), out_dim);
    var data_size = out_shape.reduce(function(a, b) { return a * b; }, 1);
    var out_data = Float32ArrayFromPtr(Module.getValue(ptr_out_data, '*'),
                                       data_size);
    ret[out_key] = ndarray(out_data, out_shape);
  }
  _CWRAP_MXNDListFree(handle);
  Module._free(ptr_out_key);
  Module._free(ptr_out_data);
  Module._free(ptr_out_shape);
  Module._free(ptr_out_dim);
  return ret;
}

/**
 * Create a predictor, this predictor must be explicitly
 * freed after use by calling predictor.destroy().
 *
 * @constructor
 * @param modelobj object, by loading mxnet json object
 * @param input_shape object, maps key to array
 */
function Predictor(modelobj, input_shapes) {
  // setup input  memory.
  var ptrarr_input_keys = [];
  var input_shape_indptr = [0];
  var input_shape_data = [];
  var offset = 0;
  for (var key in input_shapes) {
    var key_buf = Module._malloc(key.length + 1);
    Module.writeStringToMemory(key, key_buf);
    ptrarr_input_keys.push(key_buf);
    offset = offset + input_shapes[key].length;
    input_shape_indptr.push(offset);
    Array.prototype.push.apply(input_shape_data, input_shapes[key]);
  }
  var ptr_input_keys = Module._malloc(SIZEOF_POINTER * ptrarr_input_keys.length);
  for (var i = 0; i < ptrarr_input_keys.length; ++i) {
    Module.setValue(ptr_input_keys + i * SIZEOF_POINTER, ptrarr_input_keys[i], "*");
  }
  var ptr_handle_out = Module._malloc(SIZEOF_POINTER);
  symbol_json = JSON.stringify(modelobj['symbol']);
  param_bytes = base64Decode(modelobj['parambase64']);
  input_shape_indptr = Uint32Array.from(input_shape_indptr);
  input_shape_data = Uint32Array.from(input_shape_data);
  var ptr_param_bytes = Module._malloc(param_bytes.length);
  var ptr_input_shape_indptr = Module._malloc(input_shape_indptr.length * input_shape_indptr.BYTES_PER_ELEMENT);
  var ptr_input_shape_data = Module._malloc(input_shape_data.length * input_shape_data.BYTES_PER_ELEMENT);
  Module.HEAPU8.set(param_bytes, ptr_param_bytes);
  Module.HEAPU8.set(new Uint8Array(input_shape_indptr.buffer), ptr_input_shape_indptr);
  Module.HEAPU8.set(new Uint8Array(input_shape_data.buffer), ptr_input_shape_data);
  // call function
  _CWRAP_MXPredCreate(symbol_json,
                      ptr_param_bytes,
                      param_bytes.length,
                      1, 0,
                      ptrarr_input_keys.length,
                      ptr_input_keys,
                      ptr_input_shape_indptr,
                      ptr_input_shape_data,
                      ptr_handle_out);
  var handle = Module.getValue(ptr_handle_out, '*');
  // free space
  Module._free(ptr_input_keys);
  Module._free(ptr_handle_out);
  Module._free(ptr_param_bytes);
  Module._free(ptr_input_shape_data);
  Module._free(ptr_input_shape_indptr);
  for (var i = 0; i < ptrarr_input_keys.length; ++i) {
    Module._free(ptrarr_input_keys[i]);
  }
  // setup handle
  this.handle = handle;
  this.input_shapes = input_shapes;
  // setup mean image
  if ('meanimgbase64' in modelobj) {
    binarr = base64Decode(modelobj['meanimgbase64']);
    dict = NDListLoad(binarr);
    this.meanimg = dict.mean_img;
  }
}

Predictor.prototype = {
  /**
   * Destroy the predictor, need to be called after use of predictor.
   */
  destroy : function() {
    _CWRAP_MXPredFree(this.handle);
  },
  /**
   * Run forward inference.
   */
  forward : function () {
    _CWRAP_MXPredForward(this.handle);
  },
  /**
   * Run a partial forward inference.
   * This can be used to get interactive progress of prediction.
   * The forward start from step 0, and keep calling partialforward
   * with increasing step, until the returned step_left = 0.
   * var nleft = 1;
   * for (var step = 0; nleft != 0; ++step) {
   *   nleft = pred.partialforward(step);
   *   console.log("progress " + step + "/" + (nleft+step));
   * }
   *
   * @param step The current step of inference.
   * @return step_left number of step left to call inference.
   */
  partialforward : function (step) {
    var ptr_nleft = Module._malloc(SIZEOF_UINT);
    _CWRAP_MXPredPartialForward(this.handle, step, ptr_nleft);
    var nleft = Module.getValue(ptr_nleft, 'i32');
    Module._free(ptr_nleft);
    return nleft;
  },
  /**
   * Get i-th output of the predictor after calling forward.
   * @param index The output index.
   * @return an NDArray representation of i-th output.
   */
  output : function(index) {
    var ptr_shape_data = Module._malloc(SIZEOF_POINTER);
    var ptr_ndim = Module._malloc(SIZEOF_UINT);
    _CWRAP_MXPredGetOutputShape(this.handle, index, ptr_shape_data, ptr_ndim);
    ndim = Module.getValue(ptr_ndim, 'i32');
    out_shape = Uint32ArrayFromPtr(Module.getValue(ptr_shape_data, '*'), ndim);
    Module._free(ptr_shape_data);
    Module._free(ptr_ndim);
    var data_size = out_shape.reduce(function(a, b) { return a * b; }, 1);
    var ptr_data = Module._malloc(SIZEOF_FLOAT * data_size);
    _CWRAP_MXPredGetOutput(this.handle, index, ptr_data, data_size);
    var out_data = Float32ArrayFromPtr(ptr_data, data_size);
    Module._free(ptr_data);
    return ndarray(out_data, out_shape);
  },
  /**
   * Set the input of key to be data
   * @param key The key of the input, usually is "data".
   * @param nd NDArray representation created by ndarray
   * @seealso ndarray
   */
  setinput : function(key, nd) {
    var ptr_data = Module._malloc(nd.data.length * nd.data.BYTES_PER_ELEMENT);
    Module.HEAPU8.set(new Uint8Array(nd.data.buffer), ptr_data);
    _CWRAP_MXPredSetInput(this.handle, key, ptr_data, nd.data.length);
    Module._free(ptr_data);
  }
};

// export things in node
if (typeof module !== 'undefined' && module.exports) {
  module.exports.Predictor = Predictor;
  module.exports.base64Decode = base64Decode;
  module.exports.ndarray = ndarray;
}
