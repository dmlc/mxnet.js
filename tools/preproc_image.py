import sys, os
curr_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
sys.path.append("../mxnet/amalgamation/python/")
from mxnet_predict import Predictor, load_ndarray_file
import json
import numpy as np
import base64
from skimage import io, transform

jsonmodel = json.loads(open('inception-bn-model.json').read())
mean_img = load_ndarray_file(base64.b64decode(jsonmodel['meanimgbase64']))["mean_img"]

def PreprocessImage(path):
    # load image
    img = io.imread(path)
    print("Original Image Shape: ", img.shape)
    # we crop image from center
    short_egde = min(img.shape[:2])
    yy = int((img.shape[0] - short_egde) / 2)
    xx = int((img.shape[1] - short_egde) / 2)
    crop_img = img[yy : yy + short_egde, xx : xx + short_egde]
    # resize to 224, 224
    resized_img = transform.resize(crop_img, (224, 224))
    # convert to numpy.ndarray
    sample = np.asarray(resized_img) * 255
    # swap channel from RGB to BGR
    sample = sample[:, :, [2,1,0]]
    # swap axes to make image from (224, 224, 3) to (3, 224, 224)
    sample = np.swapaxes(sample, 0, 2)
    sample = np.swapaxes(sample, 1, 2)
    # sub mean
    normed_img = sample - mean_img
    normed_img.resize(1, 3, 224, 224)
    return normed_img

batch = PreprocessImage('./cat.png')
batch = batch.astype('float32')
buf = np.getbuffer(batch)
data = base64.b64encode(bytes(buf))

with open('cat.base64.json', 'w') as fo:
    fo.write('\"')
    fo.write(data)
    fo.write('\"')

