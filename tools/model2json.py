#!/usr/bin/env python
"""Simple util to convert mxnet model to json format."""
import sys
import json
import base64

if len(sys.argv) < 4:
    print('Usage: <output.json> <symbol.json> <model.param> [mean_image.nd] [synset]')
    exit(0)

symbol_json = open(sys.argv[2]).read()
model = base64.b64encode(bytes(open(sys.argv[3], 'rb').read()))
mean_image = None
synset = None

if len(sys.argv) > 4:
    mean_image = base64.b64encode(bytes(open(sys.argv[4], 'rb').read()))

if len(sys.argv) > 5:
    synset = [l.strip() for l in open(sys.argv[5]).readlines()]

with open(sys.argv[1], 'w') as fo:
    fo.write('{\n\"symbol\":\n')
    fo.write(symbol_json)
    if synset:
        fo.write(',\n\"synset\": ')
        fo.write(json.dumps(synset))
    fo.write(',\n\"parambase64\": \"')
    fo.write(model)
    fo.write('\"\n')
    if mean_image is not None:
        fo.write(',\n\"meanimgbase64\": \"')
        fo.write(mean_image)
        fo.write('\"\n')
    fo.write('}\n')
