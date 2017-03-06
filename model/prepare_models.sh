#!/usr/bin/env bash
script_name=$0

USAGE (){
    echo "  "
    echo "  "
    echo "Script downloads model from the MXNet Model Gallery "
    echo "  and prepares a combined JSON file containing the  "
    echo "  computation graph and weights."
    echo "Usage:"
    echo "${script_name} [-squeezenet] [-nin] [-caffenet] [-resnet] [-inceptionbn]"
    echo "  "
    echo "  "
}




# model2json.py script does the same as this function
prep_json_for_js(){
  # To call function
  #prep_json_for_js Inception-BN-symbol Inception-BN-0126.params

  outFile=$1
  inFile=$2
  paramsFile=$3
  synsetFile=$4

  jsName=$outFile


  # Create Symbol + params file for JSON
  cp $inFile $jsName
  sed -i '1s/^/{\n"symbol":\n/' $jsName
  sed -i '$s/$/,/' $jsName
  echo -en "\n" >> $jsName
  cat $synsetFile | sed 's/.*/"&",/' | tr '\n' ' ' | sed 's/.*/"synset": [&],/' | sed 's/, ],/],/g' >> $jsName
  echo -en "\n" >> $jsName
  base64 -w 0 $paramsFile | sed 's/.*/"parambase64": "&"/' >> $jsName
  echo -en "\n" >> $jsName
  echo } >> $jsName
}





prep_resnet_model(){
  echo "Preparing resnet18 model..."

  echo "    Downloading model from gallery..."
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/synset.txt
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/resnet/18-layers/resnet-18-0000.params
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/resnet/18-layers/resnet-18-symbol.json

  echo "   Running script to generate json model for JS..."
  prep_json_for_js ../resnet-model.json resnet-18-symbol.json resnet-18-0000.params synset.txt

}

prep_vgg_model(){
  echo "Preparing vgg-19 model..."

  echo "    Downloading model from gallery..."
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/synset.txt
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/vgg/vgg19-0000.params
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/vgg/vgg19-symbol.json

  echo "   Running script to generate json model for JS..."
  prep_json_for_js ../vgg-model.json vgg19-symbol.json vgg19-0000.params synset.txt

}

prep_nin_model(){
  echo "Preparing caffenet model..."

  echo "    Downloading model from gallery..."
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/synset.txt
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/nin/nin-0000.params
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/nin/nin-symbol.json

  echo "   Running script to generate json model for JS..."
  prep_json_for_js ../nin-model.json nin-symbol.json nin-0000.params synset.txt

}

prep_caffenet_model(){
  echo "Preparing caffenet model..."

  echo "    Downloading model from gallery..."
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/synset.txt
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/caffenet/caffenet-symbol.json
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/caffenet/caffenet-0000.params

  echo "   Running script to generate json model for JS..."
  prep_json_for_js ../caffenet-model.json caffenet-symbol.json caffenet-0000.params synset.txt

}

prep_squeezenet_model(){
  echo "Preparing squeezenet model..."

  echo "    Downloading model from gallery..."
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/synset.txt
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/squeezenet/squeezenet_v1.0-symbol.json
  wget --no-check-certificate http://data.dmlc.ml/models/imagenet/squeezenet/squeezenet_v1.0-0000.params

  echo "   Running script to generate json model for JS..."
  prep_json_for_js ../squeezenet-model.json squeezenet_v1.0-symbol.json squeezenet_v1.0-0000.params synset.txt

}

prep_inception_model(){
  echo "Preparing inceptionbn model..."

  echo "    Downloading inception model from gallery..."
  wget --no-check-certificate http://data.dmlc.ml/mxnet/models/imagenet/inception-bn.tar.gz
  tar -zxvf inception-bn.tar.gz

  echo "   Running script to generate json model for JS..."
  prep_json_for_js ../inception-bn-model.json Inception-BN-symbol.json Inception-BN-0126.params synset.txt

}


#
# Parse command-line arguments
#
while [ "$1" != "" ]; do
  case $1 in
    -help)
      USAGE
      exit 0;;
      -all | -inceptionbn | -squeezenet | -nin | -caffenet | -resnet | -vgg)
      TYPE=${1:1};;
  esac
  shift
done

if [ ! "$TYPE" ]; then
  echo "You must specify a test type"
  USAGE
  exit 0
fi


#
# Create temp dir
#
THIS_DIR=$(cd `dirname $0`; pwd)
TEMP_DIR="${THIS_DIR}/temp/"

if [[ ! -d "${TEMP_DIR}" ]]; then
  echo "${TEMP_DIR} doesn't exist, will create one";
  mkdir -p ${TEMP_DIR}
fi
cd ${TEMP_DIR}

#
# Prepare models
#
case $TYPE in
  all)
    echo "Preparing all models..."
    prep_nin_model
    prep_inception_model
    prep_squeezenet_model
    prep_resnet_model
    ;;
  nin)
    prep_nin_model
    ;;
  inceptionbn)
    prep_inception_model
    ;;
  squeezenet)
    prep_squeezenet_model
    ;;
  resnet)
    prep_resnet_model
    ;;
  vgg)
    prep_vgg_model
    ;;
  caffenet)
    prep_caffenet_model
    ;;
esac

echo "   Cleaning..."
sleep 2
cd ${THIS_DIR}
rm -rf ${TEMP_DIR}

echo "Done."
echo " "
echo "Contents for this dir:"
ls -ltrh
