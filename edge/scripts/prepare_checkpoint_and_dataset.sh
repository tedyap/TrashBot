#!/bin/bash
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Exit script on error.
set -e
# Echo each command, easier for debugging.
set -x

usage() {
  cat << END_OF_USAGE
  Downloads checkpoint and dataset needed for the tutorial.

  --network_type      Can be one of [mobilenet_v1_ssd, mobilenet_v2_ssd],
                      mobilenet_v1_ssd by default.
  --train_whole_model Whether or not to train all layers of the model. false
                      by default, in which only the last few layers are trained.
  --help              Display this help.
END_OF_USAGE
}

network_type="mobilenet_v2_ssd"
train_whole_model="false"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --network_type)
      network_type=$2
      shift 2 ;;
    --train_whole_model)
      train_whole_model=$2
      shift 2;;
    --help)
      usage
      exit 0 ;;
    --*)
      echo "Unknown flag $1"
      usage
      exit 1 ;;
  esac
done

source "$PWD/constants.sh"

echo "PREPARING checkpoint..."
mkdir -p "${LEARN_DIR}"

ckpt_link="${ckpt_link_map[${network_type}]}"
ckpt_name="${ckpt_name_map[${network_type}]}"
cd "${LEARN_DIR}"
wget -O "${ckpt_name}.tar.gz" "$ckpt_link"
tar zxvf "${ckpt_name}.tar.gz"
mv "${ckpt_name}" "${CKPT_DIR}"

echo "CHOSING config file..."
config_filename="${config_filename_map[${network_type}-${train_whole_model}]}"
cd "${OBJ_DET_DIR}"
cp "configs/${config_filename}" "${CKPT_DIR}/pipeline.config"

echo "REPLACING variables in config file..."
sed -i "s%CKPT_DIR_TO_CONFIGURE%${CKPT_DIR}%g" "${CKPT_DIR}/pipeline.config"
sed -i "s%DATASET_DIR_TO_CONFIGURE%${DATASET_DIR}%g" "${CKPT_DIR}/pipeline.config"

echo "PREPARING dataset"
mkdir -p "${ARCHIVE_DIR}"
mkdir -p "${DATASET_DIR}"

cd "${DATASET_DIR}"

## Prepare data
unzip "${OBJ_DET_DIR}"/dataset.zip
mv dataset/images/train .
mv dataset/images/valid .
mv dataset/images/test .

mv dataset/annotations/instances_train.json train
mv dataset/annotations/instances_valid.json valid
mv dataset/annotations/instances_test.json test

mv "${OBJ_DET_DIR}"/dataset.zip "${ARCHIVE_DIR}"

# mv "${OBJ_DET_DIR}"/labelmap.pbtxt "${DATASET_DIR}"/trash_label_map.pbtxt

mkdir -p "${TFRECORDS_DIR}" && cd "${TFRECORDS_DIR}"
curl -L "https://app.roboflow.ai/ds/FWaxjuc7VD?key=9JHOfJ01C3" > tfrecords.zip
unzip tfrecords.zip

echo "PREPARING label map..."
cd "${DATASET_DIR}"
cp "tfrecords/train/paper-aluminum-cans-plastic_label_map.pbtxt" "${DATASET_DIR}"
mv "paper-aluminum-cans-plastic_label_map.pbtxt" "trash_label_map.pbtxt"

# cd "${TFRECORDS_DIR}"
mv tfrecords/ "${ARCHIVE_DIR}"

echo "CONVERTING dataset to TF Record..."
cd "${OBJ_DET_DIR}"
python object_detection/dataset_tools/create_coco_tf_record.py --logtostderr\
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${DATASET_DIR}"