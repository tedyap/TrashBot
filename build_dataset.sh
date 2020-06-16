#!/bin/bash

# Exit on error
set -e

# Echo each command, for debugging
# set -x

usage() {
  cat << END_OF_USAGE

  Setup data for converting from Supervisely to COCO format

  --data_path  Path to extracted superviely data
  --output     Output data path
  --help            Display this help.
END_OF_USAGE
}

# Category names
# TODO: Find a better way of storing this
declare -a arr=("aluminum cans" "aluminum foil crumpled" "aluminum food container" 
                "banana peel" "bubble wrap" "cardboard box" "chips packet"
                "clear plastic cup full" "clear plastic lid" "clear plastic straw lid" 
                "coffee cup" "coffee cup lid" "condiment packet" "food product packaging" 
                "food scrap" "food wrapper" "glass bottle" "gloves" "mixed items"
                "orange peel" "paper bag" "paper bag full" "paper box" "paper card" 
                 "paper crumpled" "paper cup" "paper food box" "paper newspaper"
                "paper food togo container" "paper napkin" "paper scrap" "paper sheet" 
                "paper plate" "paper pulp drink tray" "paper shredded" "plastic 6 cup" 
                 "plastic bag" "plastic bottle" "plastic bottle full"
                "plastic bottle small" "plastic clear clamshell" "plastic clear lid"
                "plastic clear cup empty" "plastic clear cup full" "plastic plate"
                "plastic food container" "plastic food container full" "plastic straw"
                "plastic utensil" "product packaging" "product wrapper" "shredded paper"
                "small scrap" "styrofoam cup" "styrofoam food container" "tea bag"
                "styrofoam plate" "tetrapak carton"
                )

data_path="/home/saurabh/Desktop/TrashBot"
output="/home/saurabh/Desktop/coco"

# Handle command line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_path)
      data_path=$2
      shift 2 ;;
    --output)
      output=$2
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

printf "Setting up directory structure\n"
mkdir -p ${output}/annotations
mkdir -p ${output}/dataset

cp ${data_path}/meta.json ${output}/

# Copy all images and annotation files to a flat folder structure
printf "Flattening dataset for conversion to MS-COCO\n"
for i in "${arr[@]}"
do
    cp ${data_path}/"$i"/img/*.jpg ${output}/
    cp ${data_path}/"$i"/ann/*.json ${output}/annotations
done

# Create json format annotation file for all images
printf "Converting from supervise.ly format to COCO\n"
python data/supervisely2coco.py --meta ${output}/meta.json --annotations ${output}/annotations --output ${output}/coco.json --image_name

# Train/Test/Val split
printf "Train/Test/Val split on the COCO dataset...\n"
python data/create_coco.py --data ${output} --cocofile coco.json --output ${output}/dataset

printf "Moving files into place...\n"
mv ${output}/train/ ${output}/dataset
mv ${output}/valid/ ${output}/dataset
mv ${output}/test/ ${output}/dataset

mv ${output}/dataset/instances_train.json ${output}/dataset/train
mv ${output}/dataset/instances_valid.json ${output}/dataset/valid
mv ${output}/dataset/instances_test.json ${output}/dataset/test

printf "All done. The COCO formatted dataset can be found at ${output}\n"