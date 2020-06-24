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
                "banana peel" "bubble wrap" "cardboard box" "chips packet" "small scrap"
                "clear plastic cup full" "clear plastic lid" "clear plastic straw lid" 
                "coffee cup" "coffee cup lid" "condiment packet" "food product packaging" 
                "food scrap" "food wrapper" "glass bottle" "gloves" "orange peel"
                "paper bag" "paper bag full" "paper box" "paper card" "tetrapak carton"
                "paper crumpled" "paper cup" "paper food box" "paper newspaper"
                "paper food togo container" "paper napkin" "paper scrap" "paper sheet" 
                "paper plate" "paper pulp drink tray" "paper shredded" "plastic 6 cup" 
                "plastic bag" "plastic bottle" "plastic bottle full" "styrofoam plate"
                "plastic bottle small" "plastic clear clamshell" "plastic clear lid"
                "plastic clear cup empty" "plastic clear cup full" "plastic plate"
                "plastic food container" "plastic food container full" "plastic straw"
                "plastic utensil" "product packaging" "shredded paper" "styrofoam cup"
                "styrofoam food container" 
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
mkdir -p ${output}/ann
mkdir -p ${output}/dataset/images
mkdir -p ${output}/dataset/annotations

cp ${data_path}/meta.json ${output}/

# Copy all images and annotation files to a flat folder structure
printf "Flattening dataset for conversion to MS-COCO\n"
for i in "${arr[@]}"
do
    cp ${data_path}/"$i"/img/*.jpg ${output}/
    cp ${data_path}/"$i"/ann/*.json ${output}/ann
done

# Create json format annotation file for all images
printf "Converting from supervise.ly format to COCO\n"
python data/supervisely2coco.py --meta ${output}/meta.json --annotations ${output}/ann --output ${output}/coco.json --image_name

# Train/Test/Val split
printf "Train/Test/Val split on the COCO dataset...\n"
python data/create_coco.py --data ${output} --cocofile coco.json --output ${output}/dataset

printf "Moving files into place...\n"
mv ${output}/train/ ${output}/dataset/images
mv ${output}/valid/ ${output}/dataset/images
mv ${output}/test/ ${output}/dataset/images

mv ${output}/dataset/instances_train.json ${output}/dataset/annotations
mv ${output}/dataset/instances_valid.json ${output}/dataset/annotations
mv ${output}/dataset/instances_test.json ${output}/dataset/annotations

printf "All done. The COCO formatted dataset can be found at ${output}\n"