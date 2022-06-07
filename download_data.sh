#!/bin/bash
# Script to download and organize data.

# Training
mkdir -p data
cd data

mkdir -p train
cd train

# Download data
wget "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Train_mscoco.zip"
wget "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Train_mscoco.zip"
wget "http://images.cocodataset.org/zips/train2014.zip"

# Annotations
unzip Annotations_Train_mscoco.zip
mv mscoco_train2014_annotations.json annotations.json
rm -rf Annotations_Train_mscoco.zip

# Questions
unzip Questions_Train_mscoco.zip
rm -rf Questions_Train_mscoco.zip
rm -rf MultipleChoice_mscoco_train2014_questions.json
mv OpenEnded_mscoco_train2014_questions.json questions.json

# Images
unzip train2014.zip
rm -rf train2014.zip

cd ..

# Validation
mkdir -p val
cd val

# Download data
wget "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Annotations_Val_mscoco.zip"
wget "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/Questions_Val_mscoco.zip"
wget "http://images.cocodataset.org/zips/val2014.zip"

# Annotations
unzip Annotations_Val_mscoco.zip
rm -rf Annotations_Val_mscoco.zip
mv mscoco_val2014_annotations.json annotations.json

# Questions
unzip Questions_Val_mscoco.zip
rm -rf Questions_Val_mscoco.zip
rm -rf MultipleChoice_mscoco_val2014_questions.json
mv OpenEnded_mscoco_val2014_questions.json questions.json

# Images
unzip val2014.zip
rm -rf val2014.zip