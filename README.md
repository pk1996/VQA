# Visual Question Answering

To download and organize data for training and validation -
```
chmod +x download_data.sh
./download_data.sh
```
Genereate image features for train and validation data usnig pre-trained ResNet model -
```
python generate_image_features --imgRoot '../data/val/val2014/' --outRoot '../data/val/img_features/'
python generate_image_features --imgRoot '../data/train/train2014/' --outRoot '../data/train/img_features/'
```
Generate vocabulary for using the following - 
```
python preprocess_text.py
```
We only consider the top 3000 frequent answers with single answers \
#TODO Does the vocabulary have only single word answers?
```
python filter_data.py --split train
python filter_data.py --split val
```
