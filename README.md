# Visual Question Answering

To download and organize data for training and validation -
```
chmod +x download_data.sh
./download_data.sh
```
Genereate image features for train and validation data usnig pre-trained ResNet model -
```
python generate_image_feature.py --imgRoot '../data/val/val2014/' --outRoot '../data/val/img_features/'
python generate_image_feature.py --imgRoot '../data/train/train2014/' --outRoot '../data/train/img_features/'
```
Generate vocabulary for using the following - 
```
python preprocess_text.py
```
We only consider the top 3000 frequent answers with single answers
```
python filter_data.py --split train
python filter_data.py --split val
```
To train network
```
python train.py --experiment <name_of_experiment> --model <SAN/Tiled/Baseline>

--experiment - name of experiment
--epocs - number of epochs
--learning rate - learning rate
--batchSize - batch size
--checkpoint_dir - path to saved checkpoint
--optim - adam/sgd
--model - SAN/Tiled/Baseline
```


# TODO
Does the vocabulary have only single word answers because we are filtering it out that way \
What percetage of answers would be covered if we take 1000 most frequent answers because SAN and baseline work on 1000? \
To add eval related code 



