# Check Box Classifier
This is an opensource project for classifying a given image into following three classes:
1. Checked (The checkbox is checked.)
2. Unchecked (The checkbox is unchecked.)
3. Other (The image does not contain a checkbox.)

## High level approach
Given that the dataset is quite small, only ~500 images, we decided to go with a pretrained model.
There are many models to choose from, however Resnet 18 is a model that is somewhat lightweight while having a good performance.
We initialize it with the pretrained weights on ImageNet and replace the fully connected layer at the end with the shape: (512, 3), given there are total 3 classes for our problem.
We also freeze all the layers except for the ones present in the last block of the resnet 18 network.
Optimizer used is SGD with learning rate of 0.01 and momentum of 0.2. Loss used Cross Entropy as this is a multiclass classification problem.

## Hyperparameter Tuning
Manual Hyperparameter Tuning was carried out, a few instances are as follows:
1. **Deciding which layers to freeze**: Training all the layers in resnet 18 caused the model to be overparameterized for our problem and training only the last layer caused the model to be biased. Thus, we froze the entire network and only trained the last few layers.
2. **Deciding momentum**: SGD is known to generalize better than Adam, but it's hard to tune. Thus, we tried high, low and medium values for it and finally found that 0.2 was optimum.
3. **Deciding Input Image Size**: We tried 120x120 first and then 224x224. Later produced the best results.

## Evaluation
We splitted the data into three sets, 70% for train, ~19.35% for validation and ~10.64% for testing. We observed different metrics for classification.
E.g. Precision, Recall, Accuracy, F1-score etc.

## Results
### Validation Set
|           | precision | recall | f1-score |
|-----------|-----------|--------|----------|
| checked   | 0.88     | 0.78   | 0.83     |
| other     | 0.82      | 0.86   | 0.84     |
| unchecked | 0.79      | 0.85   | 0.82     |

**Validation Accuracy**: _0.83_, **Macro Average**: _0.83_

## Test Set
|           | precision | recall | f1-score |
|-----------|-----------|--------|----------|
| checked   | 0.94     | 0.89   | 0.92     |
| other     | 0.81      | 1.00   | 0.90     |
| unchecked | 1.00      | 0.91   | 0.95     |

**Test Accuracy**: _0.93_, **Macro Average**: _0.92_

_Note: The code was run multiple times and each time the result produce was different, however all the results had similar metric values._

## Running The Code
This code mainly consists of 2 files one for training and other for inference. By default the scripts assume the following directory structure:
```shell
- root/
  - misc/
    - data/
      - checked/
        - file1.jpg
        - file2.jpg
        - ...
      - unchecked/
        - file1.jpg
        - file2.jpg
        - ...
      - other/
        - file1.jpg
        - file2.jpg
        - ...
    - model_weights/
      - pretrained_model.pt
```
This misc folder is not present in the git repo and can be downloaded separately from google drive. [Download this zip file](https://drive.google.com/file/d/1OAiKoBXROtcfWWnYmz_GSZdtYG6mBU0s/view?usp=sharing) and extract it on the root directory.

These default paths can be overridden using commandline arguments.
### Training Script
This script takes in the path of the dataset directory and trains a model on them. Finally the new weights of the model and saved to the given path and file name.
```shell
usage: train.py [-h] [-f MODEL_PATH] [-d DATA_PATH]

Train a model to distinguish between different check boxes.

options:
  -h, --help     show this help message and exit
  -f MODEL_PATH
  -d DATA_PATH
  
default:
  -f resnet_18_weights.pt
  -d misc/data/
  
example: python train.py -f new_resnet_18_weights.pt -d data/
```

## Inference Script
This script takes in a path to the input image and prints the label on the screen.
By default it assumes that the model is placed in `misc/model_weights/checkbox_resnet18_weights.pt`, but one can always override this to a new path.

```shell
usage: inference.py [-h] -f FILE_PATH [--model-path MODEL_PATH]

Script to classify checkboxes.

options:
  -h, --help            show this help message and exit
  -f FILE_PATH
  --model-path MODEL_PATH

default:
  -d misc/model_weights/checkbox_resnet18_weights.pt
  
required:
  -f {Path to input Image}
```