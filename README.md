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
This misc folder is not present in the git repo and can be downloaded separately from google drive. [Download this zip file](https://drive.google.com/file/d/1OAiKoBXROtcfWWnYmz_GSZdtYG6mBU0s/view?usp=sharing) and extract it on the root directory of your project workspace.

These default paths can be overridden using commandline arguments.
### Training Script
This script takes in the path of the dataset directory and trains a model on it. Finally, the new weights of the model are saved to the given path and file name.
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

### Running Training and Inference via Docker
In order to train your model via docker, Dockerfile.train is provided. You can build and use it for training a model.
Similarly for inference, Dockerfile.infer is also present. It is important to note that these files rely on disk mounting to produce relevant output.

#### Training via Docker
By default, a CMD command at the end of train docker file is present that runs the training code and saves the model to `/app/docker_artifacts` directory.
In order to train your model using docker, the default method is as follows:
1. Create a directory in your host filesystem known as `docker_artifacts`.
2. Make sure that your data is present in `misc/data` in your workspace, the docker file copies this directory into the image.
3. Build the docker image using `docker build -t image_name -f Dockerfile.train .`.
4. Run the docker image after mounting the host's `docker_artifacts` directory: `docker run --name container_name -v ${PWD}/docker_artifacts:/app/docker_artifacts image_name`. `${PWD}` is for powershell, for bash use `pwd`.
5. The last command will download the weights for pretrained resnet18 and train the model. Finally, the newly trained model will appear in `docker_artifacts` directory with the name `resnet18_model.pt`.

_Note: If you want to change the paths and filenames in the above mentioned steps, you will need to override the CMD command in the dockerfile by providing a new one at `docker run ...`_.

#### Inference via Docker
Inference follows a very similar pattern to training, however, it only requires 2 additional files: 1. Image to perform inference on, 2. Pretrained Model weights.
Default Method:
1. Ensure `docker_artifacts` consists of 2 files. 1. Image with the name `image.jpg`. 2. Model weights with the name `resnet18_model.pt`.
2. Build image, similar to point 3 in training with docker.
3. Run the image, again mount the `docker_artifacts` directory and you will see the output of the inference model.
Similarly to training, if you override the CMD command, you'll no longer require to follow the directory structure mentioned above.

#### Docker development file
You can simply ignore this, this was made for the purpose of experimenting with docker, so that I can create training and inference dockerfiles.

## Run Time Analysis and Optimization
For training and inference, we used RTX 3090 GPU.
### Stats for Training
We trained our model for 100 epochs, where we noted the time for each epoch. Over 100 runs we obtained the following values:
1. **Minimum** Time taken for 1 Epoch: _0.79s_
2. **Maximum** Time taken for 1 Epoch: _1.37s_
3. **Average** Time taken per Epoch over 100 Epochs: _0.85s_
Batch size of 32 was used and total number of train images were 360.

### Stats for Inference
For each image in the test set (55 images in total), we noted the time for inference in 2 different settings: (Pytorch's) Inference Mode and Normal mode (where graph for each computation is updated).
Time obtained for both cases was very similar, the `inference_mode` code performed slightly better.
1. **Minimum** Time taken for 1 Image, **Inference Mode**: _0.015s_, **Normal Mode**: _0.016s_.
2. **Maximum** Time taken for 1 Image, **Inference Mode**: _0.076s_, **Normal Mode**: _0.077s_.
3. **Average** Time taken per Image over 55 Images, **Inference Mode**: _0.036s_, **Normal Mode**: _0.039s_.

## Future Improvements
1. This dataset only consists of ~500 images, thus techniques like Metric Learning can be used to improve the model performance.
2. Automatic Hyperparameter tuning can help in achieving even better results.
3. Quantization and Pruning can be applied to the Neural Network for better inference speed on commodity hardware (e.g. CPU).