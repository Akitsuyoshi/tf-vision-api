## Object detection in an Urban Environment

[//]: # (Image References)

[image0]: ./images/image0.png "Splash image"
[image1]: ./images/hist1.png "Histogram"
[image2]: ./images/hist2.png "Histogram2"
[image3]: ./images/vgg16.png "VGG16"
[image4]: ./images/vgg16_mark.png "VGG16 Marked"
[image5]: ./images/hist3.png "Histogram3"
[image6]: ./images/augmentation.png "Augmentation"
[image7]: ./images/angles.png "3Angles"
[image8]: ./images/model_1.png "Model"

---
 ![Splash image][image0]

## Overview

In this project, we built supervised deep learning algorithm in Waymo open dataset using tensorflow object detection api. The goal is to make bounding boxs predictions for three objects, cyclists, pedestrians and vehicles in images of urban environments. To deal with that, this project includes data analysis, monitoring model performance, and deployment model for inferencing in video. The object detection is one of the most important task to self-driving cars, which lead them to sense surrouding environments and navigate through obstacles.

---

## Table of Contents

- [Object detection in an Urban Environment](#object-detection-in-an-urban-environment)
- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Structure](#structure)
- [Getting Started](#getting-started)
  - [Dataset](#dataset)
  - [Requirements](#requirements)
  - [Steps to run the code](#steps-to-run-the-code)
- [Development](#development)
- [Dataset](#dataset-1)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Cross validation](#cross-validation)
- [Training](#training)
  - [Improve the performances](#improve-the-performances)
  - [Reference experiment](#reference-experiment)
  - [Creating an animation](#creating-an-animation)
    - [Export the trained model](#export-the-trained-model)
  - [Improve on the reference](#improve-on-the-reference)
- [Summary](#summary)
- [Discussion](#discussion)
  - [Problems during my implementation](#problems-during-my-implementation)
  - [Possible improvements](#possible-improvements)
  - [Future Feature](#future-feature)
- [References](#references)
- [Author](#author)

---

## Structure

The data in this repo will be organized as follows:
```
/data/waymo/
    - contains the tf records in the Tf Object detection api format.

/home/workspace/data/
    - test: contain the test data (empty to start)
    - train: contain the train data (empty to start)
    - val: contain the val data (empty to start)
```

The experiments folder will be organized as follow:
```
experiments/
    - exporter_main_v2.py: to create an inference model
    - model_main_tf2.py: to launch training
    - experiment0/....
    - experiment1/....
    - experiment2/...
    - pretrained-models/: contains the checkpoints of the pretrained models.
```

---

## Getting Started

### Dataset

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/). The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. 

### Requirements

- [Server with GPU](https://course.fast.ai/start_aws#pricing)
- NVIDIA GPU with the latest driver installed
- [docker / nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) / [tensorflow-docker](https://www.tensorflow.org/install/docker)

The detailed instruction to make remote ec2 server with GPU in [fast ai site](https://course.fast.ai/start_aws)

The environment example by running `nvidia-smi` looks like:
```sh
| NVIDIA-SMI 460.91.03    Driver Version: 460.91.03    CUDA Version: 11.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla T4            Off  | 00000000:00:1E.0 Off |                    0 |
| N/A   31C    P0    25W /  70W |      0MiB / 15109MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

**Note: This project works only on specific version, GPU Driver Version: 460.91.03, and CUDA Version: 11.2**


### Steps to run the code

```sh
# 0. (Optional)SSH to remote host
# It opens two port, 8000 for remote host and docker, 6006 for tenosorboard.
ssh -L 8000:localhost:8000 -L 6006:localhost:6006 ubuntu@{your-IP}

# 1. Obtain the repo
git clone https://github.com/Akitsuyoshi/tf-vision-api.git

# 2. Move into the repo folder
cd tf-vision-api

# 3. Get the processed tf record data
python download_process.py --data_dir ./data/waymo

# 4. Split data into train/val/test set
python create_splits.py --data_dir ./data/waymo

# 5. Build docker image
docker build -t project-dev -f build/Dockerfile .

# 6. Create a container with above image
# you can change --shm-size according to your server memory
docker run —-gpus all --shm-size=8gb -p 8000:8000 -p 6006:6006 -v /home/ubuntu/nd013-c1-vision-starter:/app/project/ -ti project-dev bash


# 7. Install gstuil
curl https://sdk.cloud.google.com | bash

# 8. Auth gcloud
gcloud auth login
```

---
## Development

Outside container, you can stop remote, ssh to running container, and check GPU usage.

```sh
# Stop remote ec2 server
sudo shutdown -h now

# SSH to running container
docker ps
docker exec -i -t {CONTAINER_ID} /bin/bash

# Check GPU usage
nvidia-smi
# Or
nvtop
```

Inside container, you can open jupyter notebook, make a new config for tf object detection api, train/evaluate model, and monitor them.
```sh
# Run a container
docker run —-gpus all --shm-size=8gb -p 8000:8000 -p 6006:6006 -v /home/ubuntu/nd013-c1-vision-starter:/app/project/ -ti project-dev bash

# Once in container
cd project

# Open jupyter notebook
jupyter notebook --port 8000 --ip=0.0.0.0 --allow-root

# Make a `pipeline_new.config` new config file
python edit_config.py --train_dir /app/project/data/waymo/train/ --eval_dir /app/project/data/waymo/val/ --batch_size 4 --checkpoint ./training/pretrained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map label_map.pbtxt

# Train the model
python experiments/model_main_tf2.py --model_dir=training/reference/ --pipeline_config_path=training/reference/pipeline_new.config

# Evaluate the model
CUDA_VISIBLE_DEVICES="" python experiments/model_main_tf2.py --model_dir=training/reference/ --pipeline_config_path=training/reference/pipeline_new.config --checkpoint_dir=training/reference/

# Monitor trainig/evaluation perfomance
tensorboard --logdir=training/reference/ --host=0.0.0.0
```

---

## Dataset

### Exploratory Data Analysis

This section should contain a quantitative and qualitative description of the dataset. It should include images, charts and other visualizations.

### Cross validation

This section should detail the cross validation strategy and justify your approach.

---

## Training

### Improve the performances

Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup. 

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it. 

### Reference experiment

This section should detail the results of the reference experiment. It should includes training metrics and a detailed explanation of the algorithm's performances.

### Creating an animation
#### Export the trained model
Modify the arguments of the following function to adjust it to your models:
```
python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path training/experiment0/pipeline.config --trained_checkpoint_dir training/experiment0/ckpt-50 --output_directory training/experiment0/exported_model/
```

Finally, you can create a video of your model's inferences for any tf record file. To do so, run the following command (modify it to your files):
```
python inference_video.py -labelmap_path label_map.pbtxt --model_path training/experiment0/exported_model/saved_model --tf_record_path /home/workspace/data/test/tf.record --config_path training/experiment0/pipeline_new.config --output_path animation.mp4
```

### Improve on the reference
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.

---
## Summary

---

## Discussion

### Problems during my implementation

### Possible improvements

### Future Feature

---

## References

- [Tensorflow2 Object Detction API tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#training-custom-object-detector)
- [Tensorflow Dataset tutorial in CS230](https://cs230.stanford.edu/blog/datapipeline/)

---

## Author

- [Tsuyoshi Akiyama](https://github.com/Akitsuyoshi)
