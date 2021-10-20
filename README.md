## Object detection in an Urban Environment

[//]: # (Image References)

[image0]: ./examples/run1.gif "Expected output"
[image1]: ./examples/hist1.png "Histogram"
[image2]: ./examples/hist2.png "Histogram2"
[image3]: ./examples/vgg16.png "VGG16"
[image4]: ./examples/vgg16_mark.png "VGG16 Marked"
[image5]: ./examples/hist3.png "Histogram3"
[image6]: ./examples/augmentation.png "Augmentation"
[image7]: ./examples/angles.png "3Angles"
[image8]: ./examples/model_1.png "Model"

---

## Overview

In this project, we built supervised deep learning algorithm in Waymo open dataset using tensorflow object detection api. The goal is to make bounding boxs predictions for three objects, cyclists, pedestrians and vehicles in images of urban environments. To deal with that, this project includes data analysis, monitoring model performance, and deployment model for inferencing in video as well. The object detection is one of the most important task to self-driving cars, which lead them to sense surrouding environments and navigate through obstacles. 

---

## Table of Contents

- [Object detection in an Urban Environment](#object-detection-in-an-urban-environment)
- [Overview](#overview)
- [Table of Contents](#table-of-contents)
- [Structure](#structure)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [(Optional)Connect to remote host](#optionalconnect-to-remote-host)
  - [Build](#build)
  - [Install gstuil](#install-gstuil)
- [Development](#development)
  - [Stop remote server(ec2):](#stop-remote-serverec2)
  - [SSH into running docker container:](#ssh-into-running-docker-container)
  - [Open jupyter notebook:](#open-jupyter-notebook)
  - [Make a new config file](#make-a-new-config-file)
  - [Train the model:](#train-the-model)
  - [Evaluate the model:](#evaluate-the-model)
  - [Monitor trainig/evaluation perfomance:](#monitor-trainigevaluation-perfomance)
- [Dataset](#dataset)
  - [Dataset analysis](#dataset-analysis)
  - [Cross validation](#cross-validation)
- [Training](#training)
  - [Reference experiment](#reference-experiment)
- [### Improve on the reference](#-improve-on-the-reference)
- [## Summary](#-summary)
- [Discussion](#discussion)
  - [Problems during my implementation](#problems-during-my-implementation)
  - [Possible improvements](#possible-improvements)
  - [Future Feature](#future-feature)
- [References](#references)
- [Author](#author)
- [Data](#data)
- [Structure](#structure-1)
- [Prerequisites](#prerequisites)
  - [Local Setup](#local-setup)
  - [Classroom Workspace](#classroom-workspace)
- [Instructions](#instructions)
  - [Download and process the data](#download-and-process-the-data)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Create the splits](#create-the-splits)
  - [Edit the config file](#edit-the-config-file)
  - [Training](#training-1)
  - [Improve the performances](#improve-the-performances)
  - [Creating an animation](#creating-an-animation)
    - [Export the trained model](#export-the-trained-model)
- [Submission Template](#submission-template)
  - [Project overview](#project-overview)
  - [Set up](#set-up)
  - [Dataset](#dataset-1)
    - [Dataset analysis](#dataset-analysis-1)
    - [Cross validation](#cross-validation-1)
  - [Training](#training-2)
    - [Reference experiment](#reference-experiment-1)
    - [Improve on the reference](#improve-on-the-reference)

---

## Structure

---

## Getting Started

### Requirements

- [Server with GPU](https://course.fast.ai/start_aws#pricing)
- NVIDIA GPU with the latest driver installed
- [docker / nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) / [tensorflow-docker](https://www.tensorflow.org/install/docker)

The detailed instruction to make remote ec2 server with GPU in [fast ai site](https://course.fast.ai/start_aws)

The environment example by running `nvidia-smi`
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

### (Optional)Connect to remote host

```sh
ssh -L 8000:localhost:8000 -L 6006:localhost:6006 ubuntu@{your-IP}
```
It opens two port, 8000 for remote host and docker, 6006 for tenosorboard.

### Build

After you obtain this repo, build the image with:
```sh
# Once in build folder
docker build -t project-dev -f Dockerfile .
```
Create a container with:
```sh
docker run --shm-size=8gb -p 8000:8000 -p 6006:6006 —gpu all -v /home/ubuntu/nd013-c1-vision-starter:/app/project/ -ti project-dev bash
```
`--shm-size` value 8gb can be changed according to your server memory. If you have error when using `—gpu`, you might have wrong docker. You can seach official installation way in this [page](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)

### Install gstuil

Once in container, you need to get gustil for accessing cloud storage:
```sh
curl https://sdk.cloud.google.com | bash
```
Auth gcloud:
```sh
gcloud auth login
```

The official gustil setup can be found [here](https://cloud.google.com/storage/docs/gsutil)

---
## Development

### Stop remote server(ec2):
```sh
# Outside container
sudo shutdown -h now
```

### SSH into running docker container:
```sh
# Outside container
docker ps
docker exec -i -t {CONTAINER_ID} /bin/bash
```
### Open jupyter notebook:
```sh
# In container
cd project
jupyter notebook --port 8000 --ip=0.0.0.0 --allow-root
```
### Make a new config file
```sh
# In container
cd project
python edit_config.py --train_dir /app/project/data/waymo/train/ --eval_dir /app/project/data/waymo/val/ --batch_size 4 --checkpoint ./training/pretrained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map label_map.pbtxt
```
### Train the model:
```sh
# In container
cd project
python experiments/model_main_tf2.py --model_dir=training/reference/ --pipeline_config_path=training/reference/pipeline_new.config
```
### Evaluate the model:
```sh
# In container
cd project
CUDA_VISIBLE_DEVICES="" python experiments/model_main_tf2.py --model_dir=training/reference/ --pipeline_config_path=training/reference/pipeline_new.config --checkpoint_dir=training/reference/
```
### Monitor trainig/evaluation perfomance:
```sh
# In container, connecting to localhost:6006
tensorboard --logdir=training/reference/ --host=0.0.0.0
```
If you'll see GPU usage to make sure that gpu is used when training:
```sh
# Outside container
nvidia-smi
# Or
nvtop
```

---

## Dataset
### Dataset analysis
### Cross validation

---

## Training
### Reference experiment
### Improve on the reference
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

---

## Author

- [Tsuyoshi Akiyama](https://github.com/Akitsuyoshi)


## Data

For this project, we will be using data from the [Waymo Open dataset](https://waymo.com/open/). The files can be downloaded directly from the website as tar files or from the [Google Cloud Bucket](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_2_0_individual_files/) as individual tf records. 

## Structure

The data in the classroom workspace will be organized as follows:
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

## Prerequisites

### Local Setup

For local setup if you have your own Nvidia GPU, you can use the provided Dockerfile and requirements in the [build directory](./build).

Follow [the README therein](./build/README.md) to create a docker container and install all prerequisites.

### Classroom Workspace

In the classroom workspace, every library and package should already be installed in your environment. You will not need to make use of `gcloud` to download the images.

## Instructions

### Download and process the data

**Note:** This first step is already done for you in the classroom workspace. You can find the downloaded and processed files within the `/data/waymo/` directory (note that this is different than the `/home/workspace/data` you'll use for splitting )

The first goal of this project is to download the data from the Waymo's Google Cloud bucket to your local machine. For this project, we only need a subset of the data provided (for example, we do not need to use the Lidar data). Therefore, we are going to download and trim immediately each file. In `download_process.py`, you can view the `create_tf_example` function, which will perform this processing. This function takes the components of a Waymo Tf record and saves them in the Tf Object Detection api format. An example of such function is described [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#create-tensorflow-records). We are already providing the `label_map.pbtxt` file.

You can run the script using the following (you will need to add your desired directory names):
```
python download_process.py --data_dir {processed_file_location} --temp_dir {temp_dir_for_raw_files}
```

You are downloading 100 files so be patient! Once the script is done, you can look inside your data_dir folder to see if the files have been downloaded and processed correctly.


### Exploratory Data Analysis

Now that you have downloaded and processed the data, you should explore the dataset! This is the most important task of any machine learning project. To do so, open the `Exploratory Data Analysis` notebook. In this notebook, your first task will be to implement a `display_instances` function to display images and annotations using `matplotlib`. This should be very similar to the function you created during the course. Once you are done, feel free to spend more time exploring the data and report your findings. Report anything relevant about the dataset in the writeup.

Keep in mind that you should refer to this analysis to create the different spits (training, testing and validation). 


### Create the splits

Now you have become one with the data! Congratulations! How will you use this knowledge to create the different splits: training, validation and testing. There are no single answer to this question but you will need to justify your choice in your submission. You will need to implement the `split_data` function in the `create_splits.py` file. Once you have implemented this function, run it using:
```
python create_splits.py --data_dir /home/workspace/data/
```

NOTE: Keep in mind that your storage is limited. The files should be <ins>moved</ins> and not copied. 

### Edit the config file

Now you are ready for training. As we explain during the course, the Tf Object Detection API relies on **config files**. The config that we will use for this project is `pipeline.config`, which is the config for a SSD Resnet 50 640x640 model. You can learn more about the Single Shot Detector [here](https://arxiv.org/pdf/1512.02325.pdf). 

First, let's download the [pretrained model](http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.tar.gz) and move it to `training/pretrained-models/`. 

Now we need to edit the config files to change the location of the training and validation files, as well as the location of the label_map file, pretrained weights. We also need to adjust the batch size. To do so, run the following:
```
python edit_config.py --train_dir /home/workspace/data/train/ --eval_dir /home/workspace/data/val/ --batch_size 4 --checkpoint ./training/pretrained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ckpt-0 --label_map label_map.pbtxt
```
A new config file has been created, `pipeline_new.config`.

### Training

You will now launch your very first experiment with the Tensorflow object detection API. Create a folder `training/reference`. Move the `pipeline_new.config` to this folder. You will now have to launch two processes: 
* a training process:
```
python model_main_tf2.py --model_dir=training/reference/ --pipeline_config_path=training/reference/pipeline_new.config
```
* an evaluation process:
```
python model_main_tf2.py --model_dir=training/reference/ --pipeline_config_path=training/reference/pipeline_new.config --checkpoint_dir=training/reference/
```

NOTE: both processes will display some Tensorflow warnings.

To monitor the training, you can launch a tensorboard instance by running `tensorboard --logdir=training`. You will report your findings in the writeup. 

### Improve the performances

Most likely, this initial experiment did not yield optimal results. However, you can make multiple changes to the config file to improve this model. One obvious change consists in improving the data augmentation strategy. The [`preprocessor.proto`](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto) file contains the different data augmentation method available in the Tf Object Detection API. To help you visualize these augmentations, we are providing a notebook: `Explore augmentations.ipynb`. Using this notebook, try different data augmentation combinations and select the one you think is optimal for our dataset. Justify your choices in the writeup. 

Keep in mind that the following are also available:
* experiment with the optimizer: type of optimizer, learning rate, scheduler etc
* experiment with the architecture. The Tf Object Detection API [model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) offers many architectures. Keep in mind that the `pipeline.config` file is unique for each architecture and you will have to edit it. 


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

## Submission Template

### Project overview
This section should contain a brief description of the project and what we are trying to achieve. Why is object detection such an important component of self driving car systems?

### Set up
This section should contain a brief description of the steps to follow to run the code for this repository.

### Dataset
#### Dataset analysis
This section should contain a quantitative and qualitative description of the dataset. It should include images, charts and other visualizations.
#### Cross validation
This section should detail the cross validation strategy and justify your approach.

### Training 
#### Reference experiment
This section should detail the results of the reference experiment. It should includes training metrics and a detailed explanation of the algorithm's performances.

#### Improve on the reference
This section should highlight the different strategies you adopted to improve your model. It should contain relevant figures and details of your findings.
 
