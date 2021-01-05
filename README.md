## Getting started

#### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate tracker-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate tracker-gpu
```

#### Pip
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```

### Nvidia Driver (For GPU, if you haven't set it up already)
```bash
# Ubuntu 18.04
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt install nvidia-driver-430
# Windows/Other
https://www.nvidia.com/Download/index.aspx
```

Part 1: Prerequisites

RabbitMQ:

1) Installation https://registry.hub.docker.com/_/rabbitmq/
docker pull rabbitmq
2) Start RabbitMQ service
docker run -it --rm --name rabbitmq -p 5672:5672 -p 15672:15672 rabbitmq:3-management

YoloV3 Pytorch version:

1) Installation: https://github.com/ultralytics/yolov3/wiki/Docker-Quickstart
sudo docker pull ultralytics/yolov3:latest

2) Start PyTorch YoloV3 docker
(Below is just a template, modify it accordingly on "/dockers:/usr/src/host")
sudo docker run --ipc=host --net=host --gpus all -it -v "$(pwd)"/dockers:/usr/src/host ultralytics/yolov3:latest

Part 2: 
1) Copy pyTorch_YoloV3.py from the repository to the PyTorch Yolo docker environment, rename it as detect.py (replace the original detect.py from the docker image)
2) Copy sample video from https://drive.google.com/file/d/1OApoN82WoEAjQyImBpCCS4EVj03mXLSB/view?usp=sharing to data/video/h1.mp4 and also to the
   PyTorch YoloV3 docker folder where Yolo V3 will also read the video file.  Note: the current implementation requires video read by detection and track at the same time.
3) May need some small parameter modification for example, make sure file locations are right

Part 3:
1) on PyTorch Yolo V3 docker, run:
python detect.py --source /usr/src/host/data/videos/h1.mp4 --project /usr/src/host/data
YoloV3 will start object detection and feed RabbitMQ with BBox

2) On local machine, run
conda activate tracker-gpu  (if has GPU, else cpu)
python object_speed_track.py
Object tracker will start process BBox from RabbitMQ and do tracking

Note:
May need to modify small simple things here and there to get it work.
If run with CPU, will have performance impacts.
