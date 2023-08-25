FROM nvidia/cuda:11.6.2-base-ubuntu20.04
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Pacific/Auckland
RUN apt-get update
RUN apt-get -y install git python3 python3-pip
RUN apt-get update && apt-get -y install vim
RUN apt-get update && apt-get -y install nano

RUN apt-get -y install ffmpeg libsm6 libxext6

# install cares_lib
COPY ./cares_lib /cares_lib
RUN pip install -r /cares_lib/requirements.txt
RUN pip install /cares_lib

# install cares_reinforcement_learning
COPY ./cares_reinforcement_learning /cares_reinforcement_learning
RUN pip install -r /cares_reinforcement_learning/requirements.txt
RUN pip install --editable /cares_reinforcement_learning

# copy Gripper-Code
COPY ./Gripper-Code /Gripper-Code
RUN pip install -r /Gripper-Code/requirements.txt

# RUN pip install swig==4.1.1
# RUN apt-get -y install patchelf libosmesa6-dev libegl1-mesa libgl1-mesa-glx libglfw3 libglew-dev
# RUN pip install pyglet==1.5.27
# RUN apt-get -y install libglib2.0-0
# ENTRYPOINT python3 -u /dreamerv3-torch/dreamer.py --configs gymnasium --task gymnasium_ClassicControl_MountainCar-v0 --logdir /dreamerv3-torch-logdir/mountaincar
