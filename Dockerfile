FROM pytorch/pytorch:1.2-cuda10.0-cudnn7-runtime

# Not all the dependencies are needed - but this is nice to
# try out different games really quick

RUN apt-get update &&  \
    apt-get install -y \
      python3-tk sox libsox-dev libsox-fmt-all \
      libgl1-mesa-dri libgl1-mesa-glx freeglut3-dev \
      swig xvfb ffmpeg

ADD requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

RUN mkdir -p /workspace
WORKDIR /workspace

# Only needed to import ROMs
# RUN python -m retro.import /workspace
