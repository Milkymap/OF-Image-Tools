# base image derivation 
FROM nvcr.io/nvidia/pytorch:21.08-py3

# initial argument and env 
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Paris

# setup required config 
RUN apt-get update --fix-missing && \
    apt-get install --yes --no-install-recommends \
         tzdata dialog apt-utils \ 
         gcc pkg-config git curl build-essential \
         ffmpeg libsm6 libxext6 libpcre3 libpcre3-dev

# define new user 
RUN useradd --gid root --create-home solver 
WORKDIR /home/solver

COPY . ./

# define new virutal environment 
ENV VIRTUAL_ENV=/opt/venv
RUN chmod -R g+rwx /home/solver && \
    python -m venv $VIRTUAL_ENV --system-site-packages
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# upgrade and install pip 
RUN pip install --upgrade pip && pip install loguru pyzmq opencv-python tqdm  

# usefull environments 
ENV MODELS_PATH='models/'
ENV IMAGES_PATH='images/'
ENV SERVER_PORT=8500
ENV MODELS_NAME='res18.th'

# expose port 
EXPOSE $SERVER_PORT

# entrypoint 
CMD ["python", "server.py"]