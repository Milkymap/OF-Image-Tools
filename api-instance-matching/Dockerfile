# base image derivation 
FROM python:3.7-slim-stretch

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

# define new virutal environment 
ENV VIRTUAL_ENV=/opt/venv
RUN chmod -R g+rwx /home/solver && \
    python -m venv $VIRTUAL_ENV --system-site-packages
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# upgrade and install pip 
RUN pip install --upgrade pip && pip install loguru numpy opencv-python pyzmq tqdm  

COPY . ./

# usefull environments 
ENV SOURCE='source/'
ENV TARGET='target/'
ENV SERVER_PORT=8500
ENV THRESHOLD=0.75
ENV WIDTH=128
ENV HEIGHT=128 

# expose port 
EXPOSE $SERVER_PORT

# entrypoint 
CMD ["python", "server.py"]