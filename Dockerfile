# syntax = docker/dockerfile:1.3
# See for getting secrets into builds:
#     https://pythonspeed.com/articles/docker-build-secrets/
#     https://refine.dev/blog/docker-build-args-and-env-vars/#conclusion
# In case you would like to use AWS secrets here already for the "dvc pull"
# https://medium.com/analytics-vidhya/docker-volumes-with-dvc-for-versioning-data-and-models-for-ml-projects-4885935db3ec

# Specify the parent image from which we build
# The "environment base" comes from Dockerfile_env
# (pushed to Docker Hub automatically with Github Actions):
FROM petteriteikari/minivess-mlops-env:latest as base

# https://stackoverflow.com/a/63643361/18650369
ENV USER minivessuser
ENV HOME /home/$USER

# https://medium.com/analytics-vidhya/docker-volumes-with-dvc-for-versioning-data-and-models-for-ml-projects-4885935db3ec


USER root
RUN useradd -m $USER && echo $USER:$USER | chpasswd && adduser $USER sudo
RUN chown $USER:$USER $HOME

RUN export DEBIAN_FRONTEND=noninteractive && \
    ln -sf /usr/bin/python3.8 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.8 /usr/bin/python

RUN mkdir /app
RUN chown $USER:$USER /app
WORKDIR /app
# for src. and ml_tests. module imports
ENV PYTHONPATH "${PYTHONPATH}:/app"

RUN mkdir /mnt/minivess-artifacts  \
    /mnt/minivess-dvc-cache
RUN chown $USER:$USER /mnt/minivess-artifacts  \
    /mnt/minivess-dvc-cache
VOLUME ["/mnt/minivess-dvc-cache", "/mnt/minivess-artifacts"]

# Switch to non-privileged user from superuser
USER $USER
# https://dzone.com/articles/clone-code-into-containers-how
RUN git clone https://github.com/petteriTeikari/minivess_mlops.git .

WORKDIR /app/src