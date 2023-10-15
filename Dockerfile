# Specify the parent image from which we build
# The "environment base" comes from Dockerfile_env
# (pushed to Docker Hub automatically with Github Actions):
FROM petteriteikari/minivess-mlops-env:latest as base

# https://stackoverflow.com/a/63643361/18650369
ENV USER minivessuser
ENV HOME /home/$USER

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

RUN mkdir /mnt/minivess_mlops_artifacts  \
    /mnt/minivess_mlops_artifacts/data  \
    /mnt/minivess_mlops_artifacts/output
RUN chown $USER:$USER /mnt/minivess_mlops_artifacts  \
    /mnt/minivess_mlops_artifacts/data  \
    /mnt/minivess_mlops_artifacts/output
VOLUME ["/mnt/minivess_mlops_artifacts/data", "/mnt/minivess_mlops_artifacts/output"]

# Switch to non-privileged user from superuser
USER $USER

COPY --chown=$USER:$USER ml_tests ./ml_tests
COPY --chown=$USER:$USER src ./src
WORKDIR /app/src