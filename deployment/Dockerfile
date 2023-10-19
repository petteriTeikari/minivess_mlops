# syntax = docker/dockerfile:1.3
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
ENV PYTHONPATH "${PYTHONPATH}:/app"

RUN mkdir /mnt/minivess-artifacts /mnt/minivess-artifacts_local /mnt/minivess-dvc-cache
RUN chown $USER:$USER /mnt/minivess-artifacts /mnt/minivess-artifacts_local /mnt/minivess-dvc-cache
# VOLUME ["/mnt/minivess-dvc-cache", "/mnt/minivess-artifacts"]

# Switch to non-privileged user from superuser
USER $USER

# https://dzone.com/articles/clone-code-into-containers-how
# TODO! add switch here to do either copy (local devel) or clone (remote)
# RUN git clone https://github.com/petteriTeikari/minivess_mlops.git .
COPY --chown=$USER:$USER . .

ENV PORT 8088
EXPOSE $PORT

USER root
# https://github.com/ktruckenmiller/aws-mountpoint-s3/blob/main/Dockerfile
RUN echo "user_allow_other" >> /etc/fuse.conf
COPY ./deployment/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod 755 /usr/local/bin/entrypoint.sh

# Switch to non-privileged user from superuser
USER $USER
ENTRYPOINT [ "entrypoint.sh"]

WORKDIR /app/src