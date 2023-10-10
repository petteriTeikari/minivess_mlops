# Specify the parent image from which we build
FROM ubuntu:22.04 as base

# Poetry approach from:
# https://stackoverflow.com/a/54763270/18650369
# https://github.com/wemake-services/wemake-django-template/blob/master/%7B%7Bcookiecutter.project_name%7D%7D/docker/django/Dockerfile
# https://stackoverflow.com/a/72465422/6412152
# Multiple lines ok: https://stackoverflow.com/a/49516138/18650369

# Input arguments
ARG DEPLOY_IN=local
ENV DEPLOY_ENV $DEPLOY_IN

# By default, use the "full set of R&D libraries"
# TODO! Add some switch here if you want more FDA-friendly repo with only the libraries that you need for deploy
ARG IMAGE_TYPE=dev
ENV PROD_IMAGE $IMAGE_TYPE

# Define environment

# https://stackoverflow.com/a/63643361/18650369
ENV USER dockeruser
ENV HOME /home/$USER
RUN useradd -m $USER && echo $USER:$USER | chpasswd && adduser $USER sudo
RUN chown $USER:$USER $HOME
# ENV POETRY_VERSION=1.2.2

# "Layer of your basics"
# i.e. what you need to install to this Ubuntu (without GPU)
# https://www.stereolabs.com/docs/docker/creating-your-image/#optimize-your-image-size
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && \
    # https://github.com/phusion/baseimage-docker/issues/319#issuecomment-1107760238
    apt-get install -y --no-install-recommends apt-utils && \
    apt install software-properties-common curl -y && \
    apt-get install git-all -y && \
    # apt remove --purge -y python3.10 && \
    # https://pythonspeed.com/articles/base-image-python-docker-images/
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install -y python3-pip python3.8-dev python3.8-distutils python3.8-venv && \
    # https://stackoverflow.com/questions/63936578/docker-how-to-make-python-3-8-as-default \
    # https://github.com/pyenv/pyenv
    ln -sf /usr/bin/python3.8 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.8 /usr/bin/python && \
    # python3 --version && \
    apt install -y s3fs

# not sure if really optimal "multi-staging"?
FROM base as minivess-base

# Switch to non-privileged user from superuser
USER $USER

# Install Poetry (or via pip: https://stackoverflow.com/a/54763270/18650369)
RUN curl -sSL https://install.python-poetry.org | python3 - && \
    $HOME/.local/bin/poetry --version

# Install custom packages with Poetry
# i.e. what Python packages you need to install to run the ML code
# - "Alternatively, you can call Poetry explicitly with `/root/.local/bin/poetry`."
COPY ./pyproject.toml ./poetry.lock ./

# https://stackoverflow.com/a/54763270/18650369 for advanced example with "production" and "development" images:
# * This way $YOUR_ENV will control which dependencies set will be installed: all (default)
#   or production only with --no-dev flag.

# Some people prefer to export the Poetry environment to requirements.txt and install from there
# without any virtual environment
# https://www.reddit.com/r/django/comments/syfp0l/if_i_use_poetry_for_development_should_my_project/
# Not sure why the path export was not "seen" here?
# https://github.com/python-poetry/poetry-plugin-export
RUN echo "You would have a chance here to make your poetry export here conditional to your desired image dependencies"
RUN echo " i.e. have the same Docker pipeline handle both development (R&D) and production (subset, 'FDA') dependencies"
RUN echo $PROD_IMAGE
RUN if [ "$PROD_IMAGE" = "dev" ] ; then \
        $HOME/.local/bin/poetry export --format requirements.txt --without-hashes --output $HOME/requirements.txt  && \
        # Uninstall Poetry to reduce image size? Or is this not included in minivess-mlops by default?
        curl -sSL https://install.python-poetry.org | python3 - --uninstall; \
    else \
        # how to check this above already?
        echo "Nothing yet defined for production docker"; \
    fi

# Install the required packages finally
ENV PATH="${PATH}:$HOME/.local/bin"
RUN pip install --no-cache-dir --user -r $HOME/requirements.txt

# Not sure if this is very optimal for multi-stage builds? or if this naming hierarchy is needed?
FROM minivess-base as minivess-mlops

# Install NVIDIA
# https://nvidia.github.io/nvidia-docker/
# 1)
# "Warning: apt-key is deprecated. Manage keyring files in trusted.gpg.d instead (see apt-key(8))."
# 2)
# W: https://nvidia.github.io/libnvidia-container/stable/ubuntu18.04/amd64/InRelease:
#    Key is stored in legacy trusted.gpg keyring (/etc/apt/trusted.gpg),
#    see the DEPRECATION section in apt-key(8) for details
#RUN echo $DEPLOY_ENV
#RUN if [ "$DEPLOY_ENV" = "local" ] ; then \
#        export DEBIAN_FRONTEND=noninteractive && \
#        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey && \
#        gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg && \
#        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list && \
#        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' && \
#        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list && \
#        apt-get update && \
#        apt-get install nvidia-container-runtime -y; \
#    else \
#        # how to check this above already?
#         echo "Skipping NVIDIA install"; \
#    fi
#
#FROM minivess-mlops as minivess-mlops-NVIDIA