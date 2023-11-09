#!/bin/sh -ex
# mkdir -p ${S3_MOUNTPOINT_PATH} # done already in Dockerfile
# cat $HOME/.aws/credentials # if you want to check that the AWS credentials are mounted correctly when debugging
echo Your container Python training args are: "$@"
# mount-s3 --region eu-north-1  --allow-delete --allow-other --dir-mode 0755 --file-mode 0644  ${S3_ARTIFACTS} ${DOCKER_ARTIFACTS}
# This is not very efficient, slow download speeds, pull data everytime to the instance:
# mount-s3 --region eu-north-1  --allow-delete --allow-other ${S3_CACHE} ${DOCKER_CACHE}
# There is an empty_dir in ${DOCKER_CACHE} created by the Dockerfile which will be populated by "dvc pull" below
#ls -ld ${DOCKER_ARTIFACTS}
id -u $USER
id -g $USER
#touch ${DOCKER_ARTIFACTS}/test # checking if you can write to the mounted volume,
echo "$PWD"
ls
dvc pull
python run_training.py "$@"
