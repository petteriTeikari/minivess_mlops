#!/bin/sh -ex
# mkdir -p ${S3_MOUNTPOINT_PATH} # done already in Dockerfile
# cat $HOME/.aws/credentials # if you want to check that the AWS credentials are mounted correctly when debugging
echo Your container Python training args are: "$@"
mount-s3 --region eu-north-1  --allow-delete --allow-other ${S3_ARTIFACTS} ${DOCKER_ARTIFACTS}
mount-s3 --region eu-north-1  --allow-delete --allow-other ${S3_CACHE} ${DOCKER_CACHE}
python run_training.py "$@"