#!/bin/bash

image_name="asr_kws_multitask"
username="ISedunov"
container_name=${username}-${image_name}

docker stop "${container_name}"
docker rm "${container_name}"

docker run -it \
    --gpus all \
    --expose 22 -P \
    --shm-size 8G \
    --runtime=nvidia \
    -v $PWD/../../:/home/multitask_learning_ASR_KWS \
    --detach \
    --name "${container_name}" \
    --entrypoint /bin/bash \
    ${image_name}