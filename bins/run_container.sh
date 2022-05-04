#!/bin/bash

image_name="kws_report"
username="ISedunov"
container_name=${username}-${image_name}

docker stop "${container_name}"
docker rm "${container_name}"

docker run -it \
    --gpus all \
    --expose 22 -P \
    --shm-size 8G \
    --runtime=nvidia \
    -v $PWD/../../:/home/key_word_spotting \
    --detach \
    --name "${container_name}" \
    --entrypoint /bin/bash \
    ${image_name}