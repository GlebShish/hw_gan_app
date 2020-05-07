#!/bin/bash

docker build .
img=$(docker images -q | sed -n 1p)
docker run -it $img

