#!/bin/bash

docker compose build --build-arg USERNAME=$USER --build-arg USER_UID=$(id -u) --build-arg USER_GID=$(id -u) dev
docker compose run --rm -it dev bash
