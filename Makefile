SERVICE:=atariworldmodel
DC:=docker-compose
DC_RUN:=${DC} run --rm ${SERVICE}
FILE:=/workspace/r.py

train:
	xhost +
	${DC_RUN} ${FILE}

shell:
	xhost +
	${DC_RUN} /bin/bash

build:
	${DC} build

.PHONY: train shell build
