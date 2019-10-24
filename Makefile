SERVICE:=atariworldmodel
DC:=docker-compose
DC_RUN:=${DC} run --rm ${SERVICE}
MODULE:=awm

deep:
	xhost +
	${DC_RUN} python -m ${MODULE} ${ARGS}

shell:
	xhost +
	${DC_RUN} /bin/bash

build:
	${DC} build

.PHONY: deep shell build
