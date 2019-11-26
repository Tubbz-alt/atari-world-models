SERVICE:=atariworldmodel
DC:=docker-compose
DC_RUN:=${DC} run --rm ${SERVICE}
DC_RUN_TRAVIS:=${DC} -f docker-compose-travis.yml run --rm ${SERVICE}
MODULE:=awm
DC_USER:=$(shell id -u):$(shell id -g)

deep: export DC_USER:=${DC_USER}
deep:
	xhost +
	${DC_RUN} python -m ${MODULE} ${ARGS}

shell: export DC_USER:=${DC_USER}
shell:
	xhost +
	${DC_RUN} /bin/bash

build:
	${DC} build

tests: export DC_USER:=${DC_USER}
tests:
	${DC_RUN} pytest -sv ${MODULE}

tests-travis: export DC_USER:=${DC_USER}
tests-travis:
	${DC_RUN_TRAVIS} pytest -sv ${MODULE}

style: export DC_USER:=${DC_USER}
style:
	${DC_RUN} ./isort.sh
	${DC_RUN} ./black.sh

.PHONY: deep shell build style tests tests-travis
