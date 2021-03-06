SERVICE:=atariworldmodel
DC:=docker-compose
DC_RUN:=${DC} run --rm ${SERVICE}
DC_RUN_TRAVIS:=${DC} -f docker-compose-travis.yml run --rm ${SERVICE}
MODULE:=awm
DC_USER:=$(shell id -u):$(shell id -g)

ifeq ($(GAME),)
GAME:=CarRacing-v0
endif

# Run the demo application
demo: export DC_USER:=${DC_USER}
demo:
	xhost +
	${DC_RUN} python -m ${MODULE} -v ${GAME} demo

# Custom access to commandline via ARGS="..."
deep: export DC_USER:=${DC_USER}
deep:
	xhost +
	${DC_RUN} python -m ${MODULE} ${ARGS}

observations: export DC_USER:=${DC_USER}
observations:
	xhost +
	${DC_RUN} python -m ${MODULE} -v ${GAME} gather-observations

watch-observations: export DC_USER:=${DC_USER}
watch-observations:
	xhost +
	${DC_RUN} python -m ${MODULE} -v ${GAME} gather-observations --cpus-to-use 1 --show-screen --number-of-plays 1

vae: export DC_USER:=${DC_USER}
vae:
	xhost +
	${DC_RUN} python -m ${MODULE} -v ${GAME} train-vae

z-values: export DC_USER:=${DC_USER}
z-values:
	xhost +
	${DC_RUN} python -m ${MODULE} -v ${GAME} precompute-z-values

mdnrnn: export DC_USER:=${DC_USER}
mdnrnn:
	xhost +
	${DC_RUN} python -m ${MODULE} -v ${GAME} train-mdn-rnn

controller: export DC_USER:=${DC_USER}
controller:
	xhost +
	${DC_RUN} python -m ${MODULE} -v ${GAME} train-controller

watch-controller: export DC_USER:=${DC_USER}
watch-controller:
	xhost +
	${DC_RUN} python -m ${MODULE} -v ${GAME} train-controller --cpus-to-use 1 --show-screen

play: export DC_USER:=${DC_USER}
play:
	xhost +
	${DC_RUN} python -m ${MODULE} -v ${GAME} play-game --stamp best

record: export DC_USER:=${DC_USER}
record:
	xhost +
	${DC_RUN} python -m ${MODULE} -v ${GAME} play-game --stamp best --record

# Create a shell into the Docker container
shell: export DC_USER:=${DC_USER}
shell:
	xhost +
	${DC_RUN} /bin/bash

# Build the Docker container
build:
	${DC} build

tests: export DC_USER:=${DC_USER}
tests:
	${DC_RUN} pytest -sv ${MODULE} #-k ${K}

tests-travis: export DC_USER:=${DC_USER}
tests-travis:
	${DC_RUN_TRAVIS} pytest -sv ${MODULE}

# Do auto-formatting and import sorting
style: export DC_USER:=${DC_USER}
style:
	${DC_RUN} pyflakes awm
	${DC_RUN} ./isort.sh
	${DC_RUN} ./black.sh

clean_data:
	rm -rf models samples observations

.PHONY: deep shell build style tests tests-travis observations vae mdnrnn controller play
