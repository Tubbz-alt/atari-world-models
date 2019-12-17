# Atari World Models

**Type:** Bring your own method (and a somewhat different dataset)

**Topic:** Deep Reinforcement Learning

[![Build Status](https://travis-ci.com/chsigi/atari-world-models.svg?branch=master)](https://travis-ci.com/chsigi/atari-world-models)

# Quick preview
<!---
## VAE
![VAE Output](/images/carracing-vae.png?raw=true "VAE Output")

## Controller
-->

Untrained                                                                                 | 50 hours of training
:----------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------:
![Untrained Controller](/images/untrained-controller.gif?raw=true "Untrained Controller") | ![Trained Controller](/images/trained-controller-50.gif?raw=true "Trained Controller")

# Summary

This project attempts to recreate the approach described in the first part of
the ["World Models"](https://arxiv.org/abs/1803.10122) paper.  According to the
authors the agent trained with the "World Models" approach was the first agent
to be able to solve the "CarRacing-v0" game included in the
[OpenAI Gym](https://gym.openai.com/) library.  We will apply the "World
Models" approach to a select group of Atari games included in the OpenAI Gym
library.  A different deep reinforcement learning approach from 2013 described
in [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
was able to achieve great results for 7 different Atari games.
The "Controller" part in the "World Models" approach seems to be a promising
target for trying to improve the scores achievable on Atari-style games without
having massive computing resources at hand.

## Dataset description

There is no need to prepare a dataset - we will use the OpenAI Gym library and the
included games.

## Languages, libraries, frameworks

- Python 3.6.9
- [pytorch 1.2](https://pytorch.org/)
- gym 0.15.3
- cma 2.7.0

## Approach description

**Inputs:**
The only inputs our network will receive are the video screen, a reward signal
and a game-over signal. No other access to the emulator state is allowed.

**Preprocessing:**
As described in
["Revisiting the Arcade Learning Environment"](https://arxiv.org/abs/1709.06009)
we will apply the customary preprocessing to the screen to account for Atari specific
artefacts (frame stacking, color ...). It seems that it will also be necessary to do some kind
of reward normalization.

**Comparison:** 
To compare our results we will use the guidelines given in
["Revisiting the Arcade Learning Environment"](https://arxiv.org/abs/1709.06009) and
apply them as best as possible to the OpenAI Gym environment.

**Implementation:**
See the work-breakdown structure for [detailed implementation steps](#t-implement).
After finishing the implementation of the "World Models" approach,
we will use the game "Pong-v0" as a first testbed. Pong
is a suitable first candidate because of its limited action space and visual
simplicity. After having a working implementation for "Pong-v0",
we will try to move on to "harder" games (Seaquest-v0, SpaceInvaders-v0, ...).
and gradually improve the network.

# Work-breakdown structure and progress tracking

- [ ] Exercise 1
    - [x] <a name="t-broad-topic"></a>**Find a broad topic**
        - [x] Get an overview of Deep Reinforcement Learning approaches
    - [x] <a name="t-specific-topic"></a>**Find a specific topic**
    - [x] <a name="t-lit-research"></a>**Literature and tutorial research / reading**
        - [x] Read "World Models" paper
        - [x] Read "Playing Atari with ..." paper
        - [x] Read "Revisiting the ..." paper
        - [x] Read "OpenAI Gym" paper
    - [x] <a name="t-del-ex1"></a>**Deliverables for Exercise 1**
        - [x] Create repository on github
        - [x] Create development environment
            - [x] Decide on a framework
            - [x] Create Docker container
        - [x] Write README.md
- [ ] Exercise 2
    - [ ] <a name="t-start-report"></a>**Start with the report**
        - [ ] Create LaTeX template
        - [ ] Document progress and decisions
        - [ ] Generate data to use for plots/figures
    - [ ] <a name="t-implement"></a>**Implement approach from paper**
        - [x] Implement the observation gathering
        - [x] Implement the VAE part
        - [x] Implement the RNN + MDN part
        - [x] Implement the Controller + CMA-ES part
        - [ ] Try to replicate the findings in the paper with pong
    - [ ] <a name="t-fine-tune"></a>**Try other games and fine-tune**
        - [ ] Training runs
    - [ ] <a name="t-decisions"></a>**Network decisions (optional)**
        - [ ] Get an understanding of the decisions
        - [ ] Find a way to visualize some of the decisions
- [ ] Exercise 3
    - [ ] <a name="t-finish-impl"></a>**Finish and cleanup implementation**
    - [ ] <a name="t-application"></a>**Implement web-app (optional)**
        - [ ] Host implementation on private server
        - [ ] Create a frontend / visualizer
            - [ ] Settle on a framework
            - [ ] Implement basic frontend
    - [ ] <a name="t-finish-report"></a>**Finish and cleanup report**
    - [ ] <a name="t-presentation"></a>**Presentation**
        - [ ] Create beamer LaTeX template
        - [ ] Write presentation
        - [ ] Practise presentation

# Estimates

These are the time estimates in hours including the actual time spent on a topic.

| Estimate (h) | Actual (h) | Timelogs | Task description | 
| ---: | ---: | :--- | :--- | 
| 4  | 5 | 3, 2  | [Find a broad topic](#t-broad-topic) | 
| 4  | 8 | 4, 4 | [Find a specific topic](#t-specific-topic) | 
| 12 | 9 | 2, 4, 2, 1 | [Literature and tutorial research / reading](#t-lit-research) | 
| 8  | 4 | 2, 2 | [Deliverables for Exercise 1](#t-del-ex1) | 
| 2  |   |  | [Start with the report](#t-start-report) |
| 16 |   | 3, 3, 5, 3, 3, 4, 4, 3, 4, 4, 3 | [Implement approach from paper](#t-implement) |
| 16 |   | 4 | [Try other games and fine-tune](#t-fine-tune) | 
| 4  |   |  | [Add findings to report and iterate](#t-start-report) |
| 12 |   |  | [Network decisions (optional)](#t-decisions) |
| 8  |   |  | [Finish and cleanup implementation](#t-finish-impl) |
| 12 |   |  | [Implement web-app (optional)](#t-application) |
| 4  |   |  | [Finish and cleanup report](#t-finish-report) | 
| 6  |   |  | [Presentation](#t-presentation) |
| **108** | **24** | | |

# Error metric

## Description

Our error metric will be the points that are missing from a "perfect" score in the
game "CarRacing-v0". The "World Models" paper claims an average score of about 900 points - we
regard this value as a perfect score.

## Target

We target a point difference of about 100 points. Training will stop when our network
is able to achieve an average score of 800 per 10 runs.

## Actual

Our network is currently only able to achieve a score of up to 300 points - this is partialy due to the
fact that training the controller is very expensive. The current version of the controller was trained
on 4 CPUs for around ~50 hours, while the controller in the paper was trainined on a 64 core machine
for several thousand generations. The paper is also not very clear on certain aspects of the MDN-RNN, further
investigation/tuning is still necessary.

# Usage

The whole system is contained in a Docker container - you can either directly interact with the container
via docker-compose or use the provided Makefile targets. Most Makefile targets use sane defaults - if you need
to fine-tune, then you can use the default target to access the various subcommands and pass arguments.

## Full training

```
make observations GAME=CarRacing-v0
make vae GAME=CarRacing-v0
make z-values GAME=CarRacing-v0
make mdnrnn GAME=CarRacing-v0
make controller GAME=CarRacing-v0
```

## Watching the system in action

```
make play GAME=CarRacing-v0
```

## Custom interaction

```
make ARGS="--help"
make ARGS="-v CarRacing-0 gather-observations --help"
...
```

# Papers

1. "World Models" https://worldmodels.github.io/ and https://arxiv.org/abs/1803.10122
1. "Playing Atari with Deep Reinforement Learning" https://arxiv.org/abs/1312.5602
1. "Revisiting the Arcade Learning Environment" https://arxiv.org/abs/1709.06009
1. "OpenAI Gym" https://arxiv.org/abs/1606.01540

# Helpful tutorials / code samples

## VAE

1. https://www.jeremyjordan.me/variational-autoencoders/
1. https://jaan.io/what-is-variational-autoencoder-vae-tutorial/

## RNN

1. https://colah.github.io/posts/2015-08-Understanding-LSTMs/

## MDN

1. https://mikedusenberry.com/mixture-density-networks
1. https://github.com/sagelywizard/pytorch-mdn
