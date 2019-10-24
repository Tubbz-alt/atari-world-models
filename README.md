# Atari World Models

**Type:** Bring your own method 

**Topic:** Deep Reinforcement Learning

## Summary

This project attempts to recreate the approach described in the first part of
the ["World Models"](https://arxiv.org/abs/1803.10122) paper.  The "World
Model" approach was able to beat the "CarRacing-v0" game included in the
[openai-gym](https://gym.openai.com/) library.  We will apply the "World Model"
approach to a select group of Atari games included in the openai-gym library.
A different deep reinforcement learning approach from 2013 described in
[Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
was able to achieve great results for 7 different Atari games. In the end the
"World Model" approach will be fine-tuned or improved to deliver great results
when applied to Atari-style games. A small web-application that lets people watch
the network play the Atari games will be provided.

## Dataset description

There is no need to prepare a dataset - we will use the openai-gym library and the
included games.

## Languages, libraries, frameworks

- Python 3.6.9
- [pytorch 1.2](https://pytorch.org/)
- openai-gym 0.15.3

## Approach description

**Inputs:** The only inputs our network will receive are the video screen, a reward
and a termination condition. No other access to the emulator state is allowed.

After finishing the implementation of the "World Models" approach,
we will use the game "Pong-v0" as a first testbed. Pong
is a suitable first candidate because of its limited action space and visual
simplicity. After having a working implementation for "Pong-v0",
we will try to move on to "harder" games (Seaquest-v0, SpaceInvaders-v0, ...).
and gradually improve the network. In the end we will provide a small web-application
that visualizes the network playing the games.

# Work-breakdown structure and progress tracking

- [ ] Exercise 1
    - [x] <a name="t-broad-topic"></a>**Find a broad topic**
        - [ ] Get an overview of Deep Reinforcement Learning approaches
    - [x] <a name="t-specific-topic"></a>**Find a specific topic**
    - [ ] <a name="t-lit-research"></a>**Literature and tutorial research / reading**
        - [ ] Read "World Models" paper
        - [x] Read "Playing Atari with ..." paper
        - [ ] Read "Revisiting the ..." paper
    - [ ] <a name="t-del-ex1"></a>**Deliverables for Exercise 1**
        - [x] Create repository on github
        - [x] Create development environment
            - [x] Decide on a framework
            - [x] Create Docker container
        - [ ] Write README.md
- [ ] Exercise 2
    - [ ] <a name="t-start-report"></a>**Start with the report**
        - [ ] Create LaTeX template
        - [ ] Document progress and decisions
        - [ ] Generate data to use for plots/figures
    - [ ] <a name="t-implement"></a>**Implement approach from paper**
        - [ ] Implement the VAE part
        - [ ] Implement the RNN part
        - [ ] Implement the Controller part
        - [ ] Training runs
        - [ ] Try to replicate the findings in the paper with pong
    - [ ] <a name="t-fine-tune"></a>**Try other games and fine-tune**
        - [ ] Training runs
    - [ ] <a name="t-decisions"></a>**Network decisions**
        - [ ] Get an understanding of the decisions
        - [ ] Find a way to visualize some of the decisions
- [ ] Exercise 3
    - [ ] <a name="t-application"></a>**Implement web-app**
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

| Estimate (h) | Actual (h) | Task description | 
| ---: | ---: | :--- | 
| 4  |  | [Find a broad topic](#t-broad-topic) | 
| 4  |  | [Find a specific topic](#t-specific-topic) | 
| 12 |  | [Literature and tutorial research / reading](#t-lit-research) | 
| 8  |  | [Deliverables for Exercise 1](#t-del-ex1) | 
| 2  |  | [Start with the report](#t-start-report) |
| 16 |  | [Implement approach from paper](#t-implement) |
| 12 |  | [Try other games and fine-tune](#t-fine-tune) | 
| 4  |  | [Add findings to report and iterate](#t-start-report) |
| 12 |  | [Network decisions](#t-decisions) |
| 12 |  | [Implement web-app](#t-application) |
| 4  |  | [Finish and cleanup report](#t-finish-report) | 
| 6  |  | [Presentation](#t-presentation) |
|**84**  |  |  |

# Papers

1. "World Models" https://worldmodels.github.io/ and https://arxiv.org/abs/1803.10122
1. "Playing Atari with Deep Reinforement Learning" https://arxiv.org/abs/1312.5602
1. "Revisiting the Arcade Learning Environment" https://arxiv.org/abs/1709.06009

# Helpful tutorials

## VAE

1. https://www.jeremyjordan.me/variational-autoencoders/
1. https://jaan.io/what-is-variational-autoencoder-vae-tutorial/

## RNN

1.  https://colah.github.io/posts/2015-08-Understanding-LSTMs/
