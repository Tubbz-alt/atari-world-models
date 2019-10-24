# Atari World Models

**Type:** Bring your own method 

**Topic:** Deep Reinforcement Learning

## Summary

This project attempts to recreate the neural network described in the first
part of the "[World Models](#ref-world-models)" paper. After implementation we
plan to apply it to different Atari games included in the
[openai-gym](#ref-gym) library and hopefully enhance and adapt the system to be
better suited to different Atari-style games. The resulting application lets the end
user watch the network play and provides visualizations for the decisions the
network makes when playing the well-know game Pong.

## Dataset description

There is no need to prepare a dataset - we will use the [openai-gym](#ref-gym)
library and the included games.

## Languages, libraries, frameworks

- Python 3.6.9
- [pytorch 1.2](#ref-pytorch)
- [openai-gym 0.15.3](#ref-gym)

## Approach description

The "World Models" paper is split in two parts. the first part describes a ... and the second
part ...  This Neural Network was able to solve the openai-gym racing task.


We will use the game "Pong-v0" as a testbed for the first implementation. Pong
was chosen as a first candidate because of the limited computing resources at
hand and it's simplicity. After having a working implementation for "Pong-v0",
we will try to move on to "harder" games (Riverraid-v0, Breakout-v0, ...).
Starting with Pong-v0 and moving on to "more difficult" games will hopefully
allow us to gradually see a point where the approach does not work as good any
longer and is in need of adaption. The action space of Pong-v0 and it's visual
simplicity allows us to further get a good understanding of the network's
decisions.

# Work-breakdown structure and progress tracking

- [ ] Exercise 1
    - [x] <a name="t-broad-topic"></a>Find a broad topic
        - [ ] Get an overview of Deep Reinforcement Learning approaches
    - [ ] <a name="t-specific-topic"></a>Find a specific topic
    - [ ] <a name="t-lit-research"></a>Literature and tutorial research / reading
        - [ ] Read "World Models" paper
        - [ ] Read ... paper
        - [ ] Read ... paper
    - [ ] <a name="t-del-ex1"></a>Deliverables for Exercise 1
        - [ ] Create repository on github
        - [ ] Create development environment
            - [ ] Decide on a framework
            - [ ] Create Docker container
        - [ ] Write README.md
- [ ] Exercise 2
    - [ ] <a name="t-start-report"></a> Start with the report
        - [ ] Create LaTeX template
        - [ ] Document progress and decisions
        - [ ] Generate data to use for plots/figures
    - [ ] <a name="t-implement"></a>Implement approach from paper
        - [ ] Implement the VAE part
        - [ ] Implement the RNN part
        - [ ] Implement the Controller part
        - [ ] Training runs
        - [ ] Try to replicate the findings in the paper with pong
    - [ ] <a name="t-fine-tune"></a>Try other games and fine-tune
        - [ ] Training runs
    - [ ] <a name="t-decisions"></a>Network decisions
        - [ ] Get an understanding of the decisions
        - [ ] Find a way to visualize some of the decisions
- [ ] Exercise 3
    - [ ] <a name="t-application"></a>Implement web-app
        - [ ] Host implementation on private server
        - [ ] Create a frontend / visualizer
            - [ ] Settle on a framework
            - [ ] Implement basic frontend
    - [ ] <a name="t-finish-report"></a>Finish and cleanup report
    - [ ] <a name="t-presentation"></a>Presentation
        - [ ] Create beamer LaTeX template
        - [ ] Write presentation
        - [ ] Practise presentation

# Estimates

These are the time estimates in hours including the actual time spent on a topic.

| Estimate (h) | Actual (h) | Task description | 
| ---: | ---: | :--- | 
| 4  |  | [Find a broad topic](#t-broad-topic) | 
| 4  |  | [Find a specific topic](#t-specific-topic) | 
| 12 |  | [Literature and tutorial research / reading](#t-lit-researc) | 
| 8  |  | [Deliverables for Exercise 1](#t-del-ex1) | 
| 2  |  | [Start with the report](#t-start-report) |
| 16 |  | [Implement approach from paper](#t-implement) |
| 12 |  | [Try other games and fine-tune](#t-fine-tune) | 
| 4  |  | [Add findings to report and iterate](#t-start-report) |
| 12 |  | [Network decisions](#t-decisions) |
| 12 |  | [Implement web-app](#t-application) |
| 4  |  | [Finish and cleanup report](#t-finish-report) | 
| 6  |  | [Presentation](#t-presentation) |
---
|84  |  |  |

# References

* <a name="ref-world-models"></a> https://arxiv.org/abs/1809.01999 and https://worldmodels.github.io/
* <a name="ref-gym"></a> https://gym.openai.com/
* <a name="ref-pytorch"></a> https://pytorch.org/
