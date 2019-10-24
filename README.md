# Topic, summary and project type

## Project description

**Type:** Bring your own method 

Inspired by the World Models paper (https://worldmodels.github.io/ and  [1]
) we want to first replicate the system
described in the first part of the paper and apply it to openai gym's atari games. After
gaining a detailed understanding of the approach used in the paper we will
try to modify/improve the approach and check if the system is able to transfer
it's abilities to similar games and try to find a way to visualize/describe
the network actions.

The resulting application visualizes the networks decisions and lets provides an
easy to setup environment to start experimenting with the WorldModels approach.

## Dataset description

There is no need to prepare a dataset, because we can use openai-gym and the included games.

## Approach description

Start with pong and try to move on from there. This allows us to see a point
where the approach does not work as good any longer.
Pong was choosen as a first candidate because of the limited computing resources at hand - might be
more interesting to apply the worldmodel approach to games like sonic.

Implement the approach described in the world model paper.


# Work-breakdown structure and progress tracking


- [ ] Exercise 1
    - [x] <a name="t-broad-topic"></a> Find a broad topic
        - [ ] Get an overview of Deep Reinforcement Learning approaches
    - [ ] Settle on a specific topic
    - [ ] Literature and tutorial research
        - [ ] Read "World Models" paper
        - [ ] Read ... paper
        - [ ] Read ... paper
    - [ ] Create repository on github
    - [ ] Create development environment
        - [ ] Decide on a framework
        - [ ] Create Docker container
    - [ ] Write README.md
- [ ] Exercise 2
    - [ ] Start with the report
        - [ ] Create LaTeX template
        - [ ] Document progress and decisions
        - [ ] Generate data to use for plots/figures
    - [ ] Implement approach from paper
        - [ ] Implement the VAE part
        - [ ] Implement the RNN part
        - [ ] Implement the Controller part
        - [ ] Training runs
        - [ ] Try to replicate the findings in the paper with pong
    - [ ] Try other games and finetune
        - [ ] Training runs
    - [ ] Network decisions
        - [ ] Get an understanding of the decisions
        - [ ] Find a way to visualize some of the decisions
- [ ] Exercise 3
    - [ ] Application
        - [ ] Host implementation on private server
        - [ ] Create a frontend / visualizer
            - [ ] Settle on a framework
            - [ ] Implement basic frontend
    - [ ] Finish and cleanup report
    - [ ] Presentation
        - [ ] Create beamer LaTeX template
        - [ ] Write presentation
        - [ ] Practise presentation

# Estimates

| Estimate (h) | Actual (h) | Task description | 
| ---: | ---: | :---: | 
| 4h |  | [Find a broad topic](#t-broad-topic) | 
| 4h |  | [Find a specific topic](#t-specific-topic) | 
| 12h |  | Literatur and tutorial research / reading | 
| 8h |  | Repository and development environment | 
| 2h |  | Write README.md | 
| 2h |  | Create report template |
| 12h |  | Implement approach from paper |
| 4h |  | replicate findings |
| 12h |  | other games and finetuning | 
| 4h |  | add findings to report and iterate |
| 12h |  | network decisions |
| 12h |  | Implement application |
| 4h |  |  Finish and cleanup report | 
| 4h |  | Create presentation |
| 2h |  | Practise presentation |

# References

[1] https://arxiv.org/abs/1809.01999
