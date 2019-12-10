import logging
import multiprocessing
import queue
from functools import partial
from multiprocessing import Event, Process, Queue
from pathlib import Path

import cma
import numpy as np
import torch
from torch import nn
from xvfbwrapper import Xvfb

from . import CPUS_TO_USE, SHOW_SCREEN
from .games import GymGame
from .utils import StateSavingMixin, Step
from .vae import VAE

logger = logging.getLogger(__name__)


def worker(game, play_game, solution_q, reward_q, stop, show_screen):

    if not show_screen:
        vdisplay = Xvfb()
        vdisplay.start()

    while not stop.is_set():
        try:
            (solution_id, solution) = solution_q.get(False, 0.1)
        except queue.Empty:
            pass
        else:
            name = multiprocessing.current_process().name
            logger.info("%s started working on %s", name, solution_id)
            reward = play_game(solution=solution)
            reward_q.put((solution_id, reward))

    if not show_screen:
        vdisplay.stop()


def get_best_averaged(solution_q, reward_q, solutions, rewards, average_over):
    best_reward = sorted(rewards.items(), key=lambda x: x[1])[0]
    best_solution = solutions[best_reward[0]]

    logger.info(
        "Submitting best solution %d times (best_reward: %s)", average_over, best_reward
    )
    for _ in range(average_over):
        # Solution id is not important here - same solution
        solution_q.put((None, best_solution))

    rewards = []
    for _ in range(average_over):
        _, reward = reward_q.get()
        rewards.append(reward)

    return best_solution, np.mean(rewards)


class TrainController(Step):
    hyperparams_key = "controller"

    def __call__(
        self,
        reward_threshold,
        step_limit,
        average_over,
        population_size,
        show_screen=SHOW_SCREEN,
        cpus_to_use=CPUS_TO_USE,
    ):
        logger.info("Training the controller for game %s", self.game)

        controller = Controller(self.game, self.models_dir)
        controller.load_state(stamp="best")
        best_reward = controller.load_best_reward()

        from .play import PlayGame

        play_game = PlayGame(
            self.game,
            models_dir=self.models_dir,
            observations_dir=self.observations_dir,
            samples_dir=self.samples_dir,
        )
        play_game = partial(play_game, step_limit)

        # action:
        # 0: steering -1 -> 1
        # 1: gas 0 -> 1
        # 2: break 0 -> 1

        parameters = next(controller.parameters())
        sigma = 0.1

        logger.info("Using cma with population_size %d", population_size)
        es = cma.CMAEvolutionStrategy(
            parameters.view(-1).detach(), sigma, {"popsize": population_size}
        )

        epoch = 0

        solution_q = Queue()
        reward_q = Queue()
        stop = Event()
        processes = []
        for _ in range(cpus_to_use):
            p = Process(
                target=worker,
                args=(self.game, play_game, solution_q, reward_q, stop, show_screen,),
            ).start()
            processes.append(p)

        while not es.stop():
            # solutions is a list containing *population_size* numpy arrays
            # with proposed parameters
            solutions = es.ask()

            # Turn into dict with index as key - this makes it easier to match
            # the reward to the solution.
            solutions = dict(enumerate(solutions))

            logger.info("Submitting solutions - each solution %d times", average_over)
            for solution_id, solution in solutions.items():
                for _ in range(average_over):
                    solution_q.put((solution_id, solution))

            logger.info("Collecting rewards")
            # This will map solution_ids to lists of rewards
            rewards = {}
            for _ in range(len(solutions) * average_over):
                solution_id, reward = reward_q.get()
                rewards.setdefault(solution_id, []).append(reward)

            # Calculate the mean
            logger.info("Calculating mean of rewards")
            rewards = {
                solution_id: np.mean(rewards) for solution_id, rewards in rewards.items()
            }

            def sorted_values(dict_):
                result = []
                for key in sorted(dict_.keys()):
                    result.append(dict_[key])
                return result

            # Our function we want to fit is the game we are playing
            logger.info("rewards: %s", rewards)
            es.tell(sorted_values(solutions), sorted_values(rewards))
            es.disp()

            # evaluation and saving
            if epoch % 2 == 1:
                solution, reward = get_best_averaged(
                    solution_q, reward_q, solutions, rewards, average_over
                )

                if best_reward is None or best_reward > reward:
                    best_reward = reward
                    logger.info("Found better reward: %s", best_reward)

                    # Load solution into controller and then save controller state
                    # to disk
                    controller = Controller(self.game, self.models_dir)
                    controller.load_solution(solution)
                    # Save twice - for easier loading
                    controller.save_state(stamp=str(epoch))
                    controller.save_state(stamp="best")
                    controller.save_best_reward(best_reward)

                if best_reward and best_reward <= reward_threshold:
                    break

            epoch += 1

        # Signal stop to all workers
        stop.set()
        for p in processes:
            p.join()


class Controller(StateSavingMixin, nn.Module):
    def __init__(self, game: GymGame, models_dir: Path):
        super().__init__()
        self.game = game
        self.models_dir = models_dir
        self.reward_file = self.models_dir / self.game.key / "controller_best_reward"

        hidden_size = 256

        self.f = nn.Sequential(
            nn.Linear(VAE.z_size + hidden_size, self.game.action_vector_size), nn.Tanh(),
        )

    def load_solution(self, solution):
        """ Load a solution vector obtained from ES-CMA into the parameters
        
        This currently ignores the biases.
        """
        weights = next(self.parameters())

        weights_from_solution = torch.tensor(solution).view(
            self.game.action_vector_size, -1
        )

        for actual, given in zip(weights, weights_from_solution):
            actual.data.copy_(given)

    def load_best_reward(self):
        if self.reward_file.is_file():
            with open(self.reward_file, "r") as f:
                reward = float(f.read().strip())
            logger.info("Loaded best reward: %s", reward)
        else:
            logger.warning("No reward file found")
            reward = None
        return reward

    def save_best_reward(self, best_reward):
        (self.models_dir / self.game.key).mkdir(exist_ok=True, parents=True)
        with open(self.reward_file, "w") as f:
            f.write(str(best_reward) + "\n")
            logger.info("Written best reward: %s", best_reward)

    def forward(self, z, h):
        return self.f(torch.cat((z, h)))
