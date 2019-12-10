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

from .games import GymGame
from .utils import StateSavingMixin, Step
from .vae import VAE

SHOW_SCREEN = False

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


def get_best_averaged(solution_q, reward_q, solutions, rewards):
    best_reward = sorted(rewards.items(), key=lambda x: x[1])[0]
    best_solution = solutions[best_reward[0]]

    average_over = 10

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
        self, reward_threshold, step_limit, show_screen=SHOW_SCREEN,
    ):
        logger.info("Training the controller for game %s", self.game)

        controller = Controller(self.game, self.models_dir)
        controller.load_state(stamp=53)

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
        population_size = 10

        es = cma.CMAEvolutionStrategy(
            parameters.view(-1).detach(), sigma, {"popsize": population_size}
        )

        epoch = 0
        do_eval = 2  # evaluate every other epoch

        solution_q = Queue()
        reward_q = Queue()
        stop = Event()
        for _ in range(multiprocessing.cpu_count()):
            Process(
                target=worker,
                args=(self.game, play_game, solution_q, reward_q, stop, show_screen,),
            ).start()

        highest_reward = None

        while not es.stop():
            # solutions is a list containing *population_size* numpy arrays
            # with proposed parameters
            solutions = es.ask()

            # Turn into dict with index as key - this makes it easier to match
            # the reward to the solution.
            solutions = dict(enumerate(solutions))

            logger.info("Submitting solutions")
            for solution_id, solution in solutions.items():
                solution_q.put((solution_id, solution))

            logger.info("Collecting rewards")
            rewards = {}
            for _ in range(len(solutions)):
                solution_id, reward = reward_q.get()
                rewards[solution_id] = reward

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
            if epoch % do_eval == do_eval - 1:
                solution, reward = get_best_averaged(
                    solution_q, reward_q, solutions, rewards
                )

                if highest_reward is None or highest_reward > reward:
                    highest_reward = reward
                    logger.info("Found new highest reward: %s", highest_reward)

                    # Load solution into controller and then save controller state
                    # to disk
                    controller = Controller(self.game, self.models_dir)
                    controller.load_solution(solution)
                    controller.save_state(stamp=str(epoch))

                if highest_reward and highest_reward <= reward_threshold:
                    break

            epoch += 1

        # Signal stop to all workers
        stop.set()


class Controller(StateSavingMixin, nn.Module):
    def __init__(self, game: GymGame, models_dir: Path):
        super().__init__()
        self.game = game
        self.models_dir = models_dir

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

    def forward(self, z, h):
        return self.f(torch.cat((z, h)))
