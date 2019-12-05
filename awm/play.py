import logging
from pathlib import Path

import gym
import torch
from torchvision import transforms
from torchvision.utils import save_image
from xvfbwrapper import Xvfb

from . import MODELS_DIR
from .controller import Controller
from .mdn_rnn import MDN_RNN
from .observations import transform
from .utils import Step
from .vae import VAE

logger = logging.getLogger(__name__)


class PlayGame(Step):
    hyperparams_key = "play_game"

    def __call__(self, step_limit, solution=None, stamp=None):
        logger.info("Playing game %s", self.game)

        with torch.no_grad():
            controller = Controller(self.game, self.models_dir)
            if solution is not None:
                controller.load_solution(solution)
            else:
                controller.load_state(stamp)

            vae = VAE(self.game, self.models_dir)
            vae.load_state()

            mdn_rnn = MDN_RNN(self.game, self.models_dir)
            mdn_rnn.load_state()

            env = gym.make(self.game.key)

            action = torch.zeros(self.game.action_vector_size)

            screen = env.reset()
            screen = transform(screen)
            screen.unsqueeze_(0)

            z, _, _ = vae.encoder(screen)
            _, _, _, h = mdn_rnn(z.unsqueeze(0), action.unsqueeze(0).unsqueeze(0))

            overall_reward = 0
            steps = 0
            while True:
                env.render()
                action = controller(z.squeeze(0).squeeze(0), h.squeeze(0).squeeze(0))

                screen, reward, done, _ = env.step(action.detach().numpy())
                overall_reward += reward
                screen = transform(screen)
                screen.unsqueeze_(0)

                z, _, _ = vae.encoder(screen)
                _, _, _, h = mdn_rnn(z.unsqueeze(0), action.unsqueeze(0).unsqueeze(0))

                if done or (step_limit and steps >= step_limit):
                    break

                steps += 1
            env.close()

            logger.info(
                "Game %s finished with reward %d", self.game.key, 1000 - overall_reward
            )

        return 1000 - overall_reward
