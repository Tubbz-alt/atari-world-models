import logging

import gym
import torch
from gym.wrappers.monitor import Monitor

from .controller import Controller
from .mdn_rnn import MDN_RNN, sample
from .observations import transform
from .utils import Step
from .vae import VAE

logger = logging.getLogger(__name__)


class PlayGame(Step):
    hyperparams_key = "play_game"

    def generator(self):
        """ Play the game using a generator that returns live information.

        Used in the demo application.
        """
        logger.info("Playing game %s in generator mode", self.game)

        with torch.no_grad():
            controller = Controller(self.game, self.models_dir)
            controller.load_state("best")

            vae = VAE(self.game, self.models_dir)
            vae.load_state()

            mdn_rnn = MDN_RNN(self.game, self.models_dir)
            mdn_rnn.load_state()

            env = gym.make(self.game.key)
            if self.game.wrapper is not None:
                env = self.game.wrapper(env)

            action = torch.zeros(self.game.action_vector_size)

            screen = env.reset()
            screen = transform(screen)
            screen.unsqueeze_(0)

            z, _, _ = vae.encoder(screen)
            pi, sigma, mu, h = mdn_rnn(z.unsqueeze(0), action.unsqueeze(0).unsqueeze(0))

            while True:
                real_screen = env.render(mode="rgb_array")
                action = controller(z.squeeze(0).squeeze(0), h.squeeze(0).squeeze(0))
                actual_action = self.game.transform_action(action.detach().numpy())

                yield real_screen, screen, z, vae.decoder(z), vae.decoder(
                    sample(pi, sigma, mu).view(-1, 32)
                ), actual_action

                screen, reward, done, _ = env.step(actual_action)

                screen = transform(screen)
                screen.unsqueeze_(0)

                z, _, _ = vae.encoder(screen)
                pi, sigma, mu, h = mdn_rnn(
                    z.unsqueeze(0), action.unsqueeze(0).unsqueeze(0)
                )

                if done:
                    break

            env.close()

    def __call__(self, step_limit, solution=None, stamp=None, record=False):
        logger.info("Playing game %s with step_limit %d", self.game, step_limit)

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
            if self.game.wrapper is not None:
                env = self.game.wrapper(env)
            if record:
                env = Monitor(env, "monitor", force=True)

            action = torch.zeros(self.game.action_vector_size)

            screen = env.reset()
            screen = transform(screen)
            screen.unsqueeze_(0)

            z, _, _ = vae.encoder(screen)
            _, _, _, h = mdn_rnn(z.unsqueeze(0), action.unsqueeze(0).unsqueeze(0))

            # h = torch.tensor([[[0] * 256]], dtype=torch.float32)

            overall_reward = 0
            steps = 0

            while True:
                env.render()
                action = controller(z.squeeze(0).squeeze(0), h.squeeze(0).squeeze(0))

                actual_action = self.game.transform_action(action.detach().numpy())
                screen, reward, done, _ = env.step(actual_action)

                overall_reward += reward
                screen = transform(screen)
                screen.unsqueeze_(0)

                z, _, _ = vae.encoder(screen)
                _, _, _, h = mdn_rnn(z.unsqueeze(0), action.unsqueeze(0).unsqueeze(0))

                if done or (step_limit and steps >= step_limit):
                    if done:
                        logger.info("Game reached done")
                    else:
                        logger.info("Step limit reached")

                    break

                steps += 1
            env.close()

            # Transform reward to be useful to CMA-ES
            overall_reward = self.game.transform_overall_reward(overall_reward)

            logger.info("Game %s finished with reward %d", self.game.key, overall_reward)

        return overall_reward
