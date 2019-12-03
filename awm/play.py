import gym
import torch
from torchvision import transforms
from torchvision.utils import save_image
from xvfbwrapper import Xvfb

from . import logger
from .controller import Controller
from .mdn_rnn import MDN_RNN
from .vae import VAE


def play_game(game, solution=None, stamp=None):
    logger.info("Playing game %s", game)

    tpi = transforms.ToPILImage()
    r = transforms.Resize((64, 64))
    tt = transforms.ToTensor()

    def t(screen):
        return tt(r(tpi(screen)))

    with torch.no_grad():
        controller = Controller()
        if solution is not None:
            controller.load_solution(solution)
        else:
            print("before")
            controller.load_state(game, stamp)
            print("after")

        vae = VAE()
        vae.load_state(game)

        mdn_rnn = MDN_RNN()
        mdn_rnn.load_state(game)

        env = gym.make(game)

        action = torch.zeros(3)

        screen = env.reset()
        screen = t(screen)
        screen.unsqueeze_(0)

        z, _, _ = vae.encoder(screen)
        _, _, _, h = mdn_rnn(z.unsqueeze(0), action.unsqueeze(0).unsqueeze(0))

        action_every_steps = 10
        step = 0
        overall_reward = 0
        step_limit = 6000
        steps = 0
        while True:
            env.render()

            # if step == action_every_steps:
            action = controller(z.squeeze(0).squeeze(0), h.squeeze(0).squeeze(0))
            # logger.info("Choosing action: %s", action)

            screen, reward, done, _ = env.step(action.detach().numpy())
            overall_reward += reward
            screen = t(screen)
            screen.unsqueeze_(0)

            z, _, _ = vae.encoder(screen)
            _, _, _, h = mdn_rnn(z.unsqueeze(0), action.unsqueeze(0).unsqueeze(0))

            if done or steps >= step_limit:
                break

            steps += 1
        env.close()

        logger.info("Game %s finished with reward %d", game, 1000 - overall_reward)

    return 1000 - overall_reward
