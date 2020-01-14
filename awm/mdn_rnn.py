import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torchvision.utils import save_image
from tqdm import tqdm

from . import CREATE_PROGRESS_SAMPLES, DEVICE
from .games import GymGame
from .observations import load_observations
from .utils import StateSavingMixin, Step
from .vae import VAE

logger = logging.getLogger(__name__)


def sample(pi, sigma, mu):
    """ Take a sample from the gaussian mixtures output by the MDN.

    The Gumbel sampling idea is taken from
    https://github.com/hardmaru/pytorch_notebooks/blob/master/mixture_density_networks.ipynb
    """

    def gumbel_sample(x, axis):
        """ Pick one of our gaussians """
        x = x.data.numpy()
        z = np.random.gumbel(loc=0, scale=1, size=x.shape)
        z = np.log(x) + z
        return z.argmax(axis=axis)

    k = gumbel_sample(pi, 2)
    sigma = sigma.squeeze(0).squeeze(0)
    mu = mu.squeeze(0).squeeze(0)

    indices = (k[-1][-1], np.arange(VAE.z_size))
    randoms = np.random.randn(VAE.z_size)
    result = randoms * sigma.data.numpy()[indices] + mu.data.numpy()[indices]
    return torch.tensor(result, dtype=torch.float32)


def progress_samples(game, dataset, mdn_rnn, epoch, samples_dir):
    """ Create progress samples to visualize the progress of our MDN-RNN.
    """

    samples_dir = samples_dir / Path(game.key)
    samples_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        # Use same root directory for the models as MDNRNN
        vae = VAE(game, mdn_rnn.models_dir).to(DEVICE)
        vae.load_state()

        images = None

        for _ in range(8):
            observation, _ = dataset[random.randint(0, len(dataset) - 1)]
            image = observation["screen"].unsqueeze(0)
            z = observation["z"].unsqueeze(0)[:, None, :]
            action = torch.from_numpy(observation["action"][None, None, :])

            if images is None:
                images = torch.cat([image])
            else:
                images = torch.cat([images, image])

            for _ in range(16):
                # Do 16 prediction steps into the future
                pi, sigma, mu, _ = mdn_rnn(z, action)

                sampled = sample(pi, sigma, mu)
                z = sampled

                next_image = vae.decoder(sampled.view(-1, 32))
                z = z.unsqueeze(0).unsqueeze(0)

                images = torch.cat([images, next_image])

        filename = "mdn_rnn_{}.png".format(epoch)
        save_image(images, samples_dir / filename, nrow=16 + 1)


def loss_function(pi, sigma, mu, target):
    """ Calculate the loss for our MDN.

    This is the negative-log likelihood - based on 
    https://github.com/sagelywizard/pytorch-mdn and adapted to use
    torch.distributions.Normal

    The original paper:

    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.120.5685&rep=rep1&type=pdf
    """
    sequence = pi.size()[1]

    target = target.view(-1, sequence, 1, VAE.z_size)
    normal_distributions = Normal(mu, sigma)

    # log_prob(y) ... log of pdf at value y
    log_probabilities = torch.exp(pi * normal_distributions.log_prob(target))

    result = -torch.log(torch.sum(log_probabilities, dim=2))
    return torch.mean(result)


class TrainMDNRNN(Step):
    """ The main training loop of the MDN RNN.

    Train the MDN RNN for at most *number_of_epoch* epochs or stop if
    *no_improvement_threshold* is exceeded.
    """

    hyperparams_key = "mdnrnn"

    def __call__(
        self,
        number_of_epochs,
        no_improvement_threshold,
        create_progress_samples=CREATE_PROGRESS_SAMPLES,
    ):
        logger.info("Starting to train the MDN-RNN for game %s", self.game.key)
        training, validation = load_observations(
            self.game,
            random_split=False,
            observations_dir=self.observations_dir,
            drop_z_values=False,
        )

        mdn_rnn = MDN_RNN(self.game, self.models_dir).to(DEVICE)
        mdn_rnn.load_state()

        optimizer = torch.optim.Adam(mdn_rnn.parameters())

        # mini_batch_size = 1

        last_validation_loss = None
        no_improvement_count = 0

        for epoch in range(number_of_epochs):
            mdn_rnn.train()
            training_loss = 0
            logger.info("Starting training")
            for observations, _ in tqdm(training):
                optimizer.zero_grad()
                mdn_rnn.set_hidden_state()

                z_vectors = observations["z"]
                actions = observations["action"]

                # Add mini-batch dimension size 1
                actions = actions[:, None, :]
                z_vectors = z_vectors[:, None, :]
                pi, sigma, mu, _ = mdn_rnn(z_vectors, actions)

                target = observations["next_z"]
                # print("tgt size", target.size())
                loss = loss_function(pi, sigma, mu, target)
                loss.backward()
                optimizer.step()

                training_loss += loss.item()

            training_loss /= len(training)

            # Validate
            with torch.no_grad():
                mdn_rnn.eval()
                validation_loss = 0
                logger.info("Starting validation")
                for observations, _ in tqdm(validation):
                    mdn_rnn.set_hidden_state()

                    z_vectors = observations["z"]
                    actions = observations["action"]

                    # Add mini-batch dimension size 1
                    actions = actions[:, None, :]
                    z_vectors = z_vectors[:, None, :]
                    pi, sigma, mu, _ = mdn_rnn(z_vectors, actions)

                    target = observations["next_z"]
                    loss = loss_function(pi, sigma, mu, target)

                    validation_loss += loss.item()
                validation_loss /= len(validation)

            if last_validation_loss and validation_loss >= last_validation_loss:
                logger.info("No improvement made!")
                no_improvement_count += 1
            else:
                # Reset improvement counter
                logger.info("This is an improvement - reseting improvement counter")
                no_improvement_count = 0

            last_validation_loss = validation_loss

            if create_progress_samples:
                progress_samples(
                    self.game, training.dataset, mdn_rnn, epoch, self.samples_dir
                )

            logger.info("e=%d tl=%f vl=%f", epoch, training_loss, validation_loss)

            mdn_rnn.save_state()

            # Stop if no improvements for some time
            if no_improvement_count >= no_improvement_threshold:
                logger.info("Stopping because no improvement threshold reached")
                break


class MDN_RNN(StateSavingMixin, nn.Module):
    """ This is the M part of the World Models approach

    Given a condensed representation of the screen obtained from the VAE
    (z) and the corresponding action, we try to predict the representation
    of the screen in the next step. Since this is an MDN RNN we do not
    directly predict z, but try to predict a mixture of gaussian distributions
    that match the distribution behind the "real" z.
    """

    hidden_units = 256

    def __init__(self, game: GymGame, models_dir: Path):
        self.game = game
        self.models_dir = models_dir
        super().__init__()

        # The World Models paper also uses 5 gaussian mixtures and 256 hidden
        # units in the LSTM. The value of the temperature parameter
        # (\tau in the paper) is also taken from the paper.
        self.gaussian_mixtures = 5
        self.temperature = 1.30

        self.lstm = nn.LSTM(VAE.z_size + self.game.action_vector_size, self.hidden_units)
        self.mu = nn.Linear(self.hidden_units, self.gaussian_mixtures * VAE.z_size)
        self.sigma = nn.Linear(self.hidden_units, self.gaussian_mixtures * VAE.z_size)
        self.pi = nn.Linear(self.hidden_units, self.gaussian_mixtures * VAE.z_size)

        self.hidden_state = self.set_hidden_state()

    def set_hidden_state(self):
        self.hidden_state = (
            torch.zeros(1, 1, self.hidden_units),
            torch.zeros(1, 1, self.hidden_units),
        )

    def forward(self, z, action):
        inputs = torch.cat((z, action), dim=-1)
        output, self.hidden_state = self.lstm(inputs, self.hidden_state)

        sequence = inputs.size()[1]

        # Arguments to reshape the output of the linear transformations into the
        # number of gaussian mixtures
        view_args = (-1, sequence, self.gaussian_mixtures, VAE.z_size)

        # Based on
        #   https://mikedusenberry.com/mixture-density-networks

        # FIXME: Do we need the temperature stuff the paper mentions?

        # pi ... is the mixing coefficient
        # mu, sigma ... mean, standard deviation

        # This ensures that the sum over all pi values equals 1
        pi = F.softmax(self.pi(output), dim=2).view(*view_args)
        # This makes sigma positive
        sigma = torch.exp(self.sigma(output)).view(*view_args)
        mu = self.mu(output).view(*view_args)

        return pi, sigma, mu, output
