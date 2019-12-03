import logging
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Normal
from torchvision.utils import save_image

from . import SAMPLES_DIR, OBSERVATIONS_DIR
from .observations import load_observations
from .utils import StateSavingMixin
from .vae import VAE

CREATE_PROGRESS_SAMPLES = True
NUMBER_OF_EPOCHS = 10

logger = logging.getLogger(__name__)


def progress_samples(game, dataset, mdn_rnn, epoch):
    """ Create progress samples to visualize the progress of our MDN-RNN.
    """

    samples_dir = SAMPLES_DIR / Path(game)
    samples_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Writing MDN_RNN progress samples for game %s", game)
    device = "cpu"
    with torch.no_grad():
        vae = VAE().to(device)
        vae.load_state(game)

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
                sampled = torch.sum(pi * torch.normal(mu, sigma), dim=2)
                sampled = sampled.view(1, 32)

                z = sampled

                next_image = vae.decoder(sampled)
                z = z[:, None, :]

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
    sequence = 1
    z_size = 32

    target = target.view(-1, sequence, 1, z_size)

    normal_distributions = Normal(mu, sigma)
    # log_prob(y) ... log of pdf at value y
    log_probabilities = torch.exp(pi * normal_distributions.log_prob(target))
    result = -torch.log(torch.sum(log_probabilities, dim=2))
    return torch.mean(result)


def train_mdn_rnn(
    game,
    observation_directory=OBSERVATIONS_DIR,
    create_progress_samples=CREATE_PROGRESS_SAMPLES,
    number_of_epochs=NUMBER_OF_EPOCHS,
):

    logger.info("Starting to train the MDN-RNN for game %s", game)
    dataloader, dataset = load_observations(
        game, observation_directory, drop_z_values=False
    )

    device = "cpu"

    mdn_rnn = MDN_RNN().to(device)
    mdn_rnn.load_state(game)

    optimizer = torch.optim.Adam(mdn_rnn.parameters(), lr=1e-3)

    mini_batch_size = 1

    for epoch in range(number_of_epochs):
        cumulative_loss = 0.0
        for idx, (observations, _) in enumerate(dataloader):
            optimizer.zero_grad()
            mdn_rnn.set_hidden_state()

            z_vectors = observations["z"]
            actions = observations["action"]

            # Add mini-batch dimension size 1
            actions = actions[:, None, :]
            z_vectors = z_vectors[:, None, :]
            pi, sigma, mu, _ = mdn_rnn(z_vectors, actions)

            target = observations["next_z"]
            loss = loss_function(pi, sigma, mu, target)
            loss.backward()
            optimizer.step()

            cumulative_loss += loss.data

        if create_progress_samples:
            progress_samples(game, dataset, mdn_rnn, epoch)

        logger.info("{:04}: {:.3f}".format(epoch, cumulative_loss / len(dataloader)))

    mdn_rnn.save_state(game)


class MDN_RNN(StateSavingMixin, nn.Module):
    def __init__(self):
        super().__init__()
        """ This is the M part of the World Models approach
        
        Given a condensed representation of the screen obtained from the VAE
        (z) and the corresponding action, we try to predict the representation
        of the screen in the next step. Since this is an MDN RNN we do not
        directly predict z, but try to predict a mixture of gaussian distributions
        that match the distribution behind the "real" z.
        """

        # The World Models paper also uses 5 gaussian mixtures and 256 hidden
        # units in the LSTM. The value of the temperature parameter
        # (\tau in the paper) is also taken from the paper.
        self.hidden_units = 256
        self.gaussian_mixtures = 5
        self.temperature = 1.30

        # input needs dimensions sequence length, batch_size, size of data
        self.z_size = 32
        action_size = 3

        self.lstm = nn.LSTM(self.z_size + action_size, self.hidden_units)
        self.mu = nn.Linear(self.hidden_units, self.gaussian_mixtures * self.z_size)
        self.sigma = nn.Linear(self.hidden_units, self.gaussian_mixtures * self.z_size)
        self.pi = nn.Linear(self.hidden_units, self.gaussian_mixtures * self.z_size)

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
        view_args = (-1, sequence, self.gaussian_mixtures, self.z_size)

        # Based on
        #   https://mikedusenberry.com/mixture-density-networks
        # and extended with temperature parameter

        # FIXME: Is the temperature stuff correct? We want to increase the
        # uncertainty ...

        # pi ... is the mixing coefficient
        # mu, sigma ... mean, standard deviation

        pi = self.pi(output).view(*view_args)
        sigma = self.sigma(output).view(*view_args)
        mu = self.mu(output).view(*view_args)

        pi = F.softmax(pi, dim=2) / self.temperature
        sigma = torch.exp(sigma)
        # sigma = sigma * (self.temperature ** 0.5)

        return pi, sigma, mu, output
