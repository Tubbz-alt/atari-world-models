import logging
import os
import random
import time
from itertools import zip_longest
from pathlib import Path

import gym
import matplotlib.pyplot as plt
import numpy as np
import retro
import torch
import torch.nn.functional as F
from torch import nn
from torchvision import datasets, transforms
from torchvision.utils import save_image

from . import DEVICE, MODELS_DIR
from .observations import load_observations
from .utils import StateSavingMixin, Step

NUMBER_OF_EPOCHS = 10
CREATE_PROGRESS_SAMPLES = True

logger = logging.getLogger(__name__)


def progress_samples(vae, dataset, game, epoch_number, samples_dir):
    """ Generate sample pictures that help visualize the training
    progress.
    """
    samples_dir = samples_dir / Path(game)
    samples_dir.mkdir(parents=True, exist_ok=True)

    filename_repr = samples_dir / "vae_repr_{}.png".format(epoch_number)
    filename_rand = samples_dir / "vae_rand_{}.png".format(epoch_number)

    # Select 32 random images from dataset and display them next to their
    # VAE representation
    images = None
    with torch.no_grad():
        for i in range(32):
            observation, _ = dataset[random.randint(0, len(dataset) - 1)]
            image = observation["screen"].unsqueeze(0)
            reconstruction, _, _ = vae(image)
            if images is not None:
                images = torch.cat([images, image, reconstruction])
            else:
                images = torch.cat([image, reconstruction])

        save_image(images, filename_repr)

    # Create a 64 random z vectors and decode them to images
    with torch.no_grad():
        z_size = 32
        rands = torch.randn(64, z_size)
        result = vae.decoder(rands)
        save_image(result, filename_rand)


class TrainVAE(Step):
    def __call__(
        self,
        number_of_epochs=NUMBER_OF_EPOCHS,
        create_progress_samples=CREATE_PROGRESS_SAMPLES,
    ):
        """ This is the main training loop of the VAE.

        Observations are loaded from *observations_dir* and training is performed
        for *number_of_epochs* epochs.
        """
        dataloader, dataset = load_observations(self.game, self.observations_dir)
        bs = 32
        vae = VAE(self.models_dir).to(DEVICE)
        optimizer = torch.optim.Adam(vae.parameters())

        vae.load_state(self.game)

        for epoch in range(number_of_epochs):
            cumulative_loss = 0.0
            for idx, (states, _) in enumerate(dataloader):
                images = states["screen"]
                optimizer.zero_grad()
                reconstructions, mu, logvar = vae(images)
                loss, bce, kld = loss_fn(reconstructions, images, mu, logvar)
                loss.backward()
                optimizer.step()
                cumulative_loss += loss.data / bs

            logger.info("%s: %s", epoch, cumulative_loss / len(dataloader))
            if create_progress_samples:
                progress_samples(vae, dataset, self.game, epoch, self.samples_dir)

        vae.save_state(self.game)


class VAE(StateSavingMixin, nn.Module):
    def __init__(self, models_dir=MODELS_DIR):
        super().__init__()
        self.models_dir = models_dir
        # See World Models paper - Appendix A.1 Variational Autoencoder
        self.encode = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            # 32 x 31 x 31
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            # 64 x 14 x 14
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            # -> 128 x 6 x 6
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            # -> 256 x 2 x 2
        )

        # bottleneck-size
        flattened_size = 1024
        bottleneck_size = 32

        self.bottle_mu = nn.Linear(flattened_size, bottleneck_size)
        self.bottle_logvar = nn.Linear(flattened_size, bottleneck_size)
        self.unbottle = nn.Linear(bottleneck_size, flattened_size)

        self.decode = nn.Sequential(
            # 1024
            nn.ConvTranspose2d(1024, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            # -> 128 x 5 x 5
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            # 64 x 13 x 13
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            # 32 x 30 x 30
            nn.ConvTranspose2d(32, 3, kernel_size=6, stride=2),
            nn.Sigmoid(),
            # 3 x 64 x 64
        )

    def build_z(self, mu, logvar):
        # logvar.mul_(0.5).exp_()
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decoder(self, z):
        batch_size = z.size(0)
        # Unbottle
        z = self.unbottle(z)
        # Fixup for decoding
        z = z.view(batch_size, 1024, 1, 1)
        y = self.decode(z)
        return y

    def encoder(self, x):
        batch_size = x.size(0)
        x = self.encode(x)
        x = x.view(batch_size, -1)
        # Press through bottleneck and build z
        mu, logvar = self.bottle_mu(x), self.bottle_logvar(x)
        z = self.build_z(mu, logvar)
        return z, mu, logvar

    def forward(self, x):
        z, mu, logvar = self.encoder(x)
        y = self.decoder(z)
        return y, mu, logvar


def loss_fn(reconstruction, original, mu, logvar):
    # This is the L2 loss, described in the World Models paper
    BCE = F.mse_loss(reconstruction, original, reduction="sum")

    # This is based heavily on the vae example from torch
    KLD = -0.5 * torch.sum(1 + 2 * logvar - mu.pow(2) - (2 * logvar).exp())
    return BCE + KLD, BCE, KLD


class PrecomputeZValues(Step):
    def __call__(self):
        dataloader, dataset = load_observations(
            self.game, self.observations_dir, batch_size=1, shuffle=False
        )
        with torch.no_grad():

            vae = VAE(self.models_dir).to(DEVICE)
            vae.load_state(self.game)

            # Will be filled with (z, disk_location) tuples
            results = []
            for idx, (observation, _) in enumerate(dataloader):
                disk_location = observation["disk_location"]
                z, _, _ = vae.encoder(observation["screen"])
                results.append((z[0], disk_location[0]))

        logger.debug("Precomputed z values")

        # combine z and the next z value
        results = zip_longest(
            results, results[1:], fillvalue=(torch.zeros(32), "invalid-path")
        )

        for ((z, disk_location), (next_z, next_disk_location)) in results:
            obj = np.load(disk_location, allow_pickle=True).item()
            obj["z"] = z
            obj["next_z"] = next_z
            np.save(disk_location, obj)

        logger.debug("Wrote precomputed z values to .npy files")
