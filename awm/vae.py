import logging
import random
from itertools import zip_longest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.utils import save_image
from tqdm import tqdm

from . import CREATE_PROGRESS_SAMPLES, DEVICE
from .games import GymGame
from .observations import load_observations
from .utils import StateSavingMixin, Step

logger = logging.getLogger(__name__)


def progress_samples(vae, dataset, game, epoch_number, samples_dir):
    """ Generate sample pictures that help visualize the training
    progress.
    """
    samples_dir = samples_dir / Path(game.key)
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
        rands = torch.randn(64, VAE.z_size)
        result = vae.decoder(rands)
        save_image(result, filename_rand)


class TrainVAE(Step):
    hyperparams_key = "vae"

    def __call__(
        self,
        number_of_epochs,
        no_improvement_threshold,
        create_progress_samples=CREATE_PROGRESS_SAMPLES,
    ):
        """ This is the main training loop of the VAE.

        Observations are loaded from *observations_dir* and training is performed
        for at most *number_of_epochs* epochs or aborted earlier if no improvements
        are made to the validation data for *no_improvement_threshold* number
        of epochs.
        """
        batch_size = 32
        training, validation = load_observations(
            self.game,
            random_split=True,
            observations_dir=self.observations_dir,
            batch_size=batch_size,
        )
        vae = VAE(self.game, self.models_dir).to(DEVICE)
        optimizer = torch.optim.Adam(vae.parameters())

        vae.load_state()

        logger.info("Started training vae")

        last_validation_loss = None
        no_improvement_count = 0

        for epoch in range(number_of_epochs):
            # Train
            vae.train()
            training_loss = 0
            logger.info("Starting training")
            for observations, _ in tqdm(training):
                images = observations["screen"]
                optimizer.zero_grad()
                reconstructions, mu, logvar = vae(images)
                loss, bce, kld = loss_fn(reconstructions, images, mu, logvar)
                loss.backward()
                optimizer.step()
                training_loss += loss.item()
            training_loss /= len(training)

            # Validate
            with torch.no_grad():
                vae.eval()
                validation_loss = 0
                logger.info("Starting validation")
                for observations, _ in tqdm(validation):
                    images = observations["screen"]
                    reconstructions, mu, logvar = vae(images)
                    loss, bce, kld = loss_fn(reconstructions, images, mu, logvar)
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

            logger.info("e=%d tl=%f vl=%f", epoch, training_loss, validation_loss)
            if create_progress_samples:
                progress_samples(
                    vae, training.dataset, self.game, epoch, self.samples_dir
                )

            vae.save_state()

            # Stop if no improvements for some time
            if no_improvement_count >= no_improvement_threshold:
                logger.info("Stopping because no improvement threshold reached")
                break


class VAE(StateSavingMixin, nn.Module):
    """ This is the implementation of the VAE.

    This takes in the screen and generates a vector of size *z_size* that "encodes
    important properties of the screen". This vector is later on passed to the MDN-RNN
    and the controller.
    """
    # The size of the latent vector (the size of the bottleneck)
    z_size = 32

    def __init__(self, game: GymGame, models_dir: Path):
        super().__init__()
        self.game = game
        self.models_dir = models_dir
        # See World Models paper - Appendix A.1 Variational Autoencoder
        self.encode = nn.Sequential(
            nn.Conv2d(self.game.color_channels, 32, kernel_size=4, stride=2),
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

        flattened_size = 1024

        self.bottle_mu = nn.Linear(flattened_size, self.z_size)
        self.bottle_logvar = nn.Linear(flattened_size, self.z_size)
        self.unbottle = nn.Linear(self.z_size, flattened_size)

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
            nn.ConvTranspose2d(32, self.game.color_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
            # 3 x 64 x 64
        )

    def build_z(self, mu, logvar):
        # logvar.mul_(0.5).exp_()
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decoder(self, z):
        """ Turn a z vector into an image """
        batch_size = z.size(0)
        # Unbottle
        z = self.unbottle(z)
        # Fixup for decoding
        z = z.view(batch_size, 1024, 1, 1)
        y = self.decode(z)
        return y

    def encoder(self, x):
        """ Create the z vector from a screen image """
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
    """ This step precomputes the z values for all observations.

    After training the VAE, we precompute the z values from all screens and in addition
    fill the next_z value slot in our observation files as well. We need the z and next_z
    values when training the MDN-RNN.
    """
    hyperparams_key = None

    def __call__(self):
        training, validation = load_observations(
            self.game,
            random_split=True,
            observations_dir=self.observations_dir,
            batch_size=1,
        )
        dataloaders = [training, validation]
        with torch.no_grad():

            vae = VAE(self.game, self.models_dir).to(DEVICE)
            vae.load_state()

            # Will be filled with (z, disk_location) tuples
            results = []
            logger.info("Computing z values")
            for dataloader in dataloaders:
                for observation, _ in tqdm(dataloader):
                    disk_location = observation["disk_location"]
                    z, _, _ = vae.encoder(observation["screen"])
                    results.append((z[0], disk_location[0]))

        # Sort by disk location to get a valid next_z
        results = sorted(results, key=lambda x: x[1])

        # combine z and the next z value
        results = list(
            zip_longest(
                results, results[1:], fillvalue=(torch.zeros(VAE.z_size), "invalid-path")
            )
        )

        logger.info("Writing z values to disk")
        for ((z, disk_location), (next_z, next_disk_location)) in tqdm(results):
            obj = np.load(disk_location, allow_pickle=True).item()
            obj["z"] = z
            obj["next_z"] = next_z
            np.save(disk_location, obj)

        # Write samples to disk for visual inspection
        with torch.no_grad():
            images = None
            for _ in range(32):
                idx = random.randint(0, len(results) - 1)
                ((z, _), (next_z, _)) = results[idx]
                z = z.unsqueeze(0)
                next_z = next_z.unsqueeze(0)

                image = vae.decoder(z)
                next_image = vae.decoder(next_z)

                if images is None:
                    images = torch.cat([image, next_image])
                else:
                    images = torch.cat([images, image, next_image])
            save_image(images, self.samples_dir / self.game.key / "z_next_z.png")
