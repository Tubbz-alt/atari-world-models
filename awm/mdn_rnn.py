from pathlib import Path

import torch
from torch import nn

from . import logger
from .observations import OBSERVATION_DIRECTORY, load_observations


def train_mdn_rnn(
    game, observation_directory=OBSERVATION_DIRECTORY,
):

    logger.info("Starting to train the MDN-RNN for game %s", game)
    dataloader, dataset = load_observations(game, observation_directory, drop_z_values=False)

    device = "cpu"
    # bs = 32
    # vae = VAE().to(device)
    mdn_rnn = MDN_RNN().to(device)
    optimizer = torch.optim.Adam(mdn_rnn.parameters())

    # If there is a state file - load it
    models_dir = Path("models")
    state_file = models_dir / Path("{}-mdn-rnn.torch".format(game))
    if state_file.is_file():
        mdn_rnn.load_state_dict(torch.load(str(state_file), map_location=device))

    mini_batch_size = 1

    # sequence_length = 32
    # z_size = 32
    # action_size = 1

    # zs = torch.randn(sequence_length, mini_batch_size, z_size)
    # actions = torch.randn(sequence_length, mini_batch_size, action_size)

    # print("zs", zs)
    # print("actions", actions)
    #
    #    print("output", mdn_rnn(zs, actions))

    number_of_epochs = 1
    for epoch in range(number_of_epochs):
        cumulative_loss = 0.0
        for idx, (states, _) in enumerate(dataloader):
            # images = states["z"]
            z_vectors = states["z"]
            print(z_vectors.size())
            actions = states["action"]
            # Add mini-batch dimension size 1
            actions = actions[:, None, :]
            z_vectors = z_vectors[:, None, :]
            print("a", actions.size())
            print("z", z_vectors.size())
            optimizer.zero_grad()
            print(mdn_rnn(z_vectors, actions))
            # loss, bce, kld = loss_fn(reconstructions, images, mu, logvar)
            # loss.backward()
            # optimizer.step()
            # cumulative_loss += loss.data / bs

        print("{:04}: {:.3f}".format(epoch, cumulative_loss / len(dataloader)))

    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save(mdn_rnn.state_dict(), str(state_file))


class MDN_RNN(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=3):
        super().__init__()

        # z = 32 + 1 for action?

        # 5 gaussian mixtures

        # input needs dimensions sequence length, batch_size, size of data
        z_size = 32
        a_size = 3

        self.lstm = nn.LSTM(z_size + a_size, 256)

    def forward(self, z, action):
        inputs = torch.cat((z, action), dim=-1)
        outputs, hidden = self.lstm(inputs)
        return outputs, hidden
