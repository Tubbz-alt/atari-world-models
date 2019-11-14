# TODO

## General

* Add more logging output
* Add progressbar to training loops (maybe tqdm)
* Make z_size available on VAE

## Observations

* Don't start worker with nothing to do when gathering observations
* Rename episodes to games in the context of observation gathering
* Rename target and source directory to observation directory

## VAE

* Rename episodes to epochs in train-vae context
* Add number-of-epochs to train-vae
* Add batch size to vae training
* Add rotations 90, 180, 270 to dataset
* Add resize to 64x64 so we can easily test the atari games (different screen size)
* Add split into train / test dataset and think about adding early stopping functionality
