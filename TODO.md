# TODO

## General

* Add a command that does the whole training
* Add progressbar to training loops (maybe tqdm)
* Make z_size available on VAE
* Use defaults that are sensible
* Generate more observations
* Collect all hyperparameters and put them in one place

## VAE

* Add batch size to train-vae
* Add rotations 90, 180, 270 to dataset transformations
* Add split into train / test dataset and think about adding early stopping functionality

## Controller

* When training the controller we should average over the reward n times (submit the same solution
  n times)
