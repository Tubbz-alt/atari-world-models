# TODO

## General

* Make actions_size, ... available as (global) constants
* Use defaults that are sensible

* Add progressbar to training loops (maybe tqdm)
* Generate nice gifs showing various stages of training

## VAE

* Add batch size to train-vae
* Add rotations 90, 180, 270 to dataset transformations
* Add split into train / test dataset and think about adding early stopping functionality

## Controller

* When training the controller we should average over the reward n times (submit the same solution
  n times)
