# TODO

## General

* Add more logging output
    - Switch to a formatter class that uses style="{"
* Substitute device = cpu for gpu detection
* Use SAMPLES_PATH from __init__
* Move directory defaults to __init__
* Add progressbar to training loops (maybe tqdm)
* Make z_size available on VAE
* Use defaults that are sensible
* Use logger instances per module and setup logging with dictConfig
* Generate more observations
* Collect all hyperparameters and put them in one place

## VAE

* move state loading to VAE class
* Add batch size to train-vae
* Add rotations 90, 180, 270 to dataset transformations
* Add split into train / test dataset and think about adding early stopping functionality

## Controller

* When training the controller we should average over the reward n times (submit the same solution
  n times)
