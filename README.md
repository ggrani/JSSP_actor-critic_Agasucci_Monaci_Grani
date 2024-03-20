# JSSP_actor-critic_Agasucci_Monaci_Grani
This repository contains the source code and the data related to the paper [An actor-critic algorithm with policy gradients to solve the job shop
scheduling problem using deep double recurrent agents](https://doi.org/10.1016/j.ejor.2023.07.037)
by Marta Monaci, Valerio Agasucci and Giorgio Grani.

```
@article{monaci2024actor,
title = {An actor-critic algorithm with policy gradients to solve the job shop scheduling problem using deep double recurrent agents},
journal = {European Journal of Operational Research},
volume = {312},
number = {3},
pages = {910-926},
year = {2024},
issn = {0377-2217},
doi = {https://doi.org/10.1016/j.ejor.2023.07.037},
author = {Marta Monaci and Valerio Agasucci and Giorgio Grani}
}
```
## Installation
The neural network models are implemented using [Pytorch](https://pytorch.org/).

## Requirements
The file requirements.txt reports the list of packages that must be installed to run the code. You can add a package to your environment via pip or anaconda using either pip install "package" or conda install "package". The code can be runned either on GPUs or CPUs.

## Configuration and running

In order to test the saved model, you just need to run `main_test.py`. 

Depending on the set of instances you want to test the model on, you have simply to modify the `set_to_test` parameter as follows:

* Taillard benchmark set (insert **'TaiBenchmarkSet**)
* Taillard generated set (insert **'TaiGeneratedSet'**)
* Gaussian Set (insert **'GaussianSet'**)
* Poisson set (insert **'PoissonSet'**)

To generate new sets of instances:

* according to the Taillard method[^1] (insert **'TaillardGenerator'** and modify the related parameters as you prefer)
* according to the Gaussian method (insert **'GaussianGenerator'** and modify the related parameters as you prefer)
* according to the Poisson method (insert **'PoissonGenerator'** and modify the related parameters as you prefer)

[^1]: Taken from https://github.com/zcaicaros/L2D

## Results

The output of the experiments can be found in folder **Results_TEST/**, where:

**Statistics_`set_to_test`.xlsx** will report the statistics of the experiments performed on the related set of instances.

## Team

Contributors to this code:

* [G Grani](https://github.com/GiorgioGrani)
* [Marta Monaci](https://github.com/m-monaci)
* [Valerio Agasucci](https://github.com/Valerio1994a)

# License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

* [MIT License](https://opensource.org/licenses/mit-license.php)
* Copyright 2022 © Marta Monaci, Valerio Agasucci, Giorgio Grani
