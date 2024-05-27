# Bridging discrete and continuous state spaces
## Exploring the Ehrenfest process in time-continuous diffusion models

This repository contains the code for the paper "Bridging discrete and continuous state spaces: Exploring the Ehrenfest process in time-continuous diffusion models" by [Ludwig Winkler*](https://ludwigwinkler.github.io), [Lorenz Richter*](https://scholar.google.com/citations?hl=en&user=uxlQvnUAAAAJ), and Manfred Opper. 

This work was published at [ICML 2024](https://arxiv.org/pdf/2405.03549).

## Abstract

Generative modeling via stochastic processes has led to remarkable empirical results as well as to recent advances in
their theoretical understand- ing. In principle, both space and time of the processes can be discrete or continuous. In
this work, we study time-continuous Markov jump processes on discrete state spaces and investigate their correspondence
to state-continuous diffusion processes given by SDEs. In particular, we re- visit the Ehrenfest process, which
converges to an Ornstein-Uhlenbeck process in the infinite state space limit. Likewise, we can show that the time-
reversal of the Ehrenfest process converges to the time-reversed Ornstein-Uhlenbeck process. This observation bridges
discrete and continuous state spaces and allows to carry over methods from one to the respective other setting.
Additionally, we suggest an algorithm for training the time-reversal of Markov jump processes which relies on condi-
tional expectations and can thus be directly related to denoising score matching. We demonstrate our methods in multiple
convincing numerical experi- ments.

For the first time, we can directly link state-discrete continuous-time diffusion models to their time- and
space-continuous (SDE-based) counterparts, i.e. score-based generative modeling. Credits go to the Ehrenfest process.

![Alt text](experiments/media/essence.jpeg)

## Generated Samples from the Ehrenfest Process

![Alt text](experiments/media/samples.png)

![Alt text](experiments/media/samples.gif)