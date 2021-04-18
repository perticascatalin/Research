## Relational reasoning in deep learning: a parallel between solving visual and programming tasks

### 1. A simple Neural Network Model for Relational Reasoning

#### Summary

This study presents a general machine learning model which can be used for solving relational tasks. In a similar fashion to CNNs (used for extracting spatial and translation invariant properties) and LSTMs (used for extracting sequential dependencies), the RN (relational network) contains a built-in mechanism for capturing core common properties of relational reasoning.

The RN is used on 3 different tasks, all of which require relational reasoning on a set of objects:

- visual question answering (datasets: CLEVR - 3d, Sort-of-CLEVR - 2d)
- text-based question answering
- complex reasoning about dynamic physical systems

Other similar approaches (relation-centric) include:

- graph neural networks
- gated graph sequential neural networks
- interaction networks

However, the RN is a plug & play module, meaning that it can be placed on top of other deep learning models. Its equation is defined as follows:

![RN Formula](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/formula.png)

- O set of objects {o1, o2, ..., on}
- g outputs a relation
- f overall network output (across all pairs of objects)

Sample code for Sort-of-CLEVR, open source: https://github.com/clvrai/Relation-Network-Tensorflow

#### Code Running Logs

This repository is deprecated since Tensorflow2 (attempted to migrate to TF2, but got stuck at `contrib.layers.optimize_loss`, no equivalent in TF2).

Revert to TF1. This implies reverting to Python 3.6, was using Python 3.8. Got stuck with pyenv, cannot revert to 3.6.

Finally tried using:

- `sudo python -m pip install package` (for python 2.7)
- `sudo python3 -m pip install package` (for python 3.8)

Should use Python 2.7, with Tensorflow 1.15, got a working version this way. Plots visualized via Tensorflow-Plot.

- `python generator.py`
- `python trainer.py`
- `tensorboard --logdir ./train_dir`

Tensorboard then provides us with a localhost link and we can look at training statistics in the browser:

![Tensorboard](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/tensorboard.png)

After 15.000 training steps, we take a look at the currently tested images and we can see that 3 of 4 questions were answered correctly:

![Sample images](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/samples.png)

After another 15.000 training steps the accuracy on the testing data reaches an average of 95%, while the loss drops to around 0.1:

![Accuracy and Loss](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/loss_acc.png)

### 2. Deep Coder: Learning to Write Programs