## Relational reasoning in deep learning: a parallel between solving visual and programming tasks

### 1. A simple Neural Network Model for Relational Reasoning

Code (open source): https://github.com/clvrai/Relation-Network-Tensorflow

#### Code Running Logs

Deprecated since Tensorflow2 (attempted to migrate to TF2, but got stuck at contrib.layers.optimize_loss, no equivalent in TF2).

Revert to TF1, implies reverting to Python3.6, was using Python3.8. Got stuck with pyenv, cannot revert to 3.6.

Finally tried using:

`sudo python -m pip install package` (for python2.7)
`sudo python3 -m pip install package` (for python3.8)

Should use Python 2.7, with Tensorflow 1.15, got a working version this way. Plots visualized via Tensorflow-Plot

### 2. Deep Coder: Learning to Write Programs