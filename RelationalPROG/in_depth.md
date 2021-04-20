## Relational reasoning in deep learning: a parallel between solving visual and programming tasks

**Report on Bibliography**

**Overall Goals:**

- research possible ways to integrate relational reasoning in deep learning models
- investigate the role of relational reasoning in solving programming tasks

Based on our overall research goals, the current report presents a study of our selected bibliography by making a detailed pass through the material referenced in [study abstract](https://github.com/perticascatalin/Research/blob/master/RelationalPROG/exec_abstract.md).

### Content

[1. A simple Neural Network Model for Relational Reasoning](https://arxiv.org/pdf/1706.01427.pdf)

[2. Deep Coder: Learning to Write Programs](https://arxiv.org/pdf/1611.01989.pdf)

### 1. A simple Neural Network Model for Relational Reasoning

#### 1.1 High-Level Summary

This study presents a general machine learning model used for solving relational tasks. In a similar fashion to CNNs (used for extracting spatial and translation invariant properties) and LSTMs (used for extracting sequential dependencies), the RN (relational network) contains a built-in mechanism for capturing core common properties of relational reasoning.

The RN is used on 3 different tasks, all of which require relational reasoning on a set of objects:

- Visual question answering (datasets: CLEVR - 3d, Sort-of-CLEVR - 2d)
- Text-based question answering
- Complex reasoning about dynamic physical systems

Other similar approaches (relation-centric) include:

- Graph neural networks
- Gated graph sequential neural networks
- Interaction networks

However, the RN is a plug & play module, meaning that it can be placed on top of other deep learning models. Its equation is defined as follows:

![RN Formula](https://raw.githubusercontent.com/perticascatalin/Research/master/RelationalPROG/images/formula.png)

- O set of objects {o1, o2, ..., on}
- f overall network output (across all pairs of objects)
- g outputs a relation between 2 objects (same for all pairs)
- f, g can be MLPs
- RN end-to-end differentiable

Sample open source [code](https://github.com/clvrai/Relation-Network-Tensorflow) applying relational networks to the Sort-of-CLEVR dataset.

#### 1.2 Code Running Logs

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

This test run was performed on a dataset containing images with 4 objects. The images used have 128 x 128 pixels of various colors. The sizes of the images and the number of objects can be customized. The model's performance can be compared to baseline MLP/CNN models which do not use a relational function.

### 2. Deep Coder: Learning to Write Programs

#### 2.1 High-Level Summary