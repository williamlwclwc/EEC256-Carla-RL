# SAC Baseline for EEC256 Project

## SAC in Carla simulator

Based on [PARL](https://github.com/PaddlePaddle/PARL) and PyTorch,
a parallel version of SAC was implemented and achieved high performance in the CARLA environment.

The smallest agents with only FC networks need approximately 4 hours to train. CNN version needs approximately 12 hours.

### Carla simulator introduction

We are using the latest version of CARLA. Please see [Carla simulator](https://github.com/carla-simulator/carla) to know more about Carla simulator.

+ Result was evaluated with mode `Lane`

## How to use

+ System: Ubuntu 16.04

### Dependencies

+ Simulator: [CARLA](https://github.com/carla-simulator/carla/releases/tag/0.9.6)
+ A open-ai gym wrapper: gym_carla

### Installation

1. Create conda environment

```bash
conda create -n rl_carla python=3.7
conda activate rl_carla
```

1. Download [CARLA_0.9.14](https://github.com/carla-simulator/carla/releases/tag/0.9.14),
   extract it to some folder, and add CARLA to `PYTHONPATH` environment variable

``` bash
export PYTHONPATH="SOMEFOLDER/CARLA_0.9.14/PythonAPI/carla/dist/carla-0.9.14-py3.7-linux-x86_64.egg:$PYTHONPATH"
```

2. Install the packages

``` bash
## install requirements，
## Install paddle or torch wrt which base you are using(paddle or torch)
## Make sure only one deep-learning framework exists in your conda_env during traing
pip install -r requirements.txt

## install gym env of carla
cd gym_carla
pip install -e .
```

or you can install the package that you need by `pip/conda install [package name]`

#### Training

Open another(new) terminal, enter the CARLA root folder and launch CARLA service. There are two modes to start the CARLA server:

(1) non-display mode

```bash
./CarlaUE4.sh -RenderOffScreen -carla-port=2021
```

(2) display mode

```bash
./CarlaUE4.sh -carla-port=2021
```

+ Can start multiple CARLA services (ports defined in env_config) for data collecting and training, one service (port: 2027) for evaluating.

For parallel training, we can execute the following [xparl](https://parl.readthedocs.io/en/stable/parallel_training/setup.html) command to start a PARL cluster：

```Parallelization
xparl start --port 8080
```

check xparl cluster status by `xparl status`

Start training

```bash
# by default train fc
python train.py --xparl_addr localhost:8080
# add --model to train cnn
python train.py --xparl_addr localhost:8080 --model cnn
```

#### Evaluate trained agent

Open another(new) terminal, enter the CARLA root folder and launch CARLA service with display mode.

```bash
./CarlaUE4.sh -carla-port=2029
```

Restore saved model to see performance.

``` bash
# demo using handcrafted + fc model
python evaluate.py --restore_model fc_model.ckpt
# demo using cnn + fc model
python evaluate.py --restore_model fc_model.ckpt --model cnn
```

### Reference

+ [SAC](https://arxiv.org/abs/1801.01290)
+ [RL_CARLA](https://github.com/ShuaibinLi/RL_CARLA)
