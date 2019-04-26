# COMP 767 Winter 2019
## Final Project - F.A.M.E in Reinforcement Learning

### Team:
- Joshua Holla (260784204)
- Jonathan Maloney-Lebensold (260825605)

### Overview

We explore the application of Reinforcement Learning (RL) in a new frameworkwe call F.A.M.E.:  Federated Averaging with Mixed Environments.   We drawinspiration from previous work in distilling and transmitting learned policies andformulate an algorithm-agnostic approach based on Federated Learning. FederatedLearning is a distributed machine learning approach which uses data that remainsdistributed for the task of training a single gradient-based model. Extending RLwith Federated Learning could enable agents to benefit from learners in remote anddata-sensitive environments

### Dependencies

- Numpy
- OpenAI gym
- scipy
- fire
- PyTorch 1.x

### License
MIT

### Installation
```
pip install -r requirements.txt
```

### Reference Code

Our examples forked Open Source code from OpenAI Gym as well as a
re-implementation of Distral found here:

https://github.com/xrdt/Robust-Multitask-RL


### Running experiments

#### Gridworld experiments (5 seeds):
```
ENVN=GridworldEnv
COMET_DISABLE_AUTO_LOGGING=1
for SEED in {1..5}
do
	python main.py dqn --env-name=${ENVN} --num-rounds=10 --num-episodes=30 --fame-reg=True --distral=False --env-params=4,5,7 --seed=${SEED} --mp=True &&
	python main.py dqn --env-name=${ENVN} --num-rounds=300 --num-episodes=1 --fame-reg=False --distral=True --env-params=4,5,7 --seed=${SEED} --mp=False &&
	python main.py dqn --env-name=${ENVN} --num-rounds=1 --num-episodes=300 --fame-reg=False --distral=False --env-params=4,5,7 --seed=${SEED} --mp=False
done
```

#### Pendulum:
```
for SEED in {1..5}
do
	python main.py ddpg --env-name=GravityPendulum --num-rounds=1 --num-episodes=300 --fame-reg=False --env-params=8,10,12 --seed=${SEED} --mp=True &&
	python main.py ddpg --env-name=GravityPendulum --num-rounds=10 --num-episodes=30 --fame-reg=True --env-params=8,10,12 --seed=${SEED} --mp=True &&
	python main.py ddpg --env-name=GravityPendulum --num-rounds=1 --num-episodes=300 --fame-reg=False --env-params=7,10,13 --seed=${SEED} --mp=True &&
	python main.py ddpg --env-name=GravityPendulum --num-rounds=10 --num-episodes=30 --fame-reg=True --env-params=7,10,13 --seed=${SEED} --mp=True
done
```

#### Mountain Car:
```
COMET_DISABLE_AUTO_LOGGING=1
for SEED in {1..5}
do
	python main.py ddpg --env-name=MountainCarContinuous --num-rounds=1 --num-episodes=300 --fame-reg=True --env-params=0.001,0.0015,0.0012 --seed=${SEED} --mp=True &&
	python main.py ddpg --env-name=MountainCarContinuous --num-rounds=10 --num-episodes=30 --fame-reg=True --env-params=0.001,0.0015,0.0012 --seed=${SEED} --mp=True
done
```
