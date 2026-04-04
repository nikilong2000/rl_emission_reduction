#!/bin/bash

# python 02_rl_control/models/train.py --algorithm ppo --agent_device cpu
python 02_rl_control/models/train.py --algorithm sac --agent_device cpu
python 02_rl_control/models/train.py --algorithm td3 --agent_device cpu
