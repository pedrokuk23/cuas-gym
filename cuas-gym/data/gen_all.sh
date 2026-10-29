#!/bin/bash
cd "$(dirname "$0")/.."
python3 -m cuas_gym.cli generate --split test_id --n 60 --out data/gym
python3 -m cuas_gym.cli generate --split test_terrain --n 60 --out data/gym
python3 -m cuas_gym.cli generate --split test_sensor --n 60 --out data/gym
python3 -m cuas_gym.cli generate --split test_scale --n 40 --out data/gym
python3 -m cuas_gym.cli generate --split train --n 300 --out data/gym
echo DONE
