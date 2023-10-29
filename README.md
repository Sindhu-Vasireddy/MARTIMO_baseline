# MARTIMO: Reinforcement Learning Illustrated using MADDPG

This repository implements reinforcement learning using a modified version of OpenAI's [Multi-agent Particle Environment (MPE)](https://github.com/openai/multiagent-particle-envs) An illustration is provided using MADDPG (Multi-Agent Deep Deterministic Policy Gradient) algorithm with sensor classes adapted from the latest [Stonesoup library.](https://stonesoup.readthedocs.io/en/latest/stonesoup.html). This example scenario is called mtt_scenario for initial baseline testing purposes, this can be extended for any custom scenario and additional algorithms which will have in-house support in upcoming future. PRs are welcome.

The framework is also shipped in a [docker image](https://hub.docker.com/repository/docker/sindhuvasireddy/maddpg-stonesoup-baseline/general) which is readily available to directly run the example scenario using 
```python Copy code 
docker pull sindhuvasireddy/maddpg-stonesoup-baseline:latest
```
and 
```python Copy code 
docker run sindhuvasireddy/maddpg-stonesoup-baseline:latest
```
If you would however prefer to run it with any different hyper-parameters then you could do them as well, 
```python Copy code 
docker run -it sindhuvasireddy/maddpg-stonesoup-baseline:latest python train.py --max-episode-len 10 --num-episodes 100 --batch-size 2 --save-rate 10
```

## Getting Started

To run this code locally, follow the steps below:

1. Clone this repository to your local machine.
2. Set up a virtual environment and activate it (recommended but optional).
3. Install the required packages from the `requirements.txt` file with `python==3.8.17`:

   ```bash Copy code
   pip install -r requirements.txt
   ```
Since the MADDPG and multiagent environment are used from local paths, you need to install them in editable mode. Assuming you cloned this repository in a directory named maddpg-implementation, run the following commands:

 ```bash Copy code
 pip install -e ./maddpg-implementation/maddpg
 pip install -e ./maddpg-implementation/multiagent-particle-envs 
 ```
Run the main script to start the reinforcement learning process. 
```python Copy code
python maddpg/experiments/train.py
```

## Project Structure
The project contains the following directories:

- learning_curves: Contains learning curve data as pickle files along with a python script for parsing.
- maddpg: Modified version of the MADDPG implementation, updated for the latest TensorFlow.
- multiagent-particle-envs: Modified multiagent particle environments with sensor classes adapted from the latest Stonesoup library.
policy: Policy networks.

## Contributing
If you want to contribute to this project, please follow the standard GitHub workflow:

- Fork the repository.
- Create a new branch for your feature/bugfix.
- Make your changes and commit them.
- Push the changes to your fork.
- Create a pull request to the main repository.
- Please ensure your code follows the project's coding conventions and includes appropriate unit tests.
