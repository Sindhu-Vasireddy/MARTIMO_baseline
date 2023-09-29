# Reinforcement Learning using MADDPG

This repository implements reinforcement learning using a modified version of OpenAI's MADDPG (Multi-Agent Deep Deterministic Policy Gradient) algorithm. The MADDPG algorithm is applied to a multi-agent environment using a modified version of the OpenAI multiagent particle environments, with sensor classes adapted from the latest Stonesoup library.

## Getting Started

To run this code locally, follow the steps below:

1. Clone this repository to your local machine.
2. Set up a virtual environment and activate it (recommended but optional).
3. Install the required packages from the `requirements.txt` file with `python==3.8.17`:

   ```bash Copy code
   pip install -r requirements.txt
   ```
Since the MADDPG and multiagent environment are used from local paths, you need to install them in editable mode, run the following commands from the root directory:

 ```bash Copy code
 pip install -e ./maddpg
 pip install -e ./multiagent-particle-envs 
 ```
Run the main train script to start the reinforcement learning process with the default parameters.

 ```bash Copy code
 cd maddpg/experiments
 python train.py
 ```

## Project Structure
The project contains the following directories:

- learning_curves: Contains learning curve data as pickle files along with a python script for parsing.
- maddpg: Modified version of the MADDPG implementation, updated for the latest TensorFlow.
- multiagent-particle-envs: Modified multiagent particle environments with sensor classes adapted from the latest Stonesoup library.
policy: Policy networks.
- TensorflowUpatedMaddpg: MADDPG implementation using TensorFlow 2.0 (if needed separately).

## Contributing
If you want to contribute to this project, please follow the standard GitHub workflow:

- Fork the repository.
- Create a new branch for your feature/bugfix.
- Make your changes and commit them.
- Push the changes to your fork.
- Create a pull request to the main repository.
- Please ensure your code follows the project's coding conventions and includes appropriate unit tests.