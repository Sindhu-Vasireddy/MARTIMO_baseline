import pickle
import argparse

import matplotlib.pyplot as plt

# experiment_name="mtt_test"
parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")    
parser.add_argument("--exp-name", type=str, default="mtt_test_full", help="name of the experiment")

arglist = parser.parse_args()

# Load object from a .pkl file
with open('C:\\Users\\reach\\OneDrive\\Documents\\maddpg_basic\\learning_curves\\'+arglist.exp_name+'_rewards.pkl', 'rb') as f:
    loaded_data = pickle.load(f)
# print("Individual rewards of "+experiment_name)
# Inspect the loaded object
# print(loaded_data)
# print("Aggregate rewards of "+experiment_name)
# Load object from a .pkl file
with open('C:\\Users\\reach\\OneDrive\\Documents\\maddpg_basic\\learning_curves\\'+arglist.exp_name+'_agrewards.pkl', 'rb') as f:
    loaded_data_ag = pickle.load(f)

# Inspect the loaded object
# print(loaded_data_ag)
with open('C:\\Users\\reach\\OneDrive\\Documents\\maddpg_basic\\learning_curves\\'+arglist.exp_name+'_loss.pkl', 'rb') as f:
    loaded_loss = pickle.load(f)

second_elements = [arr[1] for arr in loaded_loss]

# print(second_elements)
# Plot the training curve
plt.figure()
plt.plot(loaded_data, label='Episode Rewards')
# plt.plot(loaded_data_ag, label='Average Rewards')
plt.xlabel('Number of steps(in 100s)')
plt.ylabel('Reward')
plt.title('Training Curve')
plt.legend()
plt.show()

plt.figure()
plt.plot(second_elements, label='Loss')
plt.xlabel('Number of steps(in 100s)')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.show()

# Cmd to run the code: python ..\..\learning_curves\loadPickle.py 