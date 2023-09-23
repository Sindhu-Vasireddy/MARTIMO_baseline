import argparse
import numpy as np
import tensorflow.compat.v1 as tf
import time
import pickle
import os

import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.keras.layers as layers
from stonesoup.plotter import Plotter, Dimension
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="mtt_scenario", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=100, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=10000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=25, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=128, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default="mtt_test_full", help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="C:\\Users\\reach\\OneDrive\\Documents\\maddpg_basic\\policy\\", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=10, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=500, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="C:\\Users\\reach\\OneDrive\\Documents\\maddpg_basic\\benchmark_files\\", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="C:\\Users\\reach\\OneDrive\\Documents\\maddpg_basic\\learning_curves\\", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = tf.layers.dense(out, units=num_units, activation=tf.nn.relu)
        out = tf.layers.dense(out, units=num_units, activation=tf.nn.relu)
        out = tf.layers.dense(out, units=num_outputs, activation=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.adv_policy=='ddpg')))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            local_q_func=(arglist.good_policy=='ddpg')))
    return trainers


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        trainers = get_trainers(env, num_adversaries, obs_shape_n, arglist)
        print('Using good policy {} and adv policy {}'.format(arglist.good_policy, arglist.adv_policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [0.0]  # sum of rewards for all agents
        agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
        final_ep_rewards = []  # sum of rewards for training curve
        final_ep_ag_rewards = []  # agent rewards for training curve
        loss_arr = []
        agent_info = [[[]]]  # placeholder for benchmarking info
        saver = tf.train.Saver()
        
        obs_n = env.reset()
        episode_step = 0
        train_step = 0
        t_start = time.time()

        print('Starting iterations...')
        while True:
            # get action
            action_n = [agent.action(obs) for agent, obs in zip(trainers,obs_n)]
            # environment step
            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            episode_step += 1
            done = all(done_n)
            terminal = (episode_step >= arglist.max_episode_len)
            # collect experience
            for i, agent in enumerate(trainers):
                agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
            obs_n = new_obs_n

            for i, rew in enumerate(rew_n):
                episode_rewards[-1] += rew
                agent_rewards[i][-1] += rew

            if done or terminal:
                # save values for plotting 
                for agent in env.agents:
                    sensor_position = list(agent.movement_controller.states[0].state_vector.flatten())
                    sensor_tracks = np.array([tuple(track.states[i].state_vector[[0, 2]].flatten().tolist())
                        for track in agent.tracks for i in range(arglist.max_episode_len+1)]).reshape(len(env.landmarks), arglist.max_episode_len+1, 2)
                    
                    sensor_truth = np.array([(landmark.gtpt.states[i].state_vector[0, 0], landmark.gtpt.states[i].state_vector[2, 0])
                        for landmark in env.landmarks for i in range(arglist.max_episode_len+1)]).reshape(len(env.landmarks), arglist.max_episode_len+1, 2)
                    
                    env.sensor_position.append(sensor_position)
                    env.sensor_tracks.append(sensor_tracks)

                    env.sensor_truth.append(sensor_truth)    
                           

                obs_n = env.reset()
                episode_step = 0
                episode_rewards.append(0)
                for a in agent_rewards:
                    a.append(0)
                agent_info.append([[]])



            # increment global step counter
            train_step += 1

            # for benchmarking learned policies
            if arglist.benchmark:
                for i, info in enumerate(info_n):
                    agent_info[-1][i].append(info_n['n'])
                if train_step > arglist.benchmark_iters and (done or terminal):
                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                    print('Finished benchmarking, now saving...')
                    with open(file_name, 'wb') as fp:
                        pickle.dump(agent_info[:-1], fp)
                    break
                continue

            # for displaying learned policies
            if arglist.display:
                time.sleep(0.1)
                env.render()
                continue

            # update all trainers, if not in display or benchmark mode
            loss = None
            for agent in trainers:
                agent.preupdate()
            for agent in trainers:
                loss = agent.update(trainers, train_step)
            if loss:
                loss_arr.append(loss)
                print("Loss value at ",train_step,"is",loss)
                

            # save model, display training output
            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                U.save_state(arglist.save_dir, saver=saver)
                # print statement depends on whether or not there are adversaries
                if num_adversaries == 0:
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                else:
                    print("steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]),
                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                t_start = time.time()
                # Keep track of final episode reward
                final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                print("final episode rewards",final_ep_rewards)
                for rew in agent_rewards:
                    final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

            # saves final episode reward for plotting training curve later
            if len(episode_rewards) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
               
                # Iterate over pairs of sensor tracks and truth
                for i in range(0, len(env.sensor_tracks), len(env.agents)):
                    fig, ax = plt.subplots()
                    
                    # Plot sensor B and sensor A data
                    # 100*(3*(2,51,2))=no_of_ep*(no_of_sensors*(no_of_targets,ep_len,x_y_coordinates))
                    for k in range(len(env.agents)):  # Iterate from 0 to len(env.agents)-1
                        sensor_tracks = np.array(env.sensor_tracks[i+k])
                        sensor_truth = np.array(env.sensor_truth[i+k])
                        color = ['blue', 'red', 'magenta', 'yellow','brown'][k]  # Choose color based on the current iteration

                        for j in range(np.array(env.sensor_tracks).shape[1]):
                            ax.plot(sensor_tracks[j, :, 0], sensor_tracks[j, :, 1], color=color, marker='o', label=f'Sensor Track {j+1}')
                            ax.plot(sensor_truth[j, :, 0], sensor_truth[j, :, 1], color="green", marker='+', label=f'Target Grnd Truth {j+1}')

                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    # Add a legend to the plot with 'bbox_to_anchor' to specify the legend's position
                    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))        

                    # Generate the sensor indices for the file name
                    sensor_indices = list(range(i, i+len(env.agents)))
                    
                    # Create the file name using the sensor indices
                    file_name = f"Sensor_plot_{'_'.join(map(str, sensor_indices))}_Time_{i/len(env.agents)}.png"
                    save_path = os.path.join(r"C:\Users\reach\OneDrive\Documents\stonesoup_maddpg_home\learning_curves\plots", file_name)
                    plt.savefig(save_path, bbox_inches='tight')
                    plt.close()

                loss_file_name = arglist.plots_dir + arglist.exp_name + '_loss.pkl'
                with open(loss_file_name, 'wb') as fp:
                    pickle.dump(loss_arr, fp)
                break

if __name__ == '__main__':
    arglist = parse_args()
    print("Starting training at",time.time())
    train(arglist)
    print("finished training at",time.time())

# Run this cmd:
# cd experiments 
# python train.py --scenario mtt_scenario --exp-name mtt_test_1 > output_log.txt 