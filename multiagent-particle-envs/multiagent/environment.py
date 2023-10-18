import gym
from gym import spaces
from gym.envs.registration import EnvSpec
from stonesoup.functions import mod_bearing,pol2cart
from stonesoup.sensormanager.reward import UncertaintyRewardFunction
from stonesoup.sensor.action.dwell_action import DwellActionsGenerator 
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from stonesoup.hypothesiser.distance import DistanceHypothesiser
from stonesoup.measures import Mahalanobis
from stonesoup.dataassociator.neighbour import GNNWith2DAssignment
from ordered_set import OrderedSet
from stonesoup.types.track import Track
from stonesoup.models.measurement.nonlinear import CartesianToBearingRange
from stonesoup.types.array import StateVector
from scipy.optimize import linear_sum_assignment

import math
from datetime import timedelta
import numpy as np
from multiagent.multi_discrete import MultiDiscrete

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):

        self.world = world
        self.agents = self.world.policy_agents
        self.landmarks= self.world.entities
        # set required vectorized gym env property
        self.n = len(world.policy_agents)
        # scenario callbacks
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # environment parameters
        self.discrete_action_space = True
        # if true, action is a number 0...N, otherwise action is a one-hot N-dimensional vector
        self.discrete_action_input = False
        # if true, even the action is continuous, action will be performed discretely
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # if true, every agent has the same reward
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0
        self.sensor_position = []
        self.sensor_tracks = []
        self.sensor_truth = []

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # dwell centre action space
            u_action_space = spaces.Discrete(len(world.landmarks))
            if agent.movable:
                total_action_space.append(u_action_space)
            # total action space
            # print("total action space",total_action_space)
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # print("self.action_space",self.action_space)
        
        # set action for each agent
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        # all agents get total reward in cooperative case
        reward = np.sum(reward_n)
        if self.shared_reward:
            reward_n = [reward] * self.n

        return obs_n, reward_n, done_n, info_n

    def reset(self):
        # reset world
        self.reset_callback(self.world)
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # get observation for a particular agent
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)

    # get reward for a particular agent
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)
    
    # set env action for each agent
    def _set_action(self, action, agent, action_space, time=None):
        # process action
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            #print("this is the agent action",action)
            action = [action]
        agent.uncertainty = [] # reset for every step
        for i, target in enumerate(agent.tracks):
           act = np.argmax(action) 
           # Calculate the bearing of the chosen target from the sensor
           if i == act:
               x_target = target.state.state_vector[0]-agent.state.p_pos[0]
               y_target = target.state.state_vector[1]-agent.state.p_pos[1]
               bearing_target = mod_bearing(np.arctan2(y_target, x_target))
           agent.uncertainty.append(np.trace(target.covar))
        current_timestep = self.world.start_time + timedelta(seconds=self.world.dt)
        next_timestep = self.world.start_time + timedelta(seconds=self.world.dt+1)
        # Create action generator which contains possible actions
        action_generator = DwellActionsGenerator(agent,
                                               attribute='dwell_centre',
                                               start_time=current_timestep,
                                               end_time=next_timestep)
        # Action the environment's sensor to point towards the chosen target
        current_action = [action_generator.action_from_value(bearing_target[0])]
        config = ({agent: current_action})
        predictor = KalmanPredictor(self.landmarks[0].model) # CombinedLinearGaussianTransitionModel
        updater = ExtendedKalmanUpdater(measurement_model=agent.measurement_model) # #self.agents[0].measurement_model CartesiantoBearingRange- it gets it from the RotatingRadarBearingRange measurement_model!
        reward_function = UncertaintyRewardFunction(predictor=predictor, updater=updater)
        agent.reward += reward_function(config, agent.tracks, next_timestep)
        agent.add_actions(current_action) # update dwell centre action in stone soup
        agent.act(next_timestep) # move the sensor to new position
        hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=3)
        data_associator = GNNWith2DAssignment(hypothesiser)
        # Calculate a measurement from the sensor
        measurement = set()
        # detections
        measurement |= agent.measure(OrderedSet(landmark.gtst for landmark in self.landmarks), noise=True)

 
        last_items = [Track(states=[track.states[-1]], id=track.id, init_metadata=track.init_metadata) for track in agent.tracks]

        hypotheses = data_associator.associate(last_items,
                                               measurement,
                                               current_timestep)
    
 
        agent.meas=[]#reset observation every step

        for track in agent.tracks:
            for key in hypotheses.keys():
                if key.id == track.id:
                    hypothesis = hypotheses[key]
                    if hypothesis.measurement:
                        post = updater.update(hypothesis)
                        track.append(post)
                        agent.meas.append([post.state_vector[0],post.state_vector[2]])

                    else:  # When data associator says no detections are good enough, we'll keep the prediction
                        track.append(hypothesis.prediction)
                        agent.meas.append([hypothesis.prediction.state_vector[0],hypothesis.prediction.state_vector[2]])
