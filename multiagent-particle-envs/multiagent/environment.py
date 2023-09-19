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
# from stonesoup.plotter import Plotterly
from datetime import timedelta
import numpy as np
from multiagent.multi_discrete import MultiDiscrete

# environment for all agents in the multiagent world
# currently code assumes that no agents will be created/destroyed at runtime!
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

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
        # self.plotterB = {"sensor":[],"track":[],"groundT":[]}
        self.sensor_position = []
        self.sensor_tracks = []
        self.sensor_truth = []

        # configure spaces
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # dwell centre action space
            u_action_space = spaces.Discrete(world.dim_dc * 2 + 1)
            if agent.movable:
                total_action_space.append(u_action_space)
            # total action space
            if len(total_action_space) > 1:
                # all action spaces are discrete, so simplify to MultiDiscrete action space
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                    # print("i was heree")
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # print("Env Action space is this",self.action_space)
            # observation space
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))

        # rendering
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def step(self, action_n):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        
        # set action for each agent
        for i, agent in enumerate(self.agents):
            message = "------Setting action for agent_" + str(i+1) + " of " + str(len(self.agents)) + "------"
            print(message.center(180))
            self._set_action(action_n[i], agent, self.action_space[i])
        print("----------------------------------------------------------------------Action Set for both agents----------------------------------------------------------------------")
        # advance world state
        self.world.step()
        # record observation for each agent
        for agent in self.agents:
            print("measurement after action",agent.meas)
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
        # reset renderer
        self._reset_render()
        # record observations for each agent
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        # print("Reseting Obs",obs_n)
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
    
    # set env action for a particular agent
    def _set_action(self, action, agent, action_space, time=None):
        # print("action init",action)
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
            action = [action]
        agent.uncertainty = [] # reset for every step
        for i, target in enumerate(agent.tracks):
           act = np.argmax(action)#    act=1
           # Calculate the bearing of the chosen target from the sensor
           if i == act:
               x_target = target.state.state_vector[0]-agent.state.p_pos[0]
               y_target = target.state.state_vector[1]-agent.state.p_pos[1]
               print("length of tracks",len(agent.tracks))
                #print("action value and i value",act,"---",i)
               print("Target Position: (",target.state.state_vector[0],",",target.state.state_vector[2],")")
               #print("x_target, y_target",x_target,y_target)
               bearing_target = mod_bearing(np.arctan2(y_target, x_target))
           agent.uncertainty.append(np.trace(target.covar))
        # print("observation uncertainty",agent.uncertainty)
        # print("``````````````````````````````````````````````````````````````````````````````````````````````````````````agent before action:",agent)
        current_timestep = self.world.start_time + timedelta(seconds=self.world.dt)
        next_timestep = self.world.start_time + timedelta(seconds=self.world.dt+1)
        # Create action generator which contains possible actions
        action_generator = DwellActionsGenerator(agent,
                                               attribute='dwell_centre',
                                               start_time=current_timestep,
                                               end_time=next_timestep)
        # print("bearing_target",bearing_target[0])
        # Action the environment's sensor to point towards the chosen target
        current_action = [action_generator.action_from_value(bearing_target[0])]
        # print("current_action:",current_action)
        config = ({agent: current_action})
        predictor = KalmanPredictor(self.landmarks[0].model) # CombinedLinearGaussianTransitionModel
        updater = ExtendedKalmanUpdater(measurement_model=agent.measurement_model) # #self.agents[0].measurement_model CartesiantoBearingRange- it gets it from the RotatingRadarBearingRange measurement_model!
        reward_function = UncertaintyRewardFunction(predictor=predictor, updater=updater)
        agent.reward += reward_function(config, agent.tracks, next_timestep)
        agent.add_actions(current_action) # update dwell centre action in stone soup
        agent.act(next_timestep) # move the sensor to new position
        # print("````````````````````````````````````````````````````````````````````````````````````````````````````````````agent after action:",agent)
        hypothesiser = DistanceHypothesiser(predictor, updater, measure=Mahalanobis(), missed_distance=3)
        data_associator = GNNWith2DAssignment(hypothesiser)
        # print("target gt state Measurement",OrderedSet(landmark.gtst for landmark in self.landmarks))
        # Calculate a measurement from the sensor
        measurement = set()
        # detections
        measurement |= agent.measure(OrderedSet(landmark.gtst for landmark in self.landmarks), noise=True)
        detectable_ground_truths = [truth for truth in (OrderedSet(landmark.gtst for landmark in self.landmarks))
                                    if agent.is_detectable(truth)]
        print("agent_detectable ground truths",len(detectable_ground_truths))
        result = ",".join(str(m.state_vector) for m in measurement)
        print("agent measurement", result.replace("\n", ""))
        antenna_heading = agent.orientation[2, 0] + agent.dwell_centre[0, 0]
        rot_offset = StateVector(
                [[agent.orientation[0, 0]],
                 [agent.orientation[1, 0]],
                 [antenna_heading]])  
        val = []
        acr = CartesianToBearingRange(
        ndim_state=agent.ndim_state,
        mapping=agent.position_mapping,
        noise_covar=agent.noise_covar,
        translation_offset=agent.position,
        rotation_offset=rot_offset)
        # print(type(agent))
        for m in measurement:
            # print("detection",m)
            val.append(acr.inverse_function(detection=m))
        print("value", ",".join(str([vector[0],vector[2] ]) for vector in val))
        value =[[vector[0],vector[2] ] for vector in val]
        # print("agent changed action",agent.dwell_centre)
        # print("Sensor Measurement Length:",len(measurement))
        # for measurement_item in measurement:
        #     x = measurement_item.state_vector[1] * math.cos(measurement_item.state_vector[0])
        #     y = measurement_item.state_vector[1] * math.sin(measurement_item.state_vector[0]) 
        #     val = pol2cart(measurement_item.state_vector[1], measurement_item.state_vector[0])   
        #     print("stonesoup x,y",val)  
        #     print("Measured x and y:",x,y)
        print("ground truth values:", ",".join(str(val.state_vector[[0, 2]]) for gtp in OrderedSet(landmark.gtst for landmark in self.landmarks) for val in gtp.states).replace("\n", ""))
        ground = [val.state_vector[[0, 2]] for gtp in OrderedSet(landmark.gtst for landmark in self.landmarks) for val in gtp.states]
        paired_values = []
        errors = []

        # Calculate the Euclidean distances between each value point and ground point
        distances = np.zeros((len(value), len(ground)))
        for i, v in enumerate(value):
            for j, g in enumerate(ground):
                distances[i, j] = np.linalg.norm(np.array(v) - np.array(g))

        # Solve the assignment problem using the Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(distances)

        # Match the value points with their corresponding ground points
        paired_values = [(value[i], ground[j]) for i, j in zip(row_indices, col_indices)]
        errors = [distances[i, j] for i, j in zip(row_indices, col_indices)]

        print("Paired Values:", str(paired_values).replace("\n", ""))
        print("Total errors:", errors)

        last_items = [Track(states=[track.states[-1]], id=track.id, init_metadata=track.init_metadata) for track in agent.tracks]
        # print("ufuygihiu..........................................................................................................",agent.tracks)
        # print("bs...{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}",last_items)
        # for tack in agent.tracks:
        #     if len(tack)<3:
        #         print("actual track",tack)
        # print("Last items:", last_items)
        hypotheses = data_associator.associate(last_items,
                                               measurement,
                                               current_timestep)
        print("agent to check is",str(agent).replace("\n", ""))
        print("Last Track to check is",str(last_items).replace("\n", ""))
        print("predictor to check is",str(predictor).replace("\n", ""))
        print("prediction is",str(predictor.predict(last_items[0],current_timestep).state_vector).replace("\n", ""))
        # print("measurements 1",[m.state_vector for m in measurement][0])
        # print("measurements 2",[m.state_vector for m in measurement][1])
        # print("measurements 3",[m.state_vector for m in measurement][2])
        # for h in hypotheses.values(): 
        #     if h.measurement is not None:
        #         print("detection of association",h.measurement.state_vector)
        #     if h.measurement_prediction is not None:
        #         print("prediction of association",h.measurement_prediction.state_vector)

        # # Extracting state vectors from your data
        # measurement_states = [m.state_vector for m in measurement]
        # detection_states = [h.measurement.state_vector for h in hypotheses.values() if h.measurement is not None]
        # prediction_states = [h.measurement_prediction.state_vector for h in hypotheses.values() if h.measurement_prediction is not None]

        # # Calculating differences
        # detection_measurement_diffs = []
        # for d, m in zip(detection_states, measurement_states):
        #     if d is not None and m is not None:
        #         diff = np.array(d) - np.array(m)
        #         detection_measurement_diffs.append(diff)

        # prediction_measurement_diffs = []
        # for p, m in zip(prediction_states, measurement_states):
        #     if p is not None and m is not None:
        #         diff = np.array(p) - np.array(m)
        #         prediction_measurement_diffs.append(diff)

        # # Printing the differences
        # for diff in detection_measurement_diffs:
        #     print("Detection - Measurement Difference:", diff)

        # for diff in prediction_measurement_diffs:
        #     print("Prediction - Measurement Difference:", diff)

        # tmp=agent.meas
        # print(tmp)
        agent.meas=[]#reset observation every step
        # for key in hypotheses.keys():
        #     print("key is ",key)
        #     print(key.id)
        #     print(key.init_metadata)
        for track in agent.tracks:
            # last_item = Track(states=[track.states[-1]], id=track.id, init_metadata=track.init_metadata)
            for key in hypotheses.keys():
                if key.id == track.id:
                    hypothesis = hypotheses[key]
                    if hypothesis.measurement:
                        print("This is hypotheses meas",str(hypothesis.measurement.state_vector).replace("\n", ""),end=" ")
                        print("This is hypotheses pred",str(hypothesis.prediction.state_vector).replace("\n", ""))
                        print("This is last track pred",track.states[0].state_vector[0],track.states[0].state_vector[2])
                        print("corresponding target position",str([acr.inverse_function(detection=hypothesis.measurement)[0],acr.inverse_function(detection=hypothesis.measurement)[2]]).replace("\n", ""))
                        post = updater.update(hypothesis)
                        track.append(post)
                        print("length of track is when hypothesis is accepted",len(track))
                        print("This is added to track WITH_UPDATE(",post.state_vector[0],",",post.state_vector[2],")")
                        agent.meas.append([post.state_vector[0],post.state_vector[2]])
                    else:  # When data associator says no detections are good enough, we'll keep the prediction
                        print("This is hypotheses",str(hypothesis).replace("\n", ""),end=" ")
                        print("This is hypotheses pred",str(hypothesis.prediction.state_vector).replace("\n", ""),end=" ")
                        print("corresponding target position",str([acr.inverse_function(detection=hypothesis.prediction)[0],acr.inverse_function(detection=hypothesis.prediction)[2]]).replace("\n", ""))
                        track.append(hypothesis.prediction)
                        # track.append(Track(states=[track.states[-1]], id=track.id, init_metadata=track.init_metadata) for track in agent.tracks)
                        print("length of track is when hypothesis is rejected",len(track))
                        print("This is added to track NO_UPDATE(",hypothesis.prediction.state_vector[0],",",hypothesis.prediction.state_vector[2],")")
                        print("This is last track pred",track.states[0].state_vector[0],track.states[0].state_vector[2])
                        agent.meas.append([hypothesis.prediction.state_vector[0],hypothesis.prediction.state_vector[2]])
                        # agent.meas.append([last_items[0].states.state_vector[0],last_items[0].states.state_vector[2]])
        print("final agent measurement",str(agent.meas).replace("\n", ""))
        cov=[]
        # print("observation calculated",agent.meas)
        for i, target in enumerate(agent.tracks):
            cov.append(np.trace(target.covar))
        # print("This is the next step track covariance",cov)
        # Plot ground truths, tracks and uncertainty ellipses for each target.

    # reset rendering assets
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # render environment
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    word = '_'
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            print(message)

        for i in range(len(self.viewers)):
            # create viewers (if necessary)
            if self.viewers[i] is None:
                # import rendering only if we need it (and don't import for headless machines)
                #from gym.envs.classic_control import rendering
                from multiagent import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # create rendering geometry
        if self.render_geoms is None:
            # import rendering only if we need it (and don't import for headless machines)
            #from gym.envs.classic_control import rendering
            from multiagent import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # add geoms to viewer
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from multiagent import rendering
            # update bounds to center around agent
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # update geometry positions
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # render to display or array
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        return results

    # create receptor field locations in local coordinate frame
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # circular receptive field
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # add origin
            dx.append(np.array([0.0, 0.0]))
        # grid receptive field
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx


# vectorized wrapper for a batch of multi-agent environments
# assumes all environments have the same observation and action space
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # render environment
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
