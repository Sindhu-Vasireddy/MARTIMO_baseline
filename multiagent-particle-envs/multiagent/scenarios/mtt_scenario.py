# Scenario for stone soup
import numpy as np
from multiagent.core import World, SensorAgent, Landmark
from multiagent.scenario import BaseScenario
from stonesoup.types.state import StateVector
from stonesoup.types.state import GaussianState 
from stonesoup.types.track import Track
from stonesoup.types.groundtruth import GroundTruthState,GroundTruthPath
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from datetime import datetime, timedelta

np.random.seed(1990)

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # rand_pos=np.array([[np.random.rand()], [np.random.rand()]])
        # print("Sensor ","position",rand_pos)
        # SensorInstance = SensorAgent(position_mapping=(0,2),ndim_state=4,position=np.array([[np.random.rand()], [np.random.rand()]]),rpm=60,
        # fov_angle=np.radians(45),dwell_centre=StateVector([0.0]),noise_covar=np.array([[np.radians(0.5) ** 2, 0],[0, 1 ** 2]]),
        # max_range=np.inf)
        for i, agent in enumerate(world.agents):
            agent.name = 'SensorAgent %d' % i
            agent.collide = False
            agent.silent = True
        # add landmarks
        world.landmarks = [Landmark() for i in range(3)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'TargetLandmark %d' % i
            landmark.collide = False
            landmark.movable = False
        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        world.agents = [SensorAgent(position_mapping=(0,2),ndim_state=4,position=np.array([[10*((-1)^i)+i], [0]]),rpm=60,
        fov_angle=np.radians(30),dwell_centre=StateVector([0.0]),noise_covar=np.array([[np.radians(0.5)**2, 0],[0, 1**2]]),
        max_range=np.inf) for i in range(2)]
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.25,0.25,0.25])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.75,0.75,0.75])
            landmark.model = CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),ConstantVelocity(0.005)])
        yps = range(0, 100, 10)  # y value for prior state
        xdirection = [1, -1.25, 1]
        ydirection = [1.25, -1, -1.25]
        
        world.predictor = KalmanPredictor(world.landmarks[0].model) # CombinedLinearGaussianTransitionModel
        world.updater = ExtendedKalmanUpdater(measurement_model=None) # #self.agents[0].measurement_model CartesiantoBearingRange- it gets it from the RotatingRadarBearingRange measurement_model!
        
        ntruths = len(world.landmarks)
        initial_meas =[]
        for i, landmark in enumerate(world.landmarks):
            landmark.gtst = GroundTruthPath([GroundTruthState([0, xdirection[i], yps[i], ydirection[i]], timestamp=world.start_time+ timedelta(seconds=world.dt) )],
                            id=f"id{i}")
            print("target",i,"ground",landmark.gtst)
            landmark.gtpt = GroundTruthPath([GroundTruthState([0, xdirection[i], yps[i], ydirection[i]], timestamp=world.start_time+ timedelta(seconds=world.dt))],
                            id=f"id{i}")
            landmark.state.p_pos = [landmark.gtst.state_vector[0],landmark.gtst.state_vector[2]]
            initial_meas.append(landmark.state.p_pos)
            landmark.state.p_vel = [landmark.gtst.state_vector[1],landmark.gtst.state_vector[3]]
            # alternate directions when initiating tracks
            # for j in range(0, ntruths):
            #     xdirection *= -1
            #     if j % 2 == 0:
            #         ydirection *= -1

        # set random initial states
        for i, agent in enumerate(world.agents):
            agent.state.p_pos = np.array([[10*((-1)^i)+i], [0]]) # same as in make world
            agent.state.p_vel = np.zeros(world.dim_p) # fixed sensor only dwell changes
            agent.tracks=[]
            agent.priors=[]
            agent.meas= initial_meas
            for j in range(0, ntruths):
                agent.priors.append(GaussianState([[0], [xdirection[j]], [yps[j]+0.1], [ydirection[j]]],
                                            np.diag([0.5, 0.5, 0.5, 0.5]+np.random.normal(0,5e-4,4)),
                                            timestamp=world.start_time+ timedelta(seconds=world.dt)))
                # xdirection *= -1
                # if j % 2 == 0:
                #     ydirection *= -1
                print("target",j,"prior",agent.priors[j])

            for j, prior in enumerate(agent.priors):
                agent.tracks.append(Track([prior]))
            
    def reward(self, agent, world):
        # agent reward should be updated in set actions
        return agent.reward

    def observation(self, agent, world):
        # target positions will be updated in set actions
        observation=np.array(agent.meas, dtype=np.float32)
        new_value = np.array([agent.dwell_centre], dtype=np.float32)
        observation = np.append(observation, new_value)
        print("observation np array of shape :",observation.shape," is:",observation)
        return observation.reshape(2*len(world.landmarks)+1,)