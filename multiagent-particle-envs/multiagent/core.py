import numpy as np
from stonesoup.sensor.radar.radar import RadarRotatingBearingRange
from stonesoup.models.transition.linear import CombinedLinearGaussianTransitionModel, ConstantVelocity
from stonesoup.types.groundtruth import GroundTruthState,GroundTruthPath
from stonesoup.predictor.kalman import KalmanPredictor
from stonesoup.updater.kalman import ExtendedKalmanUpdater
from datetime import datetime, timedelta

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None

class SensorState(EntityState):
    def __init__(self):
        # dwell centre 
        self.dwell = None

class SensorAction(object):
     def __init__(self):
        # dwell centre action 
         self.dc =0   

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# properties of landmark entities
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()
        self.gtst=None
        self.gtpt=None
        self.model= CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),ConstantVelocity(0.005)])
    
class SensorAgent(Entity,RadarRotatingBearingRange):
    def __init__(self, *args, **kwargs):
        RadarRotatingBearingRange.__init__(self, *args, **kwargs)
        # name 
        self.name = ''
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = True
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # mass
        self.initial_mass = 1.0
        # dwell centre noise amount
        self.dc_noise = None
        # dwell centre range if needed
        self.dc_range_low = 0.0
        self.dc_range_high = 2.0
        # state
        self.state = SensorState()
        # action
        self.action = SensorAction()
        #uncertainty in measurements
        self.uncertainty = []
        #meas
        self.meas=[]
        # script behavior to execute
        self.action_callback = None # self.sensor_actor # UseDwellActionGenerator here
        self.tracks=[]
        self.priors = []
        self.reward= 0   

# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        # dwell centre dimensionality
        self.dim_dc = 1
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 1
        self.start_time =datetime.now()
        self.predictor = KalmanPredictor(CombinedLinearGaussianTransitionModel([ConstantVelocity(0.005),ConstantVelocity(0.005)])) # CombinedLinearGaussianTransitionModel
        self.updater = ExtendedKalmanUpdater(measurement_model=None) # #self.agents[0].measurement_model CartesiantoBearingRange- it gets it from the RotatingRadarBearingRange measurement_model!        

    # return all entities in the world
    @property
    def entities(self):
        return self.landmarks #+self.agents agents are not entities

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # update state of the world
    def step(self):
        j=0
        for landmark in self.landmarks:
            landmark.gtst= GroundTruthPath([GroundTruthState(landmark.model.function(landmark.gtst, noise=True, time_interval=timedelta(seconds=1)),
                             timestamp=self.start_time + timedelta(seconds=self.dt))],
                            id=f"id{j}")
            landmark.gtpt.append(GroundTruthState(landmark.model.function(landmark.gtst, noise=True, time_interval=timedelta(seconds=1)),
                             timestamp=self.start_time + timedelta(seconds=self.dt)))
            j+=1
            landmark.state.p_pos = [landmark.gtst.state_vector[0],landmark.gtst.state_vector[2]]
            landmark.state.p_vel = [landmark.gtst.state_vector[1],landmark.gtst.state_vector[3]]
        self.dt+=1 # increment step