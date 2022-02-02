import gym
from gym import error, spaces, utils
from gym.utils import seeding

import numpy as np
import os

wholesale_price = [2,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,6,6,6,4,3,3,2,2]
elasticity = [-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.3,-0.5,-0.5,-0.5,-0.5,-0.7,-0.7,-0.7,-0.7,-0.7,-0.5,-0.5,-0.5]
# critical load for 3 customers
crit_demand = [[4,4,4,4,4,4,4,4,4,4,4,4,4,4.5,6,7,8,9,9,9,8,7,6,5],[4,4,4,4,4,4,4,4,4,4,4,4,4,4.5,6,7,9,10,10,10,9,7,6,5],[4,4,4,4,4,4,4,4,4,4,4,4,4,5,6,7,9,9.5,10,10,9,7,6,5]]
# curtailable load for 3 customers
curt_demand = [[5,5,5,5,5,5,5,5,5,5,5,5,5,6,7,8,9,10,10,10,9,8,7,6],[5,5,5,5,5,5,5,5,5,5,5,5,5,6,7,8,10,11,11,11,10,8,7,6],[5,5,5,5,5,5,5,5,5,5,5,5,5,6,7,8,10,11,11.5,11,10,8,7,6]]
num_customers = 3
# coefficients related to dissatisfaction costs
alpha_n = [0.8,0.5,0.3]
beta_n = [0.1,0.1,0.1]
rho = 0.8

class ElectricityMarketDiscrete(gym.Env):

	def __init__(self):
		self.wholesale_price = wholesale_price
		self.elasticity = elasticity
		self.crit_demand = crit_demand
		self.curt_demand = curt_demand
		self.num_customers = num_customers
		self.alpha_n = alpha_n
		self.beta_n = beta_n
		self.rho = rho
		self.timer = 0
		# action space is the retail prices for customers
		# can use Discrete (i.e., discretize retail price)
		self.action_space = spaces.MultiDiscrete([21,21,21])
		# observation space is the energy demand, the actual energy consumption of customers
		# can use Discrete (i.e., discretize consumed energy of curtailable load)
		self.observation_space = spaces.Box(low=0, high=12,shape=(2*num_customers, ))
		self.start = np.zeros(2*self.num_customers)
		self.state = self.start
		self.done = False

	def step(self, action):
		assert self.action_space.contains(action)
		new_state = self.take_action(action)
		# reward is defined as total net profit
		# for each customer, the profit is (retail price - wholesale price) * consumed energy
		reward = 0
		for c in range(self.num_customers):
			retail_price_c = action[c]*0.5
			curt_demand_c = self.curt_demand[c][self.timer]
			curt_consumption_c = curt_demand_c * (1 + self.elasticity[self.timer]*(retail_price_c - self.wholesale_price[self.timer])/self.wholesale_price[self.timer])
			dissatisfaction_cost_c = 0.5*self.alpha_n[c]*(curt_demand_c - curt_consumption_c)**2 + self.beta_n[c]*(curt_demand_c - curt_consumption_c)
			total_consumption_c = new_state[2*c+1] 
			reward += self.rho*(retail_price_c - self.wholesale_price[self.timer])*total_consumption_c
			reward -= (1-self.rho)*(retail_price_c*total_consumption_c + dissatisfaction_cost_c)
		self.state = new_state
		self.timer += 1
		# simulate 24 hrs
		if self.timer == 24:
			self.done = True
		info = dict()
		return self.state, reward, self.done, info

	def reset(self):
		self.done = False
		self.state = self.start
		self.timer = 0
		return self.state

	def take_action(self, action):
		new_state = np.zeros(2*self.num_customers)
		for c in range(self.num_customers):
			curt_demand_c = self.curt_demand[c][self.timer]
			crit_demand_c = self.crit_demand[c][self.timer]
			retail_price_c = action[c]*0.5
			curt_consumption_c = curt_demand_c * (1 + self.elasticity[self.timer]*(retail_price_c - self.wholesale_price[self.timer])/self.wholesale_price[self.timer])
			crit_consumption_c = crit_demand_c
			total_demand_c = curt_demand_c + crit_demand_c
			total_consumption_c = curt_consumption_c + crit_consumption_c
			new_state[2*c] = total_demand_c
			new_state[2*c+1] = total_consumption_c
		return new_state



