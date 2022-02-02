from gym.envs.registration import register

register(
    id='ElectricityMarket-v0',
    entry_point='gym_electricitymarket.envs:ElectricityMarket',
)

register(
    id='ElectricityMarketDiscrete-v0',
    entry_point='gym_electricitymarket.envs:ElectricityMarketDiscrete',
)

register(
    id='ElectricityMarketDiscreteDQN-v0',
    entry_point='gym_electricitymarket.envs:ElectricityMarketDiscreteDQN',
)