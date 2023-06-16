import gym
import copy

env = gym.make("LunarLander-v2", render_mode="rgb_array")

observation = env.reset()

copy_env = copy.deepcopy(env)

copy_env.lander = env.lander


observation, reward, terminated, truncated, info = env.step(0)
print(observation)

observation, reward, terminated, truncated, info = copy_env.step(0)
print(observation)



env.close()