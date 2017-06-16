import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

# Learning Rate
LR = 1e-3

# Environment
env = gym.make('CartPole-v0')

# Reset/Starts Environment
env.reset()

# 
goal_steps = 500

# Learn from games where we scored minimum 50 points
score_requirement = 50

# Games to play
initial_games = 10000

def some_random_games_first():
	# illustrate what random games look like
	for episode in range(5):
		env.reset()
		for t in range(goal_steps):
			# See whats happening in the game
			env.render()
			# Takes random action for you initially
			acton = env.action_space.sample()
			"""
				Observation: Data from game(pole position, kart position)
				Reward:      1 or 0 depending on if balanced
				Done: 		 Is the game over
				Info:		 Any other information
			"""
			observation, reward, done, info = env.step(action)
			if done:
				break

# some_random_games_first()

# Generates imperfect training samples
def initial_population():
	# Data were interested in training on for scores above 50
	training_data = []
	scores = []
	accepted = []
	for _ in range(initial_games):
		score = 0
		# Store all movements
		game_memory = []
		prev_observation = []
		# iterate through steps
		for _ in range(goal_steps):
			# Choose random action to perform
			action = random.randrange(0,2)
			"""
				Observation: Data from game(pole position, kart position)
				Reward:      1 or 0 depending on if balanced
				Done: 		 Is the game over
				Info:		 Any other information
			"""
			observation, reward, done, info = env.step(action)
			# Check if there is a previous observation to be made
			if len(prev_observation) > 0:
				game_memory.append([prev_observation, action])
			# Set previous observation to new observation
			prev_observation = observation
			# increment whatever reward we got
			score += reward
			if done:
				break
		# Check if game was successful
		if score >= score_requirement:
			accepted_scores.append(score)
			for data in game_memory:
				if data[1] == 1:
					output = [0, 1]
				elif data[1] == 0:
					output = [1, 0]
				training_data.append([data[0], output])
		env.reset()
		scores.append(score)
	training_data_save = np.array(training_data)
	np.save('saved.npy', training_data_save)
	print('Average Accepted Score: ', mean(accepted_scores))
	print('Median Accepted Score:  ', median(accepted_scores))
	print(Counter(accepted_scores))
	return training_data

initial_population()




































