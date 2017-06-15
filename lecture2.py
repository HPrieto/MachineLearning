"""
	Supervised: 
		* requires dataset
		* know input/outputs
		* provide to machine learning algorithm

	Unsupervised: 
		* Know nothing of outputs
		* all we get is data
		* find underlying structure/data

	Semi-Supervised:
		* Only parts are labeled

	Reinforcement:
		* Agent has to exist in the world
		* Knows the inputs
		* Knows nothing/very little of the world
		* Goes through the world based on rewards

	Perceptron:
		* Output is binary
		* Each input has weight on them

	A  |  B  |  Output
	1  |  0  |    1
	0  |  1  |    1
	0  |  0  |    1
	1  |  1  |    0

	Feed Forward Neural Network:
		Input to output with no loops

	Recurrent Neural Network:
		Memory of state
		Remembers data that went through
		very hard to train
"""

"""
	Feed Forward Neural Net

	Input Layer:

	Hidden Layer:
		* Has no interaction with input or output
		* Simple a block used to compute neural network
		* Formats data from inputs to outputs

	Output Layer:
		
	1.) Whenever an input is a 5, the output neuron in charge of 5 gets excited
			and outputs a value that is closer to 1 using sigmoid function,
			while the others (hopefully) output a value that is closer to 0.
			* When they dont, we adjust the weights in such a way that they get closer to 0 or 1.
	2.) 
"""

"""
	Agent and Environment
	* At each step the agent:
		- Executes action
		- Receives observation(new state)
		- Receives reward

	* The environment:
		- Receives action
		- Emits observation(new state)
		- Emits reward
"""

"""
	Markov Decision Process:
		- Has NO memory
		- All we have is a state that we are currently in
		- We perform an action and get a reward
		- Then find ourselves in a new state
		- Repeats over and over

		- When we are in a certain state, we have no memory of what
			happened two states ago.
		- Everthing is operating on the instantaneous

	Major Components of a Reinforcement Learning Agent:
		- There is a policy
		- The function of an agent's behavior
			(For any given state, what is the action that I will take with some probability?)
		- Value function is how good or bad a function is at any particular state
		- There is a model(How we represent the environment)

	Deterministic: 
		- When you go up, you go up.

	Stochastic:
		- When you go up, you dont always go up.

	Q-Learning:
		- An off-policy approach
		- The policy is estimated as we go along
		- Represented as Q-function

	Q-Function
		- Input is a state at time t
			and an action that we choose to take at that state 
		- Goal of each state is to choose an action that maximizes the reward.
		* Able to approximate through experience the optimal Q-Function,
			the optimal function that tells us how to act in any state of the world

	* Keep playing the game, for every state, fail and win then see what is good for us
		and keep doing that.

	Epsilon Greedy Policy
		* With probability of 1-epsilon perform the optimal/greedy action
		* With probability of epsilon perform a random action

	SUDO CODE OF EQUATION:

		- initialize Q[num_states,num_actions] arbitrarily(random with no reason)
		- observe initial state s
		- repeat
			* select and carry out an action a
			* observe reward r and new state s
			* Q[s,a] = Q[s,a] + @(r + y max^a Q[s',a'] - Q[s,a])
			* s = s'
		- until terminated

	Reinforcement Learning:
		- When we start playing/living/doing whatever it is we are doing,
			we have no preconceived notion of what is good or bad, its random
		- The fact that we learn anything is amazing
"""

"""
	Deep Learning:
	- Instead of modeling the world in a QTable
	- Try to learn the function

	Hope for Supervised Learning:
	- Good at memorizing data

	Hope for Reinforcement Learning(QLearning):
	- We can extend the occasional rewards we get to generalize over the operation
		and the actions you take in that world leading up to the rewards

	Hope for Deep Learning:
	- Taking the Reinforcement Learning and taking it into a world that can
		be difined arbitrarily.
		* But still needs a formalized definition of the world.

	Deep Q-Network:
		- Nothing is known at the beginning of the game, the network learns after 
			continuously playing the game and learning the reward system.

"""

"""
	Deep Q-Learning Algorithm Pseudo Code:

	initialize replay memory D
	initialize action-value function Q with random weights
	observe initial state s
	repeat
		select an action a
			with probability 'E'(Epsilon Greedy Policy) select a random action
			otherwise select a = argmax a'Q(s,a')
		carry out action a
		observe reward r and new state s'
		store experience <s,a,r,s'> in replay memory D

		sample random transitions <ss,aa,rr,ss'> from replay memory D
		calculate target each minibatch transition
			if ss' is terminal state then tt = rr
			otherwise tt = rr + ymax a'Q(ss',aa')
		train the Q network using (tt - Q(ss,aa))^2 as loss

		s = s'
	until terminated
"""
	


































