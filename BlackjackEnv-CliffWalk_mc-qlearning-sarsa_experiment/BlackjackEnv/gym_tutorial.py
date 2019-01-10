######
#	gym.openai.com/docs/
######

import gym

env = gym.make( 'CartPole-v0' )

for i_episode in range( 20 ):
	state = env.reset()

	for t in range( 1000 ):
		env.render()
		print( state )
		action = env.action_space.sample()
		state, reward, done, _ = env.step( action )

		if done:
			print('Episode #%d finished after %d timesteps' % (i_episode, t))
			break