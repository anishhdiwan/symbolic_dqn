
def early_stopping(episode_return_ewma, threshold=100, tolerance=5):
	'''
	early_stopping stops training if the sum of epsiode returns is greater than some threshold for a tolerance number of episodes
	'''

	counter = 0

	if episode_return_ewma > threshold:
		counter += 1
		if counter >= tolerance:
			return True


def ewma(R, R_old, step, rho = 0.95):
	'''
	ewma calculates the exponentially weighted moving average of the input value and implements bias correction
	'''
	return (rho*R_old + (1-rho)*R)/(1 - rho**(step+1))

