import numpy as np
import math

"""Example 4.2 from Reinforcement Learning: An Introduction by Sutton and Barto

Example 4.2: Jack’s Car Rental Jack manages two locations for a nationwide car
rental company. Each day, some number of customers arrive at each location to rent cars.
If Jack has a car available, he rents it out and is credited $10 by the national company.
If he is out of cars at that location, then the business is lost. Cars become available for
renting the day after they are returned. To help ensure that cars are available where
they are needed, Jack can move them between the two locations overnight, at a cost of
$2 per car moved. We assume that the number of cars requested and returned at each
location are Poisson random variables, meaning that the probability that the number is
n is (lambda^n/n!)*e^-lambda, where lambda is the expected number. Suppose lambda is 3 and 4 for rental requests at
the first and second locations and 3 and 2 for returns. To simplify the problem slightly,
we assume that there can be no more than 20 cars at each location (any additional cars
are returned to the nationwide company, and thus disappear from the problem) and a
maximum of five cars can be moved from one location to the other in one night. We take
the discount rate to be  = 0.9 and formulate this as a continuing finite MDP, where
the time steps are days, the state is the number of cars at each location at the end of
the day, and the actions are the net numbers of cars moved between the two locations
overnight
"""

# Discount rate
gamma = 0.9

# Expected numbers for requests and returns at each location
lambda_request1 = 3
lambda_request2 = 4
lambda_return1 = 3
lambda_return2 = 2

# Convergence criterion for value iteration
theta = 1e-4

def poisson(lambda_,n):
    """Calculate the probability of n occurring in a poisson random variable
    with expected value lambda_
    """
    p = math.exp(-lambda_)*lambda_**n/math.factorial(n)
    return p

def state_transition(s,a,requests,returns):
    """The state transition function for the problem.  Given the current state
    (number of cars at each location), the action, and the number of cars
    requested and returned at each location, calculates the next state, and the
    reward (net pay)
    
    s: current state - a tuple where the zeroth element is number of cars at 
       location 1 and the first element is the number of cars at location 2
    a: action - an integer between -5 and 5 that specifies the number of cars
       transferred betweeen locations.  Negative a indicates cars are
       transferred from location 2 to location 1.  Positive a indicates cars
       are transerred from location 1 to location 2
    requests: a numpy array where the zeroth element is the number of requests
              received at location 1 and the first element is the number of 
              requests received at location 2
    returns: a numpy arrary where the zeroth element is the number of cars
             returned to location 1 and the first element is the number of cars
             returned to location 2
    """
    # First calculate the number of cars available to be rented by subtracting
    # a from the number of cars at location 1 and adding a to the number of 
    # cars at location 2.  This number must be between 0 and 20 for each 
    # location
    s_prime = np.clip(np.asarray(s) + np.asarray([-a, a]), np.zeros(2), 20*np.ones(2))
    # Calculate the number of cars rented from each location.  If the number of
    # cars requested is greater than the number of cars available, only the
    # avilable cars are rented.
    cars_rented1 = min([requests[0], s_prime[0]])
    cars_rented2 = min([requests[1], s_prime[1]])
    # Calculate reward.  $10 for each request that is filled, -$2 for each car
    # transferred.
    r = 10*(cars_rented1 + cars_rented2) - 2*abs(a)
    # Calculate the new number of cars at each location.  Available cars plus
    # cars returned minus cars rented.  Number is clipped between 0 and 20.
    s_prime = np.clip(s_prime + returns - np.asarray([cars_rented1, cars_rented2]), np.zeros(2), 20*np.ones(2))
    return tuple(s_prime), r

# Build list of all states
states = []
for i in range(21):
    for j in range(21):
        states.append((i,j))
# Initialize V(s) and A(s).
value = {}
policy = {}
for i in range(len(states)):
    value[states[i]] = 0
    policy[states[i]] = 0

# Boolean for ending the loop
stop = False
# Initialize iteration count
iteration = 0
while not stop:
    iteration += 1
    print('iteration {}'.format(iteration))
    # Policy Evaluation
    # Boolean for ending evaluation loop
    eval_stop = False
    # Initialize evaluation iteration count
    eval_iter = 0
    while not eval_stop:
        eval_iter += 1
        # Initialize Delta
        Delta = 0
        # Loop over all states
        for s in states:
            # Current value
            v = value[s]
            # Initialize sum of probabilities for s_prime and r
            sum_prob = 0
            # Create lists for possible values of requests and returns received
            # at each location
            possible_requests1 = list(range(s[0]+1))
            possible_requests2 = list(range(s[1]+1))
            possible_returns1 = list(range(20-s[0]+1))
            possible_returns2 = list(range(20-s[1]+1))
            # Loop over all combinations of requests and returns
            for (request1,request2,return1,return2) in zip(possible_requests1,possible_requests2,possible_returns1,possible_returns2):
                # Create requests and returns arrays for arguments for 
                # state_transition function
                requests = np.asarray([request1,request2])
                returns = np.asarray([return1,return2])
                # Calculate s_prime and r
                s_prime, r = state_transition(s,policy[s],requests,returns)
                # Calculate probability of getting this s_prime and this r by
                # multiplying probabilities for each request and return number
                prob = poisson(lambda_request1,request1)*poisson(lambda_request2,request2)*poisson(lambda_return1,return1)*poisson(lambda_return2,return2)
                # Add this probability times r + gamma*V(s_prime) to sum of
                # probabilities
                sum_prob += prob*(r + gamma*value[s_prime])
            # Update V(s)
            value[s] = sum_prob
            # Update Delta
            Delta = max([Delta, abs(v-value[s])])
        # Check stopping criterion
        if Delta < theta: eval_stop = True
    
    # Policy Improvement
    # Initialize check for policy not changing
    policy_stable = True
    # Loop over all states
    for s in states:
        # Current action
        old_action = policy[s]
        # Initialize list of values at this state for each action
        value_list = []
        # Create list of possible actions to take.  Can't transfer more cars
        # than are at either location
        possible_actions = list(range(-min([s[0],5]),min([s[1],5])+1))
        # Create lists for possible values of requests and returns received
        # at each location
        possible_requests1 = list(range(s[0]+1))
        possible_requests2 = list(range(s[1]+1))
        possible_returns1 = list(range(20-s[0]+1))
        possible_returns2 = list(range(20-s[1]+1))
        # Loop over all possible actions at this state
        for a in possible_actions:
            # Initialize sum of probabilities for s_prime and r
            sum_prob = 0
            # Loop over all combinations of requests and returns
            for (request1,request2,return1,return2) in zip(possible_requests1,possible_requests2,possible_returns1,possible_returns2):
                # Create requests and returns arrays for arguments for 
                # state_transition function
                requests = np.asarray([request1,request2])
                returns = np.asarray([return1,return2])
                # Calculate s_prime and r
                s_prime, r = state_transition(s,a,requests,returns)
                # Calculate probability of getting this s_prime and this r by
                # multiplying probabilities for each request and return number
                prob = poisson(lambda_request1,request1)*poisson(lambda_request2,request2)*poisson(lambda_return1,return1)*poisson(lambda_return2,return2)
                # Add this probability times r + gamma*V(s_prime) to sum of
                # probabilities
                sum_prob += prob*(r + gamma*value[s_prime])
            # Append this value for this action to the list of action values
            value_list.append(sum_prob)
        # Update A(s)
        policy[s] = possible_actions[np.argmax(value_list)]
        # If the updated action is different from the previous one, set 
        # policy_stable boolean to False
        if old_action != policy[s]: policy_stable = False
    # If the policy hasn't changed at all, end the main loop
    if policy_stable: stop = True