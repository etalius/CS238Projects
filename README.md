## CS238 Projects

Projects for Decision Making Under Uncertainty (CS238) at Stanford University. 

### Project 1: Bayesian Network Structure Learning
Program to learn the structure of a bayesian network given a data set of variable names and values. Uses the k2 algorithm that is initialized to an emmpty graph and then adds parents to nodes that greedily maximize the Bayesian score, which is computed after each posssible addition. Nodes are iterated through in a topologically sorted order and initialized to a uniform prior. 

### Project 2: Reinforcement Learning
Program to develop a policy for a general data set of state, action, reward and new state tuples. 2 implementations are used, Q learning, and Q learning with eligibility traces. Both update a Q-table for state and action pairs using the bellmen update equation. Q learning with eligibaility traces propagates rewards backwards to previous states, and is ideal for worlds with sparse rewards, but does require more comptation. 
