# Genetic Programming Population Initialisation w/ DQN

This document briefly explains the mathematical formulation for this approach and discusses implementation related details.

A reason for the poor solution quality of standard Genetic Programming (GP) in the Lunar Lander Env could be because the population within GP is randomly initialised. Random initialisation introduces a significant amount of stochasticity in the outcomes. This results in some test runs being more successful just by random chance that good combinations of nodes are chosen before evolution takes place. The proposed improvement of this is to start with a fitter initial population of solutions such as described in [1] and [2]. This leads to the following hypothesis: 

\begin{quotation}
It is expected that a better-than-random initial population of solutions will improve much faster with GP as the selection and crossover steps would combine parent solutions that are already much fitter than before. This will also reduce some of the stochasticity in evolution as GP will not have to rely on chance to be able to select a fitter combination of nodes. 
\end{quotation}

To come up with a better-than-random initial population of solutions, it is proposed to use Deep Q-Networks [3] to learn a policy that can generate a tree with relatively high fitness. Consider a chain of two Markov decision processes (MDPs). The first MDP $`<S, A, R, T, $\gamma$>`$ pertains to the RL problem that returns neural-guided trees while the second MDP $`<S', A', R, T', $\gamma'$>`$ pertains to the actual lunar lander environment (here S, A, R, T, $\gamma$ refer to the discrete set of states, actions, reward function, transition probabilities, and discount factor).

The state set $S$ in the first MDP represents a set of symbolic regression multitrees represented as their pre-order traversal. $s_0$ (at time $t=0$) would be initialised to be a multitree of unfilled complete binary trees of a certain depth. The action set $A(s)$ would simply be the primitive set (accounting for arity and excluding operators that are mathematically infeasible). The idea is to have a policy network $\pi(a | s)$ (where $s,a \in S,A$), that returns action values which represent the addition of an operator to the tree state in consideration. Given an action $a = \pi(s)$, it is added to the empty spot in the tree's pre-order traversal and a new state is obtained as the updated symbolic regression multitree. The resulting tree is evaluated on the current state of the second MDP (lunar lander environment) to obtain action values for the four possible actions in the lunar lander environment. Softmax action selection is used on these action values to obtain an action to step through the second environment. For every one step of the first environment, the second environment goes through some $k$ steps. This ensures that the optimality of the current solutions are evaluated across a larger sample of the second environment's states and reduces any bias caused by happenstance. Hence, any statistical anomalies can be avoided.

The reward $R$ obtained in every step of the second MDP is associated with the specific action that is chosen and by extension, the specific subtree for that action. After $k$ steps in the second environment, the rewards are summed up and this reward signal is considered as the reward for the transition from the first MDP. This maintains sound learning since maximising the reward obtained in the second MDP still maximises the "goodness" of the policy in the first one (as the reward is simply a scalar). Finally, the transitions are appended to the replay buffer as usual and the policy network parameters are updated using standard gradient descent optimisation using the temporal difference (TD) loss. The figure below shows a diagram of the MDP chain. Some key details of this implementation are:

- Tree Structure: Subtrees within a multitree are considered to be unfilled complete binary trees of depth 10, i.e. all nodes except for the lowest child nodes have an arity of 2. This avoids restricting the scope of the expressions that can be generated. However, trees can be filled up regardless of the structure of the unfilled areas and can form any arbitrary expression.

- Tree Representation \& Generation: Subtrees are represented as a state by creating an array as per their pre-order traversals. During tree generation, new nodes are added to unfilled positions according to the pre-order traversal order. Note that pre-order traversals are not enough to uniquely represent a tree unless the arity of each node is known. The addition of this information enables a complete and unique reconstruction of a tree just from its pre-order traversal.

- Node Vectorization: Operators, features, and constants are vectorized as one-hot vectors with the last bit indicating the node's arity. Other schemes for vectorization such as word embeddings or GLOVE were considered but not opted for due to their complexity and since they do not consider mathematical operators. 

- Population Generation: A population of multitrees is generated with the learnt policies after training. The diversity of this population can be controlled by varying the $\epsilon$ parameter of $\epsilon$-greedy action selection. A higher value corresponds to a more diverse population.


![]()


[1] Mundhenk TN, Landajuela M, Glatt R, Santiago CP, Faissol DM, Petersen BK. Symbolic regression via neural-guided genetic programming population seeding. arXiv preprint arXiv:2111.00053. 2021 Oct 29.

[2] Petersen BK, Landajuela M, Mundhenk TN, Santiago CP, Kim SK, Kim JT. Deep symbolic regression: Recovering mathematical expressions from data via risk-seeking policy gradients. arXiv preprint arXiv:1912.04871. 2019 Dec 10.

[3] Mnih V, Kavukcuoglu K, Silver D, Graves A, Antonoglou I, Wierstra D, Riedmiller M. Playing atari with deep reinforcement learning. arXiv preprint arXiv:1312.5602. 2013 Dec 19.
