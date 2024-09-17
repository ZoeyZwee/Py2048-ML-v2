# Py2048 ML v2

Version 1 of this project attempted to train an agent to play 2048 using [Deep Mind's DQN Algorithm](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf).

Version 2 continues the trend of exploring Deep Mind's latest work, this time training a 2048 agent with [AlphaZero](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf). 
Specifically, this project aims to implement the scheme outlined in the [Stochastic MuZero Paper](https://openreview.net/forum?id=X6D9bAHhBQ1). While the focus of the paper is the novel Stochastic MuZero algorithm, it also includes a scheme for training an AlphaZero agent in 2048
, which is then used to benchmark Stochastic MuZero. It is this AlphaZero scheme which I implement in this project.

## Key Ideas

### Monte Carlo Tree Search
MCTS is an algorithm for value estimate and policy generation. It uses the current network's policy and value estimates to produce an improved policy and value estimate for the agent's current state. This project uses a specific variant of MCTS called[Upper Confidence Bounds Applied to Trees](https://link.springer.com/chapter/10.1007/11871842_29), aka UCT. For an in-depth explanation, check out my 
blog post [MCTS, (Deep Mind Edition)](zoeyzwee.github.io/blog/MCTS).

### Actor-Learner Collaboration
AlphaZero divides training loop into 2 jobs: acting and learning. 
- Actors play out games, selecting moves with MCTS and the current iteration of the network. Each time a move is selected, the value and policy estimates returned from MCTS are used to create a learning target, which is sent to a replay buffer. 
- Learners then read from this buffer and train the value and policy outputs of the network toward the learning targets.

### Tips and Tricks for Stabilizing an Unruly Network
Neural networks are incredibly finicky. It turns out that a great deal of care is required when choosing representations for states, actions, and network outputs. If the wrong representations are chosen, the network weights will blow up to infinity, and the learning process will fail. Here are a few of the techniques used by Deep Mind in the MuZero paper.

#### Network Structure 
AlphaZero uses a single deep neural network following the [ResNet v2 architecture](https://arxiv.org/abs/1603.05027) with [Layer Normalization](https://arxiv.org/abs/1607.06450). Both layer normalization and the resnet architecture work to keep the magnitude of layer activations consistent (and bounded) within the network. This structure helps control the magnitude of gradients, effectively stabilizing learning.

#### Reward Scaling
The [original DQN paper](https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf) found that too much variance in the magnitude of network targets can interfere with training, and de-stabilize the network. Their solution was to clip rewards to the range \[-1, +1\]. Later papers found that this is an imperfect solution, and proposed an [improved reward transformation](https://arxiv.org/pdf/1805.11593). Scaling rewards in this

#### Categorical vectors everywhere!
Wherever possible, inputs and outputs are represented as either one-hot or categorical vectors. This has 3 effects:
1. It bounds the magnitude of any single input (or output) in the range \[0, 1\]. As mentioned before, this improves stability, since the network doesn't have to content with both large and small numbers (which tend to cause gradient explosions).
2. Categorical outputs change the problem from regression to classification, enabling the use of cross-entropy loss. [As Julian Schrittweiser explains on his blog](https://www.furidamu.org/blog/2020/12/22/muzero-intuition/): "The categorical representation, or more specifically the cross-entropy loss, ensures that the scale of the gradients is independent of the magnitude of the value to be predicted. This is especially important when you have values varying by many orders of magnitude, eg max episode return of 21 in pong and over a million in atlantis, and makes learning much more robust." 
3. This one is just my guess, but I would imagine that having lots of 0s in the input and output allows the network to chop itself up into many smaller networks, where the "hot" entries in the input determine which sub-networks are active. In a sense, the network can first determine which inputs are "connected" to which outputs, before fine-tuning the exact value of the active outputs.