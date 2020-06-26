# Project report
Goal: use Evolutionary Algorithms and Neural Networks to create a model which will be able to walk in pybullet enviroment.

I used 'AntMuJoCoEnv-v0' model.
![](https://i.imgur.com/3oaq4CX.png)

## About pybullet enviroment
Pybulletgym enviroment is a free substitute of MuJuCo. Despite small visual differences it is very similar. In every step of the simulation we provide an action we want to perform and we get and some informations like the observation (what our model sees) or the reward from doing our last action.

## Selected methods
### Neural Network
Our model isn't very complicated. All the knowledge it needs is a 111 element vector and we controll just 8 joints. It means that even a simple and shallow neural net should be able to perform quite well. But to teach it using only NN we'll need also a loss function which is hard to come up with. That's why we'll need to try to connect NN with other technique.

### Evolutionary Algorithms
After every move we make, we get a reward. It means that we can evaluate a model by doing a fixed amount of iterations and the score will be a mean reward collected during this run. After the evaluation we'll know which individuals are good and taking that knowledge to the next epoch we can create even better ones.

## Different approaches
### Neural Networks + CEM algorithm
This method is based on modelling $\mu \in R^N$ and $\Sigma \in R^{N \times N}$ parameters which corresponds to the mean and covariance matrix of the population. We change them at every step using $M$ best individuals.
Original algoritms uses $\Sigma$ covariance matrix but to speed up the computations I've decided to represent it by a vector being a diagonal of the original one. It means I'm only recording the variance in each dimension.
The biggest set back of this approach is the time needed to evaluate an individual, in order to do it we need to do as many forward passes as number of consecutive actions we want to check.

**CEM**:
```
for each epoch
1. draw a population from mu, sigma and epsilon
2. evaluate population
3. update params (mu, sigma, epsilon)
```
Updating parameters:
$\mu_{new} = \sum_{i=1}^{M}\lambda_i z_i$
$\Sigma_{new} = \sum_{i=1}^{M}\lambda_i(z_i - \mu_{old})^2 + \epsilon I$
where 
$\lambda_i = \frac{log(1+M)/i}{\sum_{i=1}^{M}log(1+M)/i}$,
$M \text{ - number of best individuals used to update params}$,
$z_i \text{ - i-th best individual}$

**Neural Net**:
```
Linear(111, 100),
Tanh(),
Linear(100, 8),
Tanh()
```
### Predictor of the reward + CEM algorithm
To speed up the evaluation proccess I created a predictor which given both observation and action will produce approximation of the reward received. To do it I collected a lot of data and use a Neural Network with SGD to teach it. After that rather than doing as many forward passes through our agent network as number of steps, we can do it all in just one forward pass, then concatenate observations with actions, put them through the predictor and compute the mean reward.

To have a nice variety of data 1/10 observations+actions and reward was collected from the random agent and the rest contained our pretrained model learned on pure CEM with noise to his actions to get better spread dataset.

Total of collected data points was 2,000,000.
```
Linear(119, 500),
Tanh(),
Linear(500, 500),
Tanh(),
Linear(500, 500),
Tanh(),
Linear(500, 1)

Loss - mean squared error loss
```

#### With data distibution
Collected before data allowed us to see the distribution of the observation. It mean that we can just evaluate population of CEM on new data generated from this distribution.

#### With pre-recorded data
Other idea was to use the collected data, put it into DataLoader and evaluate individuals on real data.

## Results
### CEM
Here's a plot of average reward for each action. Green line represents the best reward of the epoch, blue - mean and red - the worst.
![](https://i.imgur.com/lwTqmEg.png)
In first 200 epochs there was a huge improvement in both the best and mean individual and after that it was slightly but consistently getting better. The worst individuals were also improving but there was some ups and downs.

After that I run CEM for 2000 more iterations but unfortunately there's no plot visualizing the change in rewards. By looking at later individuals they tended to be similar or sometimes slighty better than those from 1,000-ish epochs.

Computing a total of 3,000 epochs took about 30 hours using Colab.


### With predictor
Running for a 10,000 in both cases took about 1.5 hour.

#### And generating data
![](https://i.imgur.com/WOpZQKo.png)


#### And collected data
![](https://i.imgur.com/621loqX.png)

Both plots look similar. During all the epochs there's no significant improvement, the best and the worst evals from each epoch are differing only by slight margin and even the random individuals have score near 1.5 which, comparing to the CEM plot, is very suspicious. 

The good explaination for that is the lack of very bad and very good actions recored to teach the reward predictor. 

## Conclusions
A simple model using CEM was able to learn how to walk quite well. The biggest set back was the time it took. It was also almost impossible to get a fine-tuned weights because the neural net had about 12,000 parameters.

Using a reward predictor sped up the proccess by a huge margin and made the CEM way faster. Unfortunately, it is very hard to teach the neural net to predict the reward because there's no fixed size interval for it. Reward is a real number from -infinity to +infinity. It is possible to have such a predictor but to gather all the data we'll have to have models varying from very bad to extremely good. 