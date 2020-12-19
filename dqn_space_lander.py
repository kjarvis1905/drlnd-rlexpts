from dqn_agent import Agent, LearningStrategy
from utils import ArtifactHandler
from collections import deque
import mlflow
import os
import torch
import numpy as np
import gym
import click

mlflow.set_tracking_uri("http://0.0.0.0:5000")

def dqn(
    env,
    agent,
    n_episodes=40,
    max_t=1000, 
    eps_start=1.0, 
    eps_end=0.01, 
    eps_decay=0.995,
    checkpoint_directory = "./artifacts/checkpoints"
    ):

    """Deep Q-Learning.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    chkpt = 0
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if i_episode % 250 == 0:
            print('\nSaving checkpoint after {} episodes\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            output_fname = 'checkpoint_{}.pth'.format(chkpt)
            output_path = os.path.join(checkpoint_directory, output_fname)
            torch.save(agent.qnetwork_local.state_dict(), output_path)
            chkpt += 1
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            output_path = os.path.join(checkpoint_directory, 'solved_checkpoint.pth')
            torch.save(agent.qnetwork_local.state_dict(), output_path)
            break
    return scores

def create_env():
    env = gym.make('LunarLander-v2')
    env.seed(0)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)
    return env

@click.command()
@click.option("--learning-strategy", required=False, default = 'DQN', type = click.Choice(['DQN', 'DDQN']))
@click.option("--n-episodes", required=False, default=4000)
@click.option("--experiment-name", required=False, default=None, type=str)
def run(learning_strategy, n_episodes, experiment_name):
    experiment_name = os.path.basename(__file__).split('.')[0] if experiment_name == None else experiment_name

    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():

        mlflow.log_param('Learning Strategy', learning_strategy)

        env = create_env()

        agent = Agent(state_size=8, action_size=4, seed=0, learning_strategy=LearningStrategy[learning_strategy])

        checkpoint_directory = './artifacts/checkpoints'

        if not os.path.exists(checkpoint_directory):
            os.makedirs(checkpoint_directory)

        with ArtifactHandler() as _ :
            scores = dqn(env, agent, n_episodes=n_episodes, checkpoint_directory=checkpoint_directory)
            print("Saving scores..")
            np.savetxt('./artifacts/scores.txt', scores)

if __name__ == "__main__":
    run()


