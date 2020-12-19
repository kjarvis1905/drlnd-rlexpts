from dqn_agent import Agent, LearningStrategy
from utils import ArtifactHandler
from collections import deque
import mlflow
import os
import torch
import numpy as np
import gym
import click
from enum import Enum
from dqn_agent import Agent, LearningStrategy
import matplotlib.pyplot as plt
import re
from pyvirtualdisplay import Display
import time

mlflow.set_tracking_uri("http://0.0.0.0:5000")
mlflow_client = mlflow.tracking.MlflowClient()

class AgentSource(Enum):
    FILE = "FILE"
    EXPERIMENT = "EXPERIMENT"

def retrieve_agent_checkpoint(source: AgentSource, location: str):
    assert source in AgentSource
    if source == AgentSource.FILE:
        pass
    elif source == AgentSource.EXPERIMENT:
        latest_run_id = (
            mlflow_client
            .search_runs(
                mlflow.get_experiment_by_name(location).experiment_id
                )[0]
            .info
            .run_id
        )
        checkpoints_directory = mlflow_client.download_artifacts(latest_run_id, './checkpoints/')
        checkpoint_path = os.path.join(checkpoints_directory, 'solved_checkpoint.pth')
    return checkpoint_path

@click.command()
@click.argument("agent-source", type=click.Choice(['file', 'experiment']))
@click.argument("location", type=str)
@click.option("--n-episodes", required=False, default=3)
def run(agent_source, location, n_episodes):


    source = AgentSource[agent_source.upper()]
    agent = Agent(state_size=8, action_size=4, seed=0, learning_strategy=LearningStrategy.DQN)

    path_to_agent_checkpoint = retrieve_agent_checkpoint(source, location)
    agent.qnetwork_local.load_state_dict(torch.load(path_to_agent_checkpoint, map_location=lambda storage, loc: storage))

    #display = Display(visible=0, size=(1400, 900))
    #display.start()

    env = gym.make('LunarLander-v2')
    env.seed(0)
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)


    for i in range(n_episodes):
        state = env.reset()
        #img = plt.imshow(
        env.render(mode='rgb_array')
        for j in range(500):
            action = agent.act(state)
            #img.set_data(
            env.render(mode='rgb_array')
            plt.axis('off')
            time.sleep(0.1)
            #display.display(plt.gcf())
            #display.clear_output(wait=True)
            state, reward, done, _ = env.step(action)
            if done:
                break 
            
    env.close()



if __name__ == "__main__":
    run()


