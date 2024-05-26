import os

from utils import create_agent_from_config, run_conversation, pretty_print
from situations import smalltalk_at_party
import json


def main():

    with open('api_token.txt', 'r') as file:
        api_key = file.read()
    os.environ['OPENAI_API_KEY'] = api_key

    situation_func = smalltalk_at_party
    # creates from config file and evaluates them using eval dialogue
    long_term_config = {'verbose': False}
    agent_config = {'verbose': True}

    situation_func = smalltalk_at_party
    # creates from config file and evaluates them using eval dialogue
    path_1 = "characters/brian.txt"
    path_2 = "characters/catherine.txt"
    # path_2 = "characters/tom.txt"

    print("creating agents")
    agent_1 = create_agent_from_config(path_1, agent_config, long_term_config)
    agent_2 = create_agent_from_config(path_2, agent_config, long_term_config)
    observations_1, observations_2, initial_observation, situation_summary = situation_func(agent_1.name, agent_2.name)
    for observation in observations_1:
        agent_1.long_term_memory.add_memory(observation)
    for observation in observations_2:
        agent_2.long_term_memory.add_memory(observation)
    dialogue = run_conversation([agent_1, agent_2], initial_observation=initial_observation, verbose=True, max_turns=2)
    pretty_print(dialogue)


if __name__ == '__main__':
    main()
