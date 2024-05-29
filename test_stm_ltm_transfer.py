from utils import create_agent_from_config
from gen_agent import GenerativeAgent
import os
import json
from utils import run_conversation


def main():

    with open('api_token.txt', 'r') as file:
        api_token = file.read()
    os.environ['OPENAI_API_KEY'] = api_token

    harry = GenerativeAgent.load_instance('saved_characters/harry_initial')
    melody = GenerativeAgent.load_instance('saved_characters/melody_initial')

    with open("scenario/arguments.txt", "r") as fp:
        argument_dict = json.load(fp)

    for observation in argument_dict['major']:
        melody.long_term_memory.add_memory(observation)
    # run conversation
    initial_observation = "Harry starts interviewing Melody, the local major, regarding the construction of a new industrial near oakville"
    dialogue = run_conversation([melody, harry], initial_observation=initial_observation, verbose=True, max_turns=3)



if __name__ == '__main__':
    main()
