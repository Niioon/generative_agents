import os
from situations import smalltalk_at_party
from utils import create_agent_from_config

def main():

    with open('api_token.txt', 'r') as file:
        api_key = file.read()
    os.environ['OPENAI_API_KEY'] = api_key

    USER_NAME = "Nion"  # The name you want to use when interviewing the agent.
    situation_func = smalltalk_at_party
    # creates from config file and evaluates them using eval dialogue
    path_1 = "characters/brian.txt"
    long_term_config = {'verbose': True}
    agent_config = {'verbose': True}

    agent_1 = create_agent_from_config(path_1, agent_config=agent_config, long_term_config=long_term_config)
    _, observation = agent_1.generate_dialogue_response("Nion says: hello, how are you?")
    print(observation)




if __name__ == '__main__':
    main()
