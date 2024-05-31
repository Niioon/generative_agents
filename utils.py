import math
import faiss
from langchain_community.docstore import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain_community.vectorstores import FAISS
from memory import GenerativeAgentMemory
from gen_agent import GenerativeAgent
from short_term_memory import ShortTermMemory
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAI
import json

def interview_agent(agent, message, username) -> str:
    """Help the notebook user interact with the agent."""
    new_message = f"{username} says {message}"
    return agent.generate_dialogue_response(new_message)[1]


def pretty_print(string):
    words = string.split()
    for i in range(0, len(words), 25):
        try:
            print(" ".join(words[i: i+25]))
        except IndexError:
            print(" ".join(words[i:]))


def run_conversation(agents, initial_observation: str, max_turns: int = 10, verbose: bool = False) -> list[str]:
    """Runs a conversation between agents."""

    _, observation = agents[1].generate_dialogue_response(initial_observation)
    if verbose:
        pretty_print(observation)
    turns = 0
    dialogue = [initial_observation, observation]
    while turns < max_turns:
        # Right now both agents need to end the conversation for it to stop, good???

        break_dialogue = False
        for agent in agents:
            stay_in_dialogue, observation = agent.generate_dialogue_response(
                observation
            )
            dialogue.append(observation)
            if verbose:
                print("____")
                pretty_print(observation)
            if not stay_in_dialogue:
                break_dialogue = True
        if break_dialogue:
            break

        turns += 1
    return dialogue


def create_agent_from_config(path: str, agent_config=None, long_term_config=None, short_term_config=None):
    """
    Creates agent from config
    """
    default_lt_config = {
        'llm_string': 'gpt-4-0125-preview',
        'verbose': False,
        'reflection_threshold': 50,
    }
    if long_term_config:
        default_lt_config.update(long_term_config)

    default_st_config = {
        'llm_string': 'gpt-4-0125-preview',
        'llm': OpenAI(temperature=0),
        'verbose':False
    }
    if short_term_config:
        default_st_config.update(short_term_config)

    short_term_memory = ShortTermMemory(**default_st_config)
    long_term_memory = GenerativeAgentMemory(**default_lt_config)

    with open(path, "r") as fp:
        agent_dict = json.load(fp)

    default_agent_config = {
        'llm_string': "gpt-4-0125-preview",
        'verbose': False,
    }
    # first update with agent config and then with character configs form file
    if agent_config:
        default_agent_config.update(agent_config)
    agent_dict.update(default_agent_config)
    agent = GenerativeAgent(short_term_memory=short_term_memory, long_term_memory=long_term_memory, **agent_dict)
    return agent

