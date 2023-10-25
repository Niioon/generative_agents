import math
import faiss
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from memory import GenerativeAgentMemory
from gen_agent import GenerativeAgent
from short_term_memory import ShortTermMemory
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import json

def interview_agent(agent, message, username) -> str:
    """Help the notebook user interact with the agent."""
    new_message = f"{username} says {message}"
    return agent.generate_dialogue_response(new_message)[1]

def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    # This will differ depending on a few things:
    # - the distance / similarity metric used by the VectorStore
    # - the scale of your embeddings (OpenAI's are unit norm. Many others are not!)
    # This function converts the euclidean norm of normalized embeddings
    # (0 is most similar, sqrt(2) most dissimilar)
    # to a similarity function (0 to 1)
    return 1.0 - score / math.sqrt(2)


def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    # Define your embedding model
    embeddings_model = OpenAIEmbeddings()
    # Initialize the vectorstore as empty
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model.embed_query,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=relevance_score_fn,
    )
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, other_score_keys=["importance"], k=15
    )


def pretty_print(string):
    words = string.split()
    for i in range(0, len(words), 25):
        try:
            print(" ".join(words[i: i+25]))
        except IndexError:
            print(" ".join(words[i:]))


def run_conversation(agents, initial_observation: str, max_turns: int=4, verbose=False) -> list[str]:
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
                print("____")
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
        'llm': ChatOpenAI(model_name="gpt-3.5-turbo"),
        'memory_retriever': create_new_memory_retriever(),
        'verbose': False,
        'reflection_threshold': 50,
    }
    if long_term_config:
        default_lt_config.update(long_term_config)

    default_st_config = {
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
        'llm': ChatOpenAI(model_name="gpt-3.5-turbo"),
        'verbose': False,
    }
    # first update with agent config and then with character cnofigs form file
    if agent_config:
        default_agent_config.update(agent_config)
    default_agent_config.update(agent_dict)
    agent = GenerativeAgent(short_term_memory=short_term_memory, long_term_memory=long_term_memory, **default_agent_config)
    return agent

