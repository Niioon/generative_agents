import os
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from utils import relevance_score_fn, create_new_memory_retriever, interview_agent, run_conversation
from gen_agent import GenerativeAgent
from memory import GenerativeAgentMemory
from short_term_memory import ShortTermMemory
from situations import smalltalk_at_party
import json


def main():

    os.environ['OPENAI_API_KEY'] = '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'
    USER_NAME = "Nion"  # The name you want to use when interviewing the agent.
    LLM = ChatOpenAI(model_name="gpt-3.5-turbo")

    with open("characters/catherine.txt", "r") as fp:
        catherine_dict = json.load(fp)
    with open("characters/brian.txt", "r") as fp:
        brian_dict = json.load(fp)
    with open("characters/tom.txt", "r") as fp:
        tom_dict = json.load(fp)

    VERBOSE = False
    long_term_catherine = GenerativeAgentMemory(
        llm=LLM,
        memory_retriever=create_new_memory_retriever(),
        verbose=VERBOSE,
        reflection_threshold=50,
    )

    short_term_catherine = ShortTermMemory(llm=OpenAI(temperature=0), verbose=VERBOSE)

    catherine = GenerativeAgent(
        llm=LLM,
        long_term_memory=long_term_catherine,
        short_term_memory=short_term_catherine,
        verbose=VERBOSE,
        **catherine_dict
    )

    for observation in catherine_dict['initial_observations']:
        catherine.long_term_memory.add_memory(observation)

    long_term_brian = GenerativeAgentMemory(
        llm=LLM,
        memory_retriever=create_new_memory_retriever(),
        verbose=VERBOSE,
        reflection_threshold=50,
    )

    short_term_brian = ShortTermMemory(llm=OpenAI(temperature=0), verbose=VERBOSE)

    brian = GenerativeAgent(
        llm=LLM,
        long_term_memory=long_term_brian,
        short_term_memory=short_term_brian,
        verbose=VERBOSE,
        **brian_dict
    )

    for observation in brian_dict['initial_observations']:
        brian.long_term_memory.add_memory(observation)

    long_term_tom = GenerativeAgentMemory(
        llm=LLM,
        memory_retriever=create_new_memory_retriever(),
        verbose=VERBOSE,
        reflection_threshold=50,
    )

    short_term_tom = ShortTermMemory(llm=OpenAI(temperature=0), verbose=VERBOSE)

    tom = GenerativeAgent(
        llm=LLM,
        long_term_memory=long_term_brian,
        short_term_memory=short_term_brian,
        verbose=VERBOSE,
        **tom_dict
    )

    for observation in tom_dict['initial_observations']:
        tom.long_term_memory.add_memory(observation)

    observations_1, observations_2 = smalltalk_at_party('Tom', 'Brian')

    for observation in observations_1:
        tom.long_term_memory.add_memory(observation)
    for observation in observations_2:
        brian.long_term_memory.add_memory(observation)

    initial_prompt = "Catherine gets into a conversation with brian"
    run_conversation([brian, tom], initial_observation=initial_prompt, max_turns=4)



if __name__ == '__main__':
    main()
