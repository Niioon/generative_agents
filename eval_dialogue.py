import os
from langchain_community.chat_models import ChatOpenAI

from utils import  run_conversation, create_agent_from_config
from gen_agent import GenerativeAgent
from memory import GenerativeAgentMemory
from short_term_memory import ShortTermMemory
from situations import smalltalk_at_party
import json
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


evaluation_prompt = PromptTemplate.from_template(
    "Your task is to evaluate how natural and realistic the given conversation in a certain situation is.\n"
    + "You will be provided with a character descriptions of each participant and a description of the situation in which the conversation takes place.\n"
    + "On a scale from 1 to 10 rate:\n"
    + "1: The overall naturalness of the conversation. Factor in if the behavior is adequate given the situation and if the character behave like authentic human beings\n"
    + "2: Whether each characters acts according to the information and character traits provided in their description\n"
    + "The situation: {situation}\n"
    + "{agent_1_name}'s description: {agent_1_description}\n"
    + "{agent_2_name}'s description: {agent_2_description}\n"
    + "The conversation: {dialogue}\n"
)


def eval_dialogue(dialogue, agent_1, agent_2, llm, situation_summary, verbose=False):
    kwargs = {
        "agent_1_name": agent_1.name,
        "agent_2_name": agent_2.name,
        "situation": situation_summary,
        "agent_1_summary": agent_1.get_summary(),
        "agent_2_summary": agent_2.get_summary(),
        "dialogue": dialogue,
    }
    return LLMChain(llm=llm, prompt=evaluation_prompt, verbose=verbose).run(**kwargs)


def create_and_eval_dialogues(agent_1: GenerativeAgent, agent_2: GenerativeAgent, llm=None, n: int = 1,
                              situation_func=None, save_name=None) -> None:
    """
    Runs n dialogue between two agents in the situation provided by situan_func
    Dialogues are evaluated using and LLM
    Both are stored in textformat
    """
    if not llm:
        llm = ChatOpenAI(model_name="gpt-4-0125-preview")
    print("adding situational information to memory")
    if situation_func:
        observations_1, observations_2, initial_observation, situation_summary = situation_func(agent_1.name, agent_2.name)
        for observation in observations_1:
            agent_1.long_term_memory.add_memory(observation)
        for observation in observations_2:
            agent_2.long_term_memory.add_memory(observation)
    else:
        initial_observation = f"{agent_1.name} gets into a conversation with {agent_2.name}."
        situation_summary = "A conversation without a specific context"

    if not save_name:
        save_name = f"dialogue_{agent_1.name}_{agent_2.name}_n={n}.txt"
    save_path = "dialogues/" + save_name
    with open(save_path, 'a') as f:
        for i in range(n):
            print("Creating dialogue...")
            dialogue = run_conversation([agent_2, agent_1], initial_observation=initial_observation, verbose=True)
            f.writelines(dialogue)

            print("Evaluating dialogue...")
            evaluation = eval_dialogue(dialogue, agent_1, agent_2, llm, situation_summary)
            f.write(evaluation)
            # generate evaluation


def main():
    with open('api_token.txt', 'r') as file:
         api_key = file.read()
    os.environ['OPENAI_API_KEY'] = api_key

    n = 1
    situation_func = smalltalk_at_party
    # creates from config file and evaluates them using eval dialogue
    path_1 = "characters/brian.txt"
    path_2 = "characters/catherine.txt"
    # path_2 = "characters/tom.txt"

    print("creating agents")
    agent_1 = create_agent_from_config(path_1)
    agent_2 = create_agent_from_config(path_2)

    save_name = f"dialogue_{agent_1.name}_{agent_2.name}_{situation_func}_n={n}.txt"

    create_and_eval_dialogues(agent_1, agent_2, n=1, situation_func=situation_func, save_name=save_name)

    # TODO
    # Fix agents remembering last conversation, either recreate agents every time or implemnt save load functionality


if __name__ == '__main__':
    main()

