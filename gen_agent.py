import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain_openai import ChatOpenAI

from langchain.schema.messages import BaseMessage, get_buffer_string, ChatMessage, SystemMessage
from langchain.pydantic_v1 import BaseModel, Field
from langchain.memory import CombinedMemory
from short_term_memory import ShortTermMemory
from memory import GenerativeAgentMemory
import warnings
import json
import shutil
import uuid
import os


class GenerativeAgent(BaseModel):
    """An Agent as a character with memory and innate characteristics."""

    name: str
    """The character's name."""
    age: Optional[int] = None
    """The optional age of the character."""
    character_traits: str = "N/A"
    """Permanent character_traits to ascribe to the character."""
    communication_style: str = "N/A"
    """They way the character expresses them selves"""
    current_goal: str = "N/A"
    """Current goal to achieve with characters behavior"""
    status: str
    """The character_traits of the character you wish not to change."""
    long_term_memory: GenerativeAgentMemory
    """The memory object that combines relevance, recency, and 'importance'."""
    short_term_memory: ShortTermMemory
    """Short term memory for keeping track of Conversations"""
    memory: Optional[CombinedMemory]
    """ Wrapper to pass short and longterm memory to chain together"""
    llm_string: str
    """ Identifier of OPENAI model, used to create llm instance, used for recreation of llm when loading"""
    llm: Optional[BaseLanguageModel]
    """The underlying language model."""
    initial_observations: Optional[List[str]]
    """List of initial observations which will be added to longterm memory on construction"""
    in_situation: bool = False
    """if agent is currently in a "situation", used for short term memory management"""
    verbose: bool = False
    summary: str = ""  #: :meta private:
    """Stateful self-summary generated via reflection on the character's memory."""
    summary_refresh_seconds: int = 3600  #: :meta private:
    """How frequently to re-generate the summary."""
    last_refreshed: datetime = Field(default_factory=datetime.now)  # : :meta private:
    """The last time the character's summary was regenerated."""
    daily_summaries: List[str] = Field(default_factory=list)  # : :meta private:
    """Summary of the events in the plan that the agent took."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if not self.llm:
            self.llm = ChatOpenAI(model_name=self.llm_string)
        # initialize combined memory from single memory objects
        self.memory = CombinedMemory(memories=[self.long_term_memory, self.short_term_memory])

        for observation in self.initial_observations:
            self.long_term_memory.add_memory(observation)
        # remove initial observations after adding them to memory to avoid side effects
        self.initial_observations = []



    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @staticmethod
    def _parse_list(text: str) -> List[str]:
        """Parse a newline-separated string into a list of strings."""
        lines = re.split(r"\n", text.strip())
        return [re.sub(r"^\s*\d+\.\s*", "", line).strip() for line in lines]

    def chain(self, prompt: PromptTemplate) -> LLMChain:
        """Returns chain for the given prompt with access to LTM"""
        return LLMChain(
            llm=self.llm, prompt=prompt, verbose=self.verbose, memory=self.long_term_memory
        )

    def conversation_chain(self, prompt: PromptTemplate) -> LLMChain:
        """Returns chain for the given prompt with access to LTM and STM"""
        return LLMChain(
            llm=self.llm, prompt=prompt, verbose=self.verbose, memory=self.memory,
        )

    def start_situation(self):
        if not self.in_situation:
            self.short_term_memory.clear()
            self.in_situation = True
        else:
            warnings.warn('Agent already is in a situation. This function call will be ignored')
            return

    def end_situation (self):
        if self.in_situation:
            #conv_summary = self.short_term_memory.summary
            #self.long_term_memory.add_memory(conv_summary)
            self._reflect_on_conversation()
            self.short_term_memory.clear()
            self.in_situation = False
        else:
            warnings.warn("agent is not in a situation. This function call will be ignored")

    def _appraise_statement(self, statement):
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            + "\nIt is {current_time}."
            + "\n{agent_name}'s status: {agent_status}"
            + "\nSummary of relevant context from {agent_name}'s memory:"
            + "\n{relevant_memories}"
            + "\nThe statement which {agent_name} appraises: {statement}"
            + "\nGiven the context from memory and the characters summary, what is {agent_name}'s emotional appraisal of the statement"
            + "\nThe appraisal can for example be a degree of agreement/disagreement or any kind of emotional response"
            + "\nAnswer precisely in one sentence"
        )
        agent_summary_description = self.get_summary()
        relevant_memories_str = self.summarize_related_memories(statement)
        current_time_str = datetime.now().strftime("%B %d, %Y, %I:%M %p")

        kwargs: Dict[str, Any] = dict(
            agent_summary_description=agent_summary_description,
            current_time=current_time_str,
            relevant_memories=relevant_memories_str,
            agent_name=self.name,
            current_goal=self.current_goal,
            statement=statement,
            agent_status=self.status,
        )
        return self.chain(prompt=prompt).invoke(input=kwargs)['text']
    def _reflect_on_conversation(self):
        """
        called after situation ended.
        """

        # first get key points of conversation
        prompt = PromptTemplate.from_template(
            """
            What are the key points or arguments of the following conversation and by whom were they made:
            {conversation}
            Give a list of short and precise statements in the following form
            # statement 1
            # statement 2
            ...
            """
        )
        conversation = get_buffer_string(self.short_term_memory.chat_memory.messages)
        statements = self.chain(prompt).invoke(input={'conversation': conversation})['text']
        print(statements)
        statements = statements.split('#')[1:]
        print(statements)
        # append appraisal to the respective statements
        new_memories = [", ".join([statement, self._appraise_statement(statement)]) for statement in statements]
        print(new_memories)
        for memory in new_memories:
            self.long_term_memory.add_memory(memory)


    def _get_entity_from_observation(self, observation: str) -> str:
        #prompt = PromptTemplate.from_template(
        #    "What is the observed entity in the following observation? {observation}"
        #    + "\nEntity="
        #)
        prompt = PromptTemplate.from_template(
            "Who or what is talking or acting in the following observation: {observation}"
            + "\nAnswer with a single expression"
        )
        if self.verbose: print('In _get_entity_from_observation: ')
        return self.chain(prompt).invoke(input={'observation': observation})['text']

    def _get_entity_action(self, observation: str, entity_name: str) -> str:
        prompt = PromptTemplate.from_template(
            "Describe in one sentence what {entity} is doing in the following observation? {observation}"
            + "\n{entity} is..."
        )
        return (
            self.chain(prompt).invoke(input={'observation': observation, 'entity': entity_name})['text']
        )

    def summarize_related_memories(self, observation: str) -> str:
        """Summarize memories that are most relevant to an observation."""
        prompt = PromptTemplate.from_template(
            """
            Given the presented context from memory:
            Provide a short summary of the relationship between {agent_name} and {entity_name}
            Provide a short summary of the relevant memories regarding the observation
            Context from memory:
            {relevant_memories}
            Observation:
            {observation}
            Only answer based on the provided memories and DO NOT assume anything else.
            Answer with a maximum of two sentences for each point and answer in the following form
            Relationship between {agent_name} and {entity_name}:...
            Summary of relevant memories:...
            """
        )
        if self.verbose: print('In summarize_related_memories: Get entity from Observation')
        entity_name = self._get_entity_from_observation(observation)
        if self.verbose: print(f'Answer: Entity  is {entity_name}')
        if self.verbose: print('In summarize_related_memories: Get entity action')
        entity_action = self._get_entity_action(observation, entity_name)
        if self.verbose: print(f'Answer: {entity_action}')

        q1 = f"What is the relationship between {self.name} and {entity_name}"
        q2 = entity_action
        # if self.verbose: print(f'Queries used for retrieving relevant memories. 1: {q1}, 2: {q2}')
        # if self.verbose: print(f'In summarize_related_memories: Get relationship between {self.name} and {entity_name}')
        # q1 and q2 two are used to query the memory for relevant memories
        input_dict = {
            'entity_name': entity_name,
            'agent_name': self.name,
            'observation': observation,
            'queries': [q1, q2]
        }
        return self.chain(prompt=prompt).invoke(input=input_dict)['text']

    def _generate_reaction(
            self, observation: str, suffix: str, now: Optional[datetime] = None
    ) -> str:
        """React to a given observation or dialogue act."""
        prompt = PromptTemplate.from_template(
            "{agent_summary_description}"
            + "\nIt is {current_time}."
            + "\n{agent_name}'s status: {agent_status}"
            + "\nSummary of relevant context from {agent_name}'s memory:"
            + "\n{relevant_memories}"
            + "\nSummary of what has happened so far in the current situation or conversation:"
            + "\n{history}"
            + "\nThe current observation to which {agent_name} reacts: {observation}"
            + "\n{agent_name}'s reaction should be in line with its current goal"
            + "\n{agent_name}'s reaction should be in line with the given character traits and the agents communication style"
            + "\n\n"
            + suffix
        )
        if self.verbose: print('In _generate_reaction: Get current agent summary')
        agent_summary_description = self.get_summary(now=now)

        if self.verbose: print('In _generate_reaction: Summarize related memories')
        relevant_memories_str = self.summarize_related_memories(observation)
        if self.verbose: print(f'Answer: {relevant_memories_str}')
        current_time_str = (
            datetime.now().strftime("%B %d, %Y, %I:%M %p")
            if now is None
            else now.strftime("%B %d, %Y, %I:%M %p")
        )
        kwargs: Dict[str, Any] = dict(
            agent_summary_description=agent_summary_description,
            current_time=current_time_str,
            relevant_memories=relevant_memories_str,
            agent_name=self.name,
            current_goal=self.current_goal,
            observation=observation,
            agent_status=self.status,
        )
        consumed_tokens = self.llm.get_num_tokens(
            prompt.format(history="", **kwargs)
        )
        kwargs[self.long_term_memory.most_recent_memories_token_key] = consumed_tokens
        return self.conversation_chain(prompt=prompt).invoke(input=kwargs)['text']

    def _clean_response(self, text: str) -> str:
        return re.sub(f"^{self.name} ", "", text.strip()).strip()

    def generate_reaction(
            self, observation: str, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """React to a given observation."""
        call_to_action_template = (
                "Should {agent_name} react to the observation, and if so,"
                + " what would be an appropriate reaction? Respond in one line."
                + ' If the action is to engage in dialogue, write:\nSAY: "what to say"'
                + "\notherwise, write:\nREACT: {agent_name}'s reaction (if anything)."
                + "\nEither do nothing, react, or say something but not both.\n\n"
        )
        full_result = self._generate_reaction(
            observation, call_to_action_template, now=now
        )
        print(full_result)
        result = full_result.strip().split("\n")[0]
        # AAA
        self.long_term_memory.save_context(
            {},
            {
                self.long_term_memory.add_memory_key: f"{self.name} observed "
                                            f"{observation} and reacted by {result}",
                self.long_term_memory.now_key: now,
            },
        )
        if "REACT:" in result:
            reaction = self._clean_response(result.split("REACT:")[-1])
            return False, f"{self.name} {reaction}"
        if "SAY:" in result:
            said_value = self._clean_response(result.split("SAY:")[-1])
            return True, f"{self.name} said {said_value}"
        else:
            return False, result

    def generate_dialogue_response(
            self, observation: str, now: Optional[datetime] = None
    ) -> Tuple[bool, str]:
        """React to a given observation."""
        call_to_action_template = (
            "What would {agent_name} say? To end the conversation, write:"
            ' [GOODBYE] {agent_name}: "what to say". Otherwise to continue the conversation,'
            ' write: [SAY] {agent_name}: "what to say next"\n\n'
        )
        # if observation is in format Name: sentence, transform into Name said sentence
        if observation.split()[0][-1] == ':':
            observation = re.sub(pattern=":", repl=" said", string=observation)
        full_result = self._generate_reaction(
            observation, call_to_action_template, now=now
        )
        result = full_result.strip().split("\n")[0]
        response_text = result.split(sep=' ', maxsplit=1)[-1]
        if "[GOODBYE]" in result:
            return False, response_text
        if "[SAY]" in result:
            return True, response_text
        else:
            warnings.warn("LLM did not adhere to output structure. Output might not make sense")
            return False, response_text

    ######################################################
    # Agent stateful' summary methods.                   #
    # Each dialog or response prompt includes a header   #
    # summarizing the agent's self-description. This is  #
    # updated periodically through probing its memories  #
    ######################################################
    def _compute_agent_summary(self) -> str:
        """
        Computes the agent summary from relevant memories.
        Memories are queried using the agents name
        """
        prompt = PromptTemplate.from_template(
            "How would you summarize {name}'s core characteristics given the"
            + " following statements:\n"
            + "{relevant_memories} \n"
            + "Do not embellish."
            + "\n\nSummary: "
        )

        # The agent seeks to think about their core characteristics.
        if self.verbose: print('In _compute_agent_summary: ')
        return (
            self.chain(prompt)
            .invoke({'name': self.name, 'queries': [f"{self.name}'s core characteristics"]})['text']
            .strip()
        )

    def get_summary(
            self, force_refresh: bool = False, now: Optional[datetime] = None
    ) -> str:
        """Return a descriptive summary of the agent."""
        current_time = datetime.now() if now is None else now
        since_refresh = (current_time - self.last_refreshed).seconds
        if (
                not self.summary
                or since_refresh >= self.summary_refresh_seconds
                or force_refresh
        ):
            if self.verbose: print('In get_summary: Compute new agent summary')
            self.summary = self._compute_agent_summary()
            self.last_refreshed = current_time
        else:
            if self.verbose: print('In get_summary: Using old summary')
        age = self.age if self.age is not None else "N/A"
        return (
                f"Name: {self.name} (age: {age})"
                + f"\nCharacter traits: {self.character_traits}"
                + f"\nCurrent Goal: {self.current_goal}"
                + f"\nCommunication Style{self.communication_style}"
                + f"\n{self.summary}"
        )

    def get_full_header(
            self, force_refresh: bool = False, now: Optional[datetime] = None
    ) -> str:
        """Return a full header of the agent's status, summary, and current time."""
        now = datetime.now() if now is None else now
        summary = self.get_summary(force_refresh=force_refresh, now=now)
        current_time_str = now.strftime("%B %d, %Y, %I:%M %p")
        return (
            f"{summary}\nIt is {current_time_str}.\n{self.name}'s status: {self.status}"
        )

    @staticmethod
    def save_instance(agent, path=None):
        if not path:
            path = 'saved_characters/' + uuid.uuid4().hex
        agent_json = agent.json(exclude={'llm', 'memory', 'long_term_memory', 'short_term_memory'})
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)
            os.makedirs(path)

        with open( path + '/agent_dict.json', 'w') as outfile:
            outfile.write(agent_json)

        # save ltm
        GenerativeAgentMemory.save_instance(agent.long_term_memory, path)
        #save stm config
        stm_json = agent.short_term_memory.json(exclude={'llm', 'chat_memory', 'prompt'})
        with open(path + '/stm_dict.json', 'w') as outfile:
            outfile.write(stm_json)

    @staticmethod
    def load_instance(path):
        with open(path + '/agent_dict.json', 'r') as openfile:
            agent_dict = json.load(openfile)
        agent_dict['llm'] = ChatOpenAI(model_name=agent_dict['llm_string'])

        # load ltm
        agent_dict['long_term_memory'] = GenerativeAgentMemory.load_instance(path)
        # load stm
        with open(path + '/stm_dict.json', 'r') as openfile:
            stm_dict = json.load(openfile)
        # for now temperature is hardcoded
        stm_dict['llm'] = ChatOpenAI(model_name=agent_dict['llm_string'])
        agent_dict['short_term_memory'] = ShortTermMemory(**stm_dict)

        return GenerativeAgent(**agent_dict)





