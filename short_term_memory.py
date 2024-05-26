from typing import Any, Dict, List, Optional
from langchain.pydantic_v1 import Field

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from langchain.schema import BaseMemory, BaseChatMessageHistory, BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.messages import BaseMessage, get_buffer_string, ChatMessage, SystemMessage
from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
from langchain.memory.prompt import SUMMARY_PROMPT
from langchain.utils import mock_now
import re
import logging
import warnings

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fh = logging.FileHandler('short_term.log', mode='w')
logger.addHandler(fh)


class ShortTermMemory(BaseMemory):
    """Short-term Memory for the generative agent.
    For now focused on keeping track of a conversation
    Oriented after ConversationSummaryMemory from langchain
    """

    llm_string: str
    """ Identifier of OPENAI model, used to create llm instance, used for recreation of llm when loading"""
    llm: Optional[BaseLanguageModel]
    """The underlying language model."""
    chat_memory: BaseChatMessageHistory = Field(default_factory=ChatMessageHistory)
    """Memory of the conversation"""
    summary: str = "nothing"
    """Current Summary"""
    prompt: BasePromptTemplate = SUMMARY_PROMPT
    """Prompt for summarizing"""
    input_key: str = "observation"
    """"""
    memory_key: str = "history"
    """Key for inserting summary of chat history into prompt"""
    verbose: bool = False

    # prefix: str = "agent"
    # """Prefix for storing messages, should be agents name"""
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if not self.llm:
            self.llm = ChatOpenAI(model_name=self.llm_string)

    def predict_new_summary(
            self, messages: List[BaseMessage], existing_summary: str
    ) -> str:
        new_lines = get_buffer_string(
            messages,
        )
        chain = LLMChain(llm=self.llm, prompt=self.prompt, verbose=self.verbose)
        return chain.predict(summary=existing_summary, new_lines=new_lines)

    def process_message(self, message: str) -> BaseMessage:
        """
        Processes strings into langchain Basemessage
        Can deal with messages of the form:
            person: message: content
            person said content
        Other messages will be stored as system messages that provide context to the conversation
        """

        if message.split()[0] == ('[SAY]' or "[GOODBYE]"):
            message = message.split(sep=' ', maxsplit=1)[-1]

        if message.split()[0][-1] == ':':
            role, content = re.split(':', message, maxsplit=1)
            # if self.verbose: logger.info(f'{role}:{content}')
            p_message = ChatMessage(content=content, role=role)
        elif message.split()[1] == 'said':
            role, content = re.split('said', message, maxsplit=1)
            # if self.verbose: logger.info(f'{role}:{content}')
            p_message = ChatMessage(content=content, role=role)
        else:
            p_message = SystemMessage(content=message)

        return p_message

    @property
    def memory_variables(self) -> List[str]:
        """Will always return list of memory variables.

        :meta private:
        """
        return [self.memory_key]

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Return history summary."""
        return {self.memory_key: self.summary}

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save context from this conversation to summary."""
        input_str = inputs.get(self.input_key)
        output_str = outputs['text'].strip().split("\n")[0]
        # logger.info(input_str)
        # logger.info(output_str)
        if input_str:
            input_message = self.process_message(input_str)
            output_message = self.process_message(output_str)
            self.chat_memory.add_message(input_message)
            self.chat_memory.add_message(output_message)
        else:
            warnings.warn("No input message for the short term memory")
        self.summary = self.predict_new_summary(
            self.chat_memory.messages[-2:], self.summary
        )
        if self.verbose:
            logger.info(self.summary)
            logger.info(self.chat_memory.messages)

    def clear(self) -> None:
        """Clear memory contents."""
        self.chat_memory.clear()
        self.summary = ""
