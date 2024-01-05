import os
import logging
from typing import List, Union, Generator
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods, ModelTypes
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from lib.model import Message
from lib.llama_utils import generate_llama_prompt, parse_llama_response

WATSONX_URL = os.getenv("WATSONX_URL")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
logger = logging.getLogger(__name__)


class WatsonxLLM():
    def __init__(self):
        assert WATSONX_URL, "Specify a WATSONX_URL env variable"
        assert WATSONX_API_KEY, "Specify a WATSONX_API_KEY env variable"
        assert WATSONX_PROJECT_ID, "Specify a WATSONX_PROJECT_ID env variable"
        self.model = Model(
            credentials={ "url": WATSONX_URL, "apikey": WATSONX_API_KEY },
            project_id=WATSONX_PROJECT_ID,
            model_id=ModelTypes.LLAMA_2_70B_CHAT,
            params={
                GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
                GenParams.MIN_NEW_TOKENS: 1,
                GenParams.MAX_NEW_TOKENS: 250
            }
        )
        logger.info("Watsonx is ready")

    def complete_chat(self, messages: List[Message], stream=False) -> Union[Message, Generator[Message, None, None]]:
        """
        Returns a chat completion, i.e. a message from the assistant.
        Optionally generates the response in a streaming fashion.
        """
        # Generate a prompt for the chat
        prompt = generate_llama_prompt(messages)
        logger.debug("Prompt:", prompt)
        # Streaming mode
        if stream:
            res_txt = ""
            for chunk in self.model.generate_text_stream(prompt):
                if not chunk:
                    continue
                res_txt += chunk
                logger.debug("Next response chunk:", res_txt)
                res_msg = parse_llama_response(res_txt)
                yield res_msg
            logger.info("Parsed response message:", res_msg)
        else:
            # Non-streaming mode
            res_txt = self.model.generate_text(prompt)
            logger.debug("Plain text response:", res_txt)
            res_msg = parse_llama_response(res_txt)
            logger.info("Parsed response message:", res_msg)
            return res_msg
