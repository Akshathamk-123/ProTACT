import os
import re
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI

class Evaluator():
    def __init__(self):
        # Load OpenAI API key from environment
        self.model = os.getenv("MODEL")
        if self.openai_api_key is None:
            raise ValueError("OpenAI API key not found. Please set it in the .env file.")

        # Initialize OpenAI Chat Model
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=self.model)

    def evaluate(self, essay_prompt, essay):
        # Define the prompt template
        template = """
        You are an essay evaluator for grade 7-10 students. Please evaluate the following essay based on content, organization, word choice, sentence fluency, conventions, prompt adherence, language, and narrativity. Provide scores between 0 and 1. If a score is not relevant, set it to -1.

        Essay Topic: {prompt}
        Essay: {essay}

        Scores:
        - Overall Score:
        - Content Score:
        - Organization Score:
        - Word Choice Score:
        - Sentence Fluency Score:
        - Conventions Score:
        - Prompt Adherence Score:
        - Language Score:
        - Narrativity Score:
        """

        # Create a prompt template
        prompt = PromptTemplate(input_variables=["essay", "prompt"], template=template)

        # Initialize the LLMChain with the prompt and LLM
        chain = LLMChain(llm=self.llm, prompt=prompt)

        # Generate the response
        response = chain.run({"essay": essay, "prompt": essay_prompt})

        # Extract and convert scores from the response
        scores = re.findall(r"[-+]?\d*\.\d+|\d+", response)
        scores = [float(score) if float(score) != 0 else -1 for score in scores]

        return scores
