import numpy as np
import pandas as pd
import os
from langchain import PromptTemplate, LLMChain
#from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SequentialChain
from dotenv import load_dotenv

# Load environment variables from .env file if not already loaded
load_dotenv()


class EssayFeedbackGenerator:
    def __init__(self):
        # Load OpenAI API key from environment and initialize the model
        #openai_api_key = os.getenv("OPENAI_API_KEY")
        

        # Initialize the fine-tuned OpenAI model
        self.openai_llm = OpenAI(
            model_name="ft:gpt-3.5-turbo-1106:pes:finetune-feedback:AQwJEq7T:ckpt-step-800",
            openai_api_key="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"
        )

        # Define the prompt template
        self.cot_prompt_template = PromptTemplate(
            input_variables=[
                "essay", "prompt", 
                "overall_score", "content_score", "organization_score", 
                "word_choice_score", "sentence_fluency_score", "conventions_score", 
                "prompt_adherence_score", "language_score", "narrativity_score"
            ],
            template="""
Essay Prompt: {prompt}
Essay: {essay}

You are a helpful assistant generating constructive feedback for essays based on specified scores (ranging from 0 to 1, where 1 is excellent and 0 is the least satisfactory). Follow the chain-of-thought reasoning to structure your response step-by-step, considering each aspect logically before providing feedback.

Begin with each parameter in the following order and address it based on its score (omit parameters with a score of -1):

1. **Overall Essay Score** ({overall_score}): Start with an overview, describing the general quality of the essay, considering the overall coherence, engagement, and relevance.
2. **Content** ({content_score}): Reflect on the essay’s depth and relevance, identifying strengths and offering ways to enhance the content if needed.
3. **Organization** ({organization_score}): Evaluate the structure, discussing whether ideas flow logically and where improvement in organization may enhance clarity.
4. **Word Choice** ({word_choice_score}): Assess vocabulary usage for precision and effectiveness, suggesting more fitting or expressive alternatives if beneficial.
5. **Sentence Fluency** ({sentence_fluency_score}): Analyze sentence variation and flow, recommending smoother transitions or variations where appropriate.
6. **Conventions** ({conventions_score}): Address grammar, punctuation, and spelling, noting any recurring errors and suggesting consistency improvements.
7. **Prompt Adherence** ({prompt_adherence_score}): Examine how closely the essay follows the prompt, suggesting ways to improve alignment if needed.
8. **Language** ({language_score}): Evaluate expressiveness and tone, noting how language choices enhance or detract from clarity and engagement.
9. **Narrativity** ({narrativity_score}): Consider the narrative flow, identifying ways to heighten reader engagement with sensory details or emotional resonance.

**Chain-of-Thought Reasoning Instructions**:
- For each parameter, reflect briefly on the strengths, followed by specific suggestions for enhancement.
- Provide coherent and cohesive feedback, integrating positive observations before constructive criticism.
- Avoid listing scores; instead, create a single flowing paragraph that incorporates each relevant point naturally.
- Aim for 150-180 words in total.

Generate feedback below in a coherent and constructive manner:
{{
    "Feedback": "The essay presents a well-considered account with some commendable strengths and areas for improvement..."
}}
            """
        )

    def generate_feedback(self, essay_text, essay_prompt, scores):
        # Set up the feedback chain
        feedback_chain = LLMChain(
            llm=self.openai_llm,
            prompt=self.cot_prompt_template
        )

        # Run the chain to get feedback
        result = feedback_chain.invoke({
            "essay": essay_text,
            "prompt": essay_prompt,
            **scores
        })

        # Extract and return the feedback text
        feedback_text = result.get('text', '')
        return feedback_text
