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
You are an essay evaluator for grade 7–10 students. Please evaluate the following essay based on the following criteria. For each criterion, provide a detailed step-by-step evaluation, followed by a score between 0 and 1 (where 1 is excellent and 0 is the least satisfactory). If a criterion is not relevant, set it to -1. Focus on areas that need improvement and be critical when necessary.

Essay Topic: {prompt}
Essay: {essay}

Please evaluate the essay step-by-step as follows:

1. **Content Evaluation**:
   - Analyze the content of the essay critically. Does the essay address the topic in enough detail? Are there areas where the essay lacks depth or clarity? Are there key points missing or parts of the essay that seem off-topic or underdeveloped? Pay close attention to weak arguments, generalizations, and any unsupported claims.
   - Provide a score for content based on the quality and depth, but be critical if there are any gaps or weaknesses in addressing the topic.

2. **Organization Evaluation**:
   - Critically evaluate the structure of the essay. Is the essay well-organized, or are there confusing parts? Does the argument follow a logical progression, or are there abrupt shifts in topics? Are paragraphs appropriately divided, and do they maintain focus? If the essay lacks cohesion or has poor transitions between ideas, mention it.
   - Provide a score for organization, focusing on the essay’s structure, logical flow, and clarity.

3. **Word Choice Evaluation**:
   - Evaluate the vocabulary used critically. Are the words appropriate for the topic and audience? Does the vocabulary reflect the required level of sophistication for a grade 7–10 student? Are there any words that are misused or could be replaced with more precise alternatives? Consider areas where vague or imprecise language weakens the essay.
   - Provide a score for word choice, being critical if the vocabulary is either too simple or incorrectly applied.

4. **Sentence Fluency Evaluation**:
   - Critically assess sentence fluency. Are the sentences well-constructed, or do they feel awkward or disjointed? Are there excessive run-on sentences, fragments, or overly simplistic sentence structures? Does the essay have variety in sentence length and complexity? If the essay is choppy or lacks fluency, point it out.
   - Provide a score for sentence fluency, highlighting any problems with readability or sentence structure.

5. **Conventions Evaluation**:
   - Critically evaluate the essay's adherence to grammar, spelling, and punctuation conventions. Are there recurring errors in grammar, spelling, or punctuation? Do these errors impact the readability of the essay? Is the writing hard to follow because of mechanical mistakes? If there are serious or frequent issues, be sure to mention them.
   - Provide a score for conventions, offering constructive criticism for any mistakes and suggesting improvements.

6. **Prompt Adherence Evaluation**:
   - Assess how well the essay stays on topic and answers the prompt. Does the essay wander off-topic or fail to fully address the prompt? Are key elements of the prompt ignored or inadequately explored? If the essay deviates from the prompt or doesn’t fully engage with it, this should be noted.
   - Provide a score for prompt adherence, being critical if the essay does not fully or adequately address the prompt.

7. **Language Evaluation**:
   - Critically evaluate the language used in the essay. Does the language reflect the appropriate tone for the topic and audience? Are there sections where the tone feels inconsistent, too casual, or too formal? Does the language clearly express ideas, or are there moments of confusion? Pay attention to any unclear, ambiguous, or vague language.
   - Provide a score for language, being critical if the language impedes understanding or doesn’t match the expected tone.

8. **Narrativity Evaluation**:
   - If the essay includes narrative elements, assess the narrative quality critically. Does the essay engage the reader, or does it feel flat? Are emotional or sensory details included, or is the narrative lacking in depth? Does the story feel compelling, or does it fail to hold the reader’s attention? If the narrative feels weak or unconvincing, make sure to point it out.
   - Provide a score for narrativity, being critical if the essay fails to effectively engage or lacks necessary details.

**Overall Evaluation**:
   - After evaluating each criterion individually, provide an overall score for the essay. This should reflect the general quality of the essay, but make sure to account for any serious shortcomings identified in the individual evaluations. If any criteria were significantly lacking, the overall score should reflect that.
   - Provide constructive feedback on the essay’s strengths and areas for improvement.

Now, please provide the scores for each criterion as well as the overall score without any additional explanation:
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
