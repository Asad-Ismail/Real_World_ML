import dspy
import os
from dotenv import load_dotenv
from typing import List
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

# Configure the LLM
# Make sure your LM is configured correctly
lm = dspy.LM("openrouter/moonshotai/kimi-k2:free", api_key=os.getenv("OPENROUTER_API_KEY"), api_base="https://openrouter.ai/api/v1")
dspy.configure(lm=lm)

# This signature is fine, no changes needed
class GenerateProblemAndContext(dspy.Signature):
    """[... your existing prompt for this class ...]"""
    current_date: str = dspy.InputField(desc="The current date to ground the reasoning, e.g., 'Wednesday, August 27, 2025'.")
    research_topic: str = dspy.InputField(desc="A broad area of AI research.")
    problem_statement: str = dspy.OutputField(desc="A specific, well-defined problem within the research topic.")
    context: str = dspy.OutputField(desc="A concise summary of the SOTA and challenges related to this problem.")

# THIS IS THE UPDATED SIGNATURE
class GenerateResearchIdea(dspy.Signature):
    """
    # TASK GOAL
    Describe the landmark, historically significant solution to the given research problem. The solution should be a well-known, foundational concept or paper that is widely recognized for having effectively addressed the challenge.

    # ROLE
    Act as a world-class AI historian and educator, with deep technical knowledge. Your goal is to accurately recount and explain the pivotal solutions to major problems in the history of AI.

    # INSTRUCTIONS AND CONSTRAINTS
    1.  **Identify the Correct Historical Solution:** Your primary task is to identify the famous, widely-accepted solution to the provided problem. For example, for "vanishing gradients in deep networks," the answer is ResNet. For "slow sequential processing in RNNs," the answer is the Transformer.
    2.  **Describe, Do Not Invent:** Do not generate a new or fictional idea. Your entire response must be based on the actual historical solution.
    3.  **Grounding in Reality:** The description must be grounded in the established principles of the actual solution.
    4.  **Mathematical Foundation:** You must explicitly state the mathematical or theoretical concept that underpins the real solution.
    5.  **Actionable Implementation:** Describe the implementation plan as it was originally presented or is commonly understood.
    6.  **Synthesize Knowledge:** Use the provided context to frame your explanation of why this solution was so impactful.

    ---
    # FEW-SHOT EXAMPLES
    ## EXAMPLE 1 OF 3
    problem_statement: "Existing image classification methods based on hand-crafted features have saturated in performance."
    context: "Before 2012, the dominant methods for image classification involved hand-crafting feature extractors like SIFT and HOG..."
    problem_domain: "Computer Vision"
    solution_title: "AlexNet: Deep Convolutional Neural Networks for Scalable Image Recognition"
    core_concept: "Instead of hand-crafting features, we can learn a hierarchy of features directly from pixel data using a deep stack of convolutional layers. By leveraging large labeled datasets like ImageNet and parallelizing training on GPUs, we can train a much larger and deeper network than previously thought possible. Key innovations include using the ReLU activation function for faster training, implementing dropout for regularization, and using overlapping pooling."
    mathematical_foundation: "The core is the convolution operation ($$(f * g)(t) = \\int f(\\tau) g(t - \\tau) d\\tau$$), applied in 2D to learn spatial hierarchies of features. The ReLU activation function, $$f(x) = \\max(0, x)$$, prevents the saturation and vanishing gradient problems seen with sigmoid or tanh units in very deep networks."
    implementation_plan: "Architecture: A stack of 5 convolutional layers followed by 3 fully-connected layers. Use ReLU non-linearity after each convolutional and fully-connected layer. Apply dropout with a probability of 0.5 after the first two fully-connected layers. Train using stochastic gradient descent with momentum."
    verification_path: "Hypothesis: A deep CNN trained on GPUs will significantly outperform all previous methods on the ImageNet LSVRC-2010 contest. Key Metric: Top-5 error rate. Baseline: The previous SOTA which used sparse coding and SIFT features. The goal is to reduce the top-5 error rate from ~26% to below 16%."
    inspiration_cross_domain: "Neuroscience (Inspired by the hierarchical structure of the visual cortex in animals, specifically the work of Hubel and Wiesel)."

    ## EXAMPLE 2 OF 3
    problem_statement: "Increasing the depth of neural networks leads to performance degradation due to optimization difficulties like vanishing gradients."
    context: "As researchers tried to build deeper networks post-AlexNet, they encountered a degradation problem..."
    problem_domain: "Deep Learning Optimization"
    solution_title: "ResNet: Deep Residual Learning for Image Recognition"
    core_concept: "Instead of forcing a stack of layers to learn an underlying mapping $$H(x)$$, we reframe the problem to have the layers learn a residual mapping $$F(x) := H(x) - x$$. The original mapping is then recast as $$H(x) = F(x) + x$$. This is implemented via 'shortcut' or 'skip' connections that bypass one or more layers. The intuition is that it's easier to optimize the residual to zero (if an identity mapping is optimal) than to fit an identity mapping with a stack of non-linear layers."
    mathematical_foundation: "The formulation $$y = F(x, \\{W_i\\}) + x$$ allows gradients to flow directly through the identity connection. During backpropagation, the gradient $$\\frac{\\partial L}{\\partial x}$$ has a term that propagates directly from the deeper layer, ensuring that gradients do not vanish. The chain rule shows $$\\frac{\\partial L}{\\\partial x_l} = \frac{\\partial L}{\\partial x_{l+1}} (1 + \frac{\\partial}{\\partial x_l} F(x_l))$$, which contains a '1' that prevents the product of many small derivatives from shrinking to zero."
    implementation_plan: "Modify standard VGG-style networks by inserting shortcut connections every two convolutional layers. The shortcut connection performs an identity mapping, and its output is added to the output of the stacked layers. If dimensions change, a linear projection (a 1x1 convolution) is used to match the dimensions."
    verification_path: "Hypothesis: A 152-layer residual network will outperform shallower plain networks (like VGG-19) and achieve a new state-of-the-art on ImageNet classification. Key Metric: Top-5 error rate. Baseline: VGGNet, GoogLeNet. The goal is to demonstrate that networks can be made substantially deeper while still improving accuracy."
    inspiration_cross_domain: "Control Theory (Related to the concept of feedback and feedforward systems)."
    """
    problem_statement: str = dspy.InputField()
    context: str = dspy.InputField()
    problem_domain: str = dspy.OutputField(desc="A short string categorizing the field.")
    solution_title: str = dspy.OutputField(desc="The title of the landmark paper or concept.")
    core_concept: str = dspy.OutputField(desc="An elevator pitch of the solution's core idea.")
    mathematical_foundation: str = dspy.OutputField(desc="The mathematical principle the solution is based on.")
    implementation_plan: str = dspy.OutputField(desc="The original or standard implementation of the solution.")
    verification_path: str = dspy.OutputField(desc="The original hypothesis, benchmarks, and metrics that proved the solution's effectiveness.")
    inspiration_cross_domain: str = dspy.OutputField(desc="Any cross-domain inspiration for the solution.")


class ResearchPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.problem_generator = dspy.ChainOfThought(GenerateProblemAndContext)
        self.idea_generator = dspy.ChainOfThought(GenerateResearchIdea)

    def forward(self, research_topic):
        now = datetime.now()
        current_date_str = now.strftime("%A, %B %d, %Y") 
        problem_and_context = self.problem_generator(research_topic=research_topic, current_date=current_date_str)
        
        solution = self.idea_generator(
            problem_statement=problem_and_context.problem_statement,
            context=problem_and_context.context
        )
        
        return dspy.Prediction(
            research_topic=research_topic,
            problem_statement=problem_and_context.problem_statement,
            context=problem_and_context.context,
            # UPDATED to match new field names
            landmark_solution={
                "domain": solution.problem_domain,
                "title": solution.solution_title,
                "concept": solution.core_concept,
                "math_foundation": solution.mathematical_foundation,
                "implementation": solution.implementation_plan,
                "verification": solution.verification_path,
                "inspiration": solution.inspiration_cross_domain
            }
        )


research_pipeline = ResearchPipeline()
topic = "Vanishing gradient problem in very deep networks"
final_output = research_pipeline(research_topic=topic)

output_data = {
    "research_topic": final_output.research_topic,
    "problem_statement": final_output.problem_statement,
    "context": final_output.context,
    "landmark_solution": final_output.landmark_solution
}

import json
print(json.dumps(output_data, indent=4))