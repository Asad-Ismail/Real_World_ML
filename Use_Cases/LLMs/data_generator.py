import dspy
import os
from dotenv import load_dotenv
import json
import time
import yaml


# Load environment variables from .env file
load_dotenv()

# Configure the LLM
# Make sure your LM is configured correctly
lm = dspy.LM("openrouter/moonshotai/kimi-k2:free", api_key=os.getenv("OPENROUTER_API_KEY"), api_base="https://openrouter.ai/api/v1")
dspy.configure(lm=lm)

class DescribeLandmarkSolution(dspy.Signature):
    """
    # TASK GOAL
    Given the name of a famous AI paper or concept, describe the specific problem it solved, the historical context at the time, and a detailed breakdown of its solution.

    # ROLE
    Act as a world-class AI historian and educator with deep technical knowledge. Your goal is to accurately recount and explain pivotal solutions to major problems in the history of AI.

    # INSTRUCTIONS
    1.  Your entire response must be based on the provided 'solution_name'. Do not invent or describe a different concept.
    2.  Accurately formulate the specific 'problem_statement' that this solution was designed to address.
    3.  Describe the 'context' of the research field just before this solution was introduced. What was the previous state-of-the-art and its limitations?
    4.  Provide a detailed, technically accurate breakdown of the solution itself.
    5.  **Formulate a 'simplified_problem_statement' that summarizes the technical problem in a few general words (e.g., 'more efficient sequential modeling', 'avoiding vanishing gradients').**

    ---
    # FEW-SHOT EXAMPLES
    ---

    ## EXAMPLE 1 OF 3
    solution_name: "ResNet (Deep Residual Learning for Image Recognition)"

    # Using your suggested simplified statement and fixing the typo
    simplified_problem_statement: "Training very deep neural networks."
    problem_statement: "Increasing the depth of neural networks leads to performance degradation due to optimization difficulties like vanishing gradients, where deeper models perform worse than their shallower counterparts."
    context: "Following the success of AlexNet, researchers tried building deeper networks (like VGGNet) but encountered a degradation problem. A 56-layer plain network performed worse on ImageNet than a 20-layer one, not due to overfitting, but because gradients could not effectively propagate back to early layers."
    landmark_solution:
    {
        "domain": "Deep Learning Optimization",
        "title": "ResNet: Deep Residual Learning for Image Recognition",
        "concept": "Instead of forcing layers to learn a complete mapping H(x), reframe it so they learn a residual mapping F(x) := H(x) - x. The full mapping is then H(x) = F(x) + x, implemented via 'shortcut' or 'skip' connections that bypass layers. It's easier for layers to learn to output zero (an identity mapping) than to fit an identity with non-linearities.",
        "math_foundation": "The formulation y = F(x) + x allows gradients to flow directly through the identity connection via the chain rule: dL/dx_l = dL/dx_{l+1} * (1 + dF/dx_l). The '1' in this term ensures that gradients can propagate back through many layers without vanishing.",
        "implementation": "Modify standard deep networks by inserting shortcut connections every two convolutional layers. The shortcut performs an identity mapping, and its output is added element-wise to the output of the stacked layers. If dimensions change, a 1x1 convolution is used on the shortcut to match dimensions.",
        "verification": "A 152-layer ResNet substantially outperformed shallower plain networks like VGG-19 on the ImageNet classification task, achieving a 3.57% top-5 error rate and proving that networks could be made much deeper while improving accuracy.",
        "inspiration": "Control Theory (concepts of feedback and feedforward systems)."
    }

    ---
    ## EXAMPLE 2 OF 3
    solution_name: "Attention Is All You Need (Transformer)"

    # Using your suggested simplified statement
    simplified_problem_statement: "Long relationship capturing in sequence modelling which is favorable to parallel computation"
    problem_statement: "Recurrent Neural Networks (RNNs) process sequential data token-by-token, which prevents parallelization, making them slow to train. This sequential nature also makes it difficult to model very long-range dependencies in the data."
    context: "Before 2017, the dominant architecture for sequence-to-sequence tasks like machine translation was the Encoder-Decoder model, which used Recurrent Neural Networks (LSTMs or GRUs). The encoder RNN would process the input sequence and compress all information into a single fixed-size vector (the 'context vector'). The decoder RNN would then use this vector to generate the output sequence. This fixed-size vector was a major bottleneck, making it difficult for the model to handle long sequences and causing it to forget information from the beginning of the input."
    landmark_solution:
    {
        "domain": "Natural Language Processing",
        "title": "Attention Is All You Need",
        "concept": "The model completely discards recurrence and convolutions, relying entirely on a 'self-attention' mechanism. It processes all input tokens in parallel. For each token, self-attention computes a weighted sum of all other tokens in the sequence, where the weights are dynamically calculated based on the compatibility between tokens. This allows the model to directly model relationships between any two tokens in the sequence, regardless of their distance.",
        "math_foundation": "The core operation is Scaled Dot-Product Attention, defined as: $$Attention(Q, K, V) = softmax(\\frac{QK^T}{\\sqrt{d_k}})V$$. Queries (Q), Keys (K), and Values (V) are projections of the input. The softmax creates a weighted distribution over the values, and the scaling factor $$\\sqrt{d_k}$$ prevents vanishing gradients in the softmax function for large inner products.",
        "implementation": "The architecture is a stack of identical layers, each containing a Multi-Head Self-Attention sub-layer and a simple Feed-Forward Network sub-layer. Residual connections and layer normalization are used around each sub-layer. Since there is no recurrence, positional encodings are added to the input embeddings to give the model information about token order.",
        "verification": "The Transformer model set a new state-of-the-art in machine translation on the WMT 2014 English-to-German and English-to-French tasks, achieving higher BLEU scores while training in a fraction of the time required by previous SOTA models like Google's GNMT.",
        "inspiration": "None (Represents a novel paradigm shift within sequence modeling)."
    }

    ---
    ## EXAMPLE 3 OF 3
    solution_name: "GAN (Generative Adversarial Networks)"

    simplified_problem_statement: "High-fidelity generative image modeling."
    problem_statement: "Training generative models to produce sharp, realistic samples was challenging. Models that maximized likelihood often required intractable partition functions or approximated them, while others like VAEs tended to produce blurry, overly smooth results."
    context: "Prior to 2014, prominent generative models included Variational Autoencoders (VAEs) and autoregressive models. These methods typically worked by maximizing the log-likelihood of the data, or a lower bound of it. This objective often led to models that were difficult to train, computationally expensive, or produced samples that were blurry and lacked the sharp details of real-world data, as optimizing for likelihood does not necessarily mean optimizing for perceptual quality."
    landmark_solution:
    {
        "domain": "Generative Modeling",
        "title": "Generative Adversarial Networks",
        "concept": "A framework for training generative models via a two-player minimax game. A 'Generator' network (G) learns to create plausible data from random noise. A 'Discriminator' network (D) learns to distinguish between real data from the training set and 'fake' data from the Generator. G's goal is to fool D, and D's goal is to not be fooled. As they compete, both networks improve, and G learns to produce highly realistic samples.",
        "math_foundation": "The training is framed as a minimax game with the value function: $$\\min_G \\max_D V(D, G) = \\mathbb{E}_{x \\sim p_{data}(x)}[\\log D(x)] + \\mathbb{E}_{z \\sim p_z(z)}[\\log(1 - D(G(z)))]$$ The Discriminator (D) tries to maximize this value, while the Generator (G) tries to minimize it.",
        "implementation": "Two neural networks are defined. The Generator takes a random noise vector `z` as input and outputs a data sample (e.g., an image). The Discriminator takes a data sample as input and outputs a single scalar probability that the sample is real. Training alternates between updating D for a few steps on real and fake data, and then updating G to better fool D.",
        "verification": "The original paper demonstrated that GANs could generate sharp, visually convincing images on datasets like MNIST, TFD, and CIFAR-10, which were qualitatively superior to samples from other contemporary generative models. The primary verification was the visual fidelity of the generated samples.",
        "inspiration": "Game Theory (specifically, the concept of a zero-sum, minimax game)."
    }

    """
    solution_name: str = dspy.InputField(desc="The name of the landmark AI paper or concept.")
    
    simplified_problem_statement: str = dspy.OutputField(desc="A short, high-level summary of the problem, like a research area.")
    problem_statement: str = dspy.OutputField()
    context: str = dspy.OutputField()
    landmark_solution: dict = dspy.OutputField(desc="A dictionary containing the detailed breakdown of the solution.")


generate_datapoint = dspy.Predict(DescribeLandmarkSolution)

with open("/home/asad/dev/Real_World_ML/Use_Cases/LLMs/dl_research.yaml") as f:
    data = yaml.safe_load(f)

landmark_solutions_to_document = data["landmark_solutions_to_document"]

# Create a directory to save the outputs
output_dir = "historical_ai_landmarks"
os.makedirs(output_dir, exist_ok=True)

for i, solution_name in enumerate(landmark_solutions_to_document,start=71):
    print(f"[{i+1}/{len(landmark_solutions_to_document)}] Documenting: '{solution_name}'...")
    try:
        start_time = time.monotonic()
        # Run the generation
        result = generate_datapoint(solution_name=solution_name)

        # Structure and save the data
        output_data = {
            "solution_name": solution_name,
            "simplified_problem": result.simplified_problem_statement, 
            "problem_it_solved": result.problem_statement,
            "historical_context": result.context,
            "landmark_solution_details": result.landmark_solution
        }
        end_time = time.monotonic()
        print(f"Time to process one query {end_time-start_time}s")

        print(output_data)
        #break
        file_name = f"solution_{i+1}_{solution_name.split('(')[0].strip().replace(' ', '_').lower()}.json"
        file_path = os.path.join(output_dir, file_name)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            # The landmark_solution is already a dict, so we can dump directly
            json.dump(output_data, f, indent=4)
            
        print(f"  -> Successfully documented and saved to {file_path}")

    except Exception as e:
        print(f"  -> An error occurred while processing '{solution_name}': {e}")

print("\nDataset generation complete.")