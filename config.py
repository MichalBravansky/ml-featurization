import json
import pandas as pd
from openai import OpenAI


#____USER CONFIGURABLE SECTION (EDITABLE)____

API_KEY = ""

def get_prompt_template(row, features):
    """
    Row is a row from the dataframe used to initialize the program, while features is a string containing the features to be appended to the instructions, which should always go at the end of the user message.
    
    Args:
    row (pd.Series): A row from the dataframe.
    features (str): A string containing the features to be appended to the user message.

    Returns:
    tuple: A tuple containing:
        - system_message (str): The system's objective.
        - user_message (str): The user's message with instructions and features.
        - assistant_message (str): The assistant's response, which perplexity will be calculated over.
    """

    features = features.replace("The selected string", "The new response")

    response_chain = "\n".join([f"Instruction: {message}" if i%2 == 0 else f"Response: {message}" for i, message in enumerate(row["instruction"])])
    
    system_message = "Your objective is to provide a response to the last instruction."
    user_message = f"{response_chain}\n\n---\n\nProvide only the response to the last instruction, ensuring it follows the rules below.{features}"
    
    assistant_message = f"Response: {row['string']}"

    return system_message, user_message, assistant_message


WANDB_PROJECT = "vectorization"
WANDB_USERNAME = "featurization"
WANDB_MODE = "disabled"


#____ADDITIONAL LOGIC SECTION (NO NEED TO EDIT)____

MODEL="gpt-4o"

GENERATOR_SYSTEM_PROMPT = """Your job is to analyze data and propose unique, creative features."""

GENERATOR_USER_PROMPT = """Consider the following strings:
{strings}

Now, specifically analyze this selected string:
{selected_string}

Based on the strings above, identify five unique features that differentiate the selected text from the others. Provide specific reasons in ten words or fewer, focusing on content, structure, subjects, or writing style.
If the selected text is a single piece, describe the features that set it apart. If it’s a pair, highlight unique aspects of their connection or contrast compared to the rest.
Always suggest as feature that starts with 'The selected string...'.

Reply as a JSON similar to: {{"feature": ["<YOUR FEATURE TEXT>", "<YOUR NEXT FEATURE TEXT>", ...]}}.
DO NOT respond with any text apart from the JSON format above!
DO NOT add markdown formatting around JSON.
ONLY REPLY IN JSON FORMAT."""

VERIFICATION_SYSTEM_PROMPT = """You are an expert at identifying features in given strings."""

VERIFICATION_USER_PROMPT = """Selected string: {text}

Given the string above, check whether it satisfies any of the features below. Ensure the classification is accurate and consistent with each feature description.
{features}

Answer in JSON format, e.g., {{"0": "Y", "1": "N", ...}}.
Put "Y" if the string satisfies the feature and "N" if it does not.
No ties are allowed; only one of "Y" or "N".
Vote for all features, even if you are unsure.
DO NOT respond with any text apart from the JSON format above!
DO NOT add markdown formatting around the JSON.
ONLY REPLY IN JSON FORMAT."""

client = OpenAI(
    api_key = API_KEY
)

NUM_EXAMPLE_SAMPLED = 5

CLUSTER_SIZE = 5

VERIFICATION_SPLIT = 10

SEED = 0