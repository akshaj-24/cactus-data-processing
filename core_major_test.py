import random
import pandas as pd
from faker import Faker
import re
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from ollama import Client
import traceback
import json
import sys

fake = Faker('en_CA')

df = pd.read_json("hf://datasets/Psychotherapy-LLM/CBT-Bench/core_major_test.json")
print(df.shape)

df.drop(columns=["id"], inplace=True)
df.rename(columns={"ori_text": "patient_context",
                   "thoughts": "automatic_thoughts",
                   "core_belief_major": "core_beliefs"}, inplace=True)

# Add columns for:
# ID 200 + n
df["patient_ID"] = df.index + 10

model1 = OllamaLLM(base_url="localhost:11434", model="qwen3:32b", temperature=0)

def call_llm(prompt: str) -> str:
    msgs = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that extracts and categorizes information accurately according to the instructions provided by the user."),
        ("user", prompt)
    ]).format_messages()

    response = model1.invoke(msgs)
    
    # print(response)
    return response

error_log_path = "core_fine_seed_errors.log"

# Gender: LLM call to suggest an appropriate gender based on context
df["patient_gender"] = None

def assign_gender(row):
    context = row.get("patient_context")
    prompt = f"""Based on the following context, determine the most appropriate gender for the patient. Output 'M' for male and 'F' for female. Only output the gender letter and nothing else.
    Context: {context}
    --------------------------------------------
    Output Gender:"""
    try:
        llm_response = call_llm(prompt)
        # Extract everything after </think>
        match = re.search(r"</think>\s*\n(.*)", llm_response, re.DOTALL)
        if match:
            result = match.group(1).strip()
        else:
            result = ""

        result = result.strip('\'"')
    except Exception as e:
        with open(error_log_path, "a") as f:
            f.write(f"Error in row {row.name}:\n{traceback.format_exc()}\n\n")
        result = "m"  # default to male
        
        
    print(f"processed row index: {row.name} of {df.shape[0]}, assigned gender: {result}")
    return result
    

# Name random name generator
df["patient_name"] = None

def random_name(row):
    gender = row.get("patient_gender")
    gender = gender.lower()
    return fake.first_name_male() if gender=="m" else fake.first_name_female()

# df["patient_name"] = df.apply(lambda row: random_name(row), axis=1)

# Age: LLM call to suggest an appropriate age based on context
df["patient_age"] = None

def assign_age(row):
    context = row.get("patient_context")
    prompt = f"""Based on the following context, determine the most appropriate age for the patient. Output only the age as a number and nothing else.
    Context: {context}
    --------------------------------------------
    Output Age:"""
    try:
        llm_response = call_llm(prompt)
        # Extract everything after </think>
        match = re.search(r"</think>\s*\n(.*)", llm_response, re.DOTALL)
        if match:
            result = match.group(1).strip()
        else:
            result = ""

        result = result.strip('\'"')
    except Exception as e:
        with open(error_log_path, "a") as f:
            f.write(f"Error in row {row.name}:\n{traceback.format_exc()}\n\n")
        result = "0"  # default to 0
        
        
    print(f"processed row index: {row.name} of {df.shape[0]}, assigned age: {result}")
    return result


# Marital Status: LLM call to suggest an appropriate marital status based on context
df["patient_marital_status"] = None

def assign_marital_status(row):
    context = row.get("patient_context")
    prompt = f"""Based on the following context, determine the most appropriate marital status for the patient. Output single, married, common-law, divorced, widowed, or separated.
    Output only the marital status as a string and nothing else.
    Context: {context}
    --------------------------------------------
    Output Marital Status:"""
    try:
        llm_response = call_llm(prompt)
        # Extract everything after </think>
        match = re.search(r"</think>\s*\n(.*)", llm_response, re.DOTALL)
        if match:
            result = match.group(1).strip()
        else:
            result = ""

        result = result.strip('\'"')
    except Exception as e:
        with open(error_log_path, "a") as f:
            f.write(f"Error in row {row.name}:\n{traceback.format_exc()}\n\n")
        result = "not specified"  # default to not specified
        
        
    print(f"processed row index: {row.name} of {df.shape[0]}, assigned marital status: {result}")
    return result

# Occupation: LLM call to check if context contains occupation, or NULL
df["patient_occupation"] = None

def assign_age(row):
    context = row.get("patient_context")
    prompt = f"""Based on the following context, determine if the occupation is mentioned. Output only the occupation as a string or NULL if not mentioned.
    Context: {context}
    --------------------------------------------
    Output Occupation:"""
    try:
        llm_response = call_llm(prompt)
        # Extract everything after </think>
        match = re.search(r"</think>\s*\n(.*)", llm_response, re.DOTALL)
        if match:
            result = match.group(1).strip()
        else:
            result = ""

        result = result.strip('\'"')
    except Exception as e:
        with open(error_log_path, "a") as f:
            f.write(f"Error in row {row.name}:\n{traceback.format_exc()}\n\n")
        result = "NULL"  # default to NULL
        
    print(f"processed row index: {row.name} of {df.shape[0]}, assigned occupation: {result}")
    return result

# Education: LLM call to check if context contains education, or NULL
df["patient_education"] = None

def assign_education(row):
    context = row.get("patient_context")
    prompt = f"""Based on the following context, determine if the education is mentioned. Output only the education as a string or NULL if not mentioned.
    Context: {context}
    --------------------------------------------
    Output Education:"""
    try:
        llm_response = call_llm(prompt)
        # Extract everything after </think>
        match = re.search(r"</think>\s*\n(.*)", llm_response, re.DOTALL)
        if match:
            result = match.group(1).strip()
        else:
            result = ""

        result = result.strip('\'"')
    except Exception as e:
        with open(error_log_path, "a") as f:
            f.write(f"Error in row {row.name}:\n{traceback.format_exc()}\n\n")
        result = "NULL"  # default to NULL
        
        
    print(f"processed row index: {row.name} of {df.shape[0]}, assigned education: {result}")
    return result

# Context: Rename from ori_text
# Done above

# Situation
# In dataset

# Core Beliefs
# In dataset

# Automatic thoughts
# In dataset

# Attitude
df["attitude"] = None

def assign_attitude(row):
    return random.choice(["positive", "negative", "neutral"])

# df["attitude"] = df.apply(lambda row: assign_attitude(row), axis=1)

# Intermediate Thoughts: LLM generated
df["intermediate_beliefs"] = None

def intermediate_beliefs(row):
    prompt = (
        f"Patient story:\n{row.get('patient_context', '')}\n\n"
        f"Core beliefs:\n{row.get('core_beliefs', '')}\n\n"
        f"Situation:\n{row.get('situation', '')}\n\n"
        f"Automatic thoughts:\n{row.get('automatic_thoughts', '')}\n\n"
        f"Attitude:\n{row.get('attitude', '')}\n\n"
        "Based on the above, generate a concise intermediate belief consistent with the thought and core beliefs to construct a cognitive model of the patient. "
        "Only output the intermediate beliefs as a string, nothing else."
    )
    llm_response = call_llm(prompt)
    # Extract everything after </think>
    match = re.search(r"</think>\s*\n(.*)", llm_response, re.DOTALL)
    if match:
        result = match.group(1).strip()
    else:
        result = ""

    result = result.strip('\'"')
    print(result)
    return result

# Conversation Styles: LLM generated
df["conversation_style"] = None

model1.temperature=0.6
model1.top_p=0.95
model1.top_k=20

def conversational_styles(row):
    prompt = f"""
    You are labeling "conversational style" for a therapy intake transcript.

DEFINITION:
Conversational style = HOW the patient speaks (tone, structure, interaction pattern), not the diagnosis and not the cognitive distortion names.
Choose 3-7 styles from the allowed list. Do not invent new styles.

ALLOWED STYLES (choose only from these):
anxious, guilty, self-blaming, ruminative, avoidant, reassurance-seeking, shame-based, socially vigilant, sensitive-to-rejection,
catastrophic-framing, pessimistic, rigid-all-or-nothing, overexplaining, detail-heavy, hesitant, guarded, emotionally intense,
hopeless, low-energy, tearful, self-critical, people-pleasing, conflict-avoidant

INPUT:
Patient story:
{row.get('patient_context', '')}

Core beliefs:
{row.get('core_beliefs', '')}

Automatic thoughts:
{row.get('automatic_thoughts', '')}

Age: {row.get('patient_age', '')}

TASK:
Return only a comma-separated list of 3-7 styles from the allowed list.
No extra words, no quotes, no punctuation except commas.

OUTPUT:
<styles>style1, style2, style3</styles>
    """
    llm_response = call_llm(prompt)
    # Extract everything after </think>
    match = re.search(r"</think>\s*\n(.*)", llm_response, re.DOTALL)
    if match:
        result = match.group(1).strip()
    else:
        result = ""

    result = result.strip('\'"')
    styles_list = [s.strip() for s in result.split(",") if s.strip()]
    return styles_list


# Run all functions
jsonl_path = "core_fine_test_results.jsonl"
csv_path = "core_fine_test_results.csv"
xlsx_path = "core_fine_test_results.xlsx"
log_path = "core_fine_test_progress.txt"

class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()
        
        
sys.stdout = Logger(log_path)

results = []

for idx, row in df.iterrows():
    # --- Your LLM and assignment functions here ---
    row["patient_gender"] = assign_gender(row)
    row["patient_name"] = random_name(row)
    row["patient_age"] = assign_age(row)
    row["patient_marital_status"] = assign_marital_status(row)
    row["patient_occupation"] = assign_age(row)
    row["patient_education"] = assign_education(row)
    row["attitude"] = assign_attitude(row)
    row["intermediate_beliefs"] = intermediate_beliefs(row)
    row["conversation_style"] = conversational_styles(row)

    row_dict = row.to_dict()
    results.append(row_dict)

    # --- Save to JSONL ---
    with open(jsonl_path, "a") as f_jsonl:
        f_jsonl.write(json.dumps(row_dict) + "\n")

    # --- Save to CSV ---
    pd.DataFrame([row_dict]).to_csv(csv_path, mode="a", header=(idx==0), index=False)

    # # --- Save to XLSX (overwrite each time) ---
    # pd.DataFrame(results).to_excel(xlsx_path, index=False)

    print(f"Processed row {idx+1}/{len(df)}: {row_dict.get('patient_ID', '')}")

print("Processing complete.")
