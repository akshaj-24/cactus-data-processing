from ollama import Client
import re
import sys
import pandas as pd
# from langchain_ollama import OllamaLLM
# from langchain_core.prompts import ChatPromptTemplate
from ollama import Client
from ollama import AsyncClient
import traceback
import json
import asyncio
import requests


# Initialize the client with your custom base URL
client = AsyncClient(host="http://localhost:11434")

MODEL = "qwen3:32b-copy"

# Define your custom generation parameters
options = {
    "temperature": 0.4,
    "top_p": 0.9,
    "top_k": 40
}

# Simple ASCII: tab/newline + printable ASCII (0x20-0x7E)
ascii_schema = {
  "type": "object",
  "properties": {
    "text": {
      "type": "string",
      "pattern": r"^[\t\n -~]*$"
    }
  },
  "required": ["text"],
  "additionalProperties": False
}


# load cactus dataset
df = pd.read_json("hf://datasets/LangAGI-Lab/cactus/cactus.json")

# Drop unnecessary columns
cols_to_drop = ["cbt_technique", "cbt_plan", "dialogue", "attitude"]
df.drop(columns=cols_to_drop, inplace=True)

error_log_path = "error_log.txt"

# subset selection
df = df.drop_duplicates(subset=df.columns.difference(["patterns"]), keep="first")


df["conversational_styles"]=None #LLM
df["patient_ID"]=df.index+10000 #auto assign from 10000
df["patient_name"]=None #regex
df["patient_age"]=None #regex
df["patient_gender"]=None #regex
df["patient_marital_status"]=None #regex
df["patient_education"]=None # regex + LLM categorization
df["patient_occupation"]=None # regex + LLM categorization
df["intermediate_beliefs"]=None # LLM
df["situation"]=None # LLM

#print(df.columns)
df.rename(columns={"patterns":"core_beliefs",
                   "thought":"automatic_thoughts",
                   "intake_form": "patient_context"}, inplace=True)

desired_order = [
    "patient_ID",
    "patient_name",
    "patient_age",
    "patient_gender",
    "patient_marital_status",
    "patient_education",
    "patient_occupation",
    "patient_context",
    "conversational_styles",
    "core_beliefs",
    "situation",
    "intermediate_beliefs",
    "automatic_thoughts",
]

#df = df.reindex(columns=desired_order)

#print(df)


def adjust_context(row):
    context = str(row["patient_context"])
    data = {}

    # Define simple field mappings: (key_name, regex_pattern, type_cast_func)
    fields = [
        ("patient_name", r"Name:\s*([^\n]+)", str),
        ("patient_age", r"Age:\s*(\d+)", int),
        ("patient_gender", r"Gender:\s*([^\n]+)", str),
        ("patient_marital_status", r"Marital Status:\s*([^\n]+)", str),
        ("patient_education", r"Education:\s*([^\n]+)", str),
        ("patient_occupation", r"Occupation:\s*([^\n]+)", str),
    ]

    for key, pattern, cast in fields:
        match = re.search(pattern, context)
        # Extract and cast value
        val = match.group(1).strip() if match else None
        data[key] = cast(val) if val and cast else val
        # Remove the line from context
        context = re.sub(pattern + r"\n?", "", context)

    # Specific Data Normalization
    # Gender mapping
    gender_map = {"male": "m", "female": "f"}
    data["patient_gender"] = gender_map.get(str(data["patient_gender"]).lower())

    # Marital Status mapping
    marital_map = {"single": "single", "married": "married"}
    data["patient_marital_status"] = marital_map.get(
        str(data["patient_marital_status"]).lower(), "not specified"
    )

    # Clean up remaining headers/sections in one regex pass
    sections_to_remove = [
        r"Family Details:\s*[^\n]+\n?",
        r"\d\.\s*(Presenting Problem|Reason for Seeking Counseling|Past History|Academic/occupational functioning level|Social Support System)[^:]*:?"
    ]
    for section in sections_to_remove:
        context = re.sub(section, "", context, flags=re.IGNORECASE | re.DOTALL)

    data["patient_context"] = context.strip()
    return pd.Series(data)

adjusted_cols = [
    "patient_name", "patient_age", "patient_gender",
    "patient_marital_status", "patient_education",
    "patient_occupation", "patient_context"
]

df[adjusted_cols] = df.apply(adjust_context, axis=1)

print("Adjusted patient context and extracted demographics.")



async def call_llm(prompt: str) -> str:
    response = await client.chat(
        model="qwen3:32b",
        messages=[
            {"role": "system",
             "content": "You are an assistant that follows instructions. Return ONLY JSON matching the schema."},
            {"role": "user", "content": prompt},
        ],
        format=ascii_schema,  # Ensures the model outputs valid JSON per your schema
        stream=False,
        options={
            "temperature": 0.4,
            "top_p": 0.9,
            "top_k": 40
        }
    )

    # Parse the content string back into a dictionary
    content = response['message']['content']
    # obj = json.loads(content)
    #
    # return obj["text"]
    try:
        # Standard load attempt
        obj = json.loads(content)
    except json.JSONDecodeError:
        # FALLBACK: Extract the first JSON object using regex
        # This finds the content between the first '{' and the last '}'
        match = re.search(r'(\{.*\})', content, re.DOTALL)
        if match:
            try:
                obj = json.loads(match.group(1))
            except json.JSONDecodeError:
                # If it still fails, it's truly malformed
                print(f"FAILED TO PARSE: {content}")
                return "Parsing Error"
        else:
            return "No JSON found"

    return obj.get("text", "Key 'text' not found")

def get_situation_prompt(row):
    return (
        f"Core beliefs:\n{row.get('core_beliefs', '')}\n\n"
        f"Automatic thoughts:\n{row.get('automatic_thoughts', '')}\n\n"
        "Based on the above, extract a concise, factual description of the situation as experienced by the patient. "
        "Do not include emotions, judgments, or interpretations. Only describe what happened. "
        "Write the situation in first person as the patient. "
        "Only output the situation as a string, nothing else."
    )

def get_intermediate_beliefs_prompt(row):
    return (
        f"Patient story:\n{row.get('patient_context', '')}\n\n"
        f"Core beliefs:\n{row.get('core_beliefs', '')}\n\n"
        f"Situation:\n{row.get('situation', '')}\n\n"
        f"Automatic thoughts:\n{row.get('automatic_thoughts', '')}\n\n"
        "Based on the above, generate a concise intermediate belief consistent with the thought and core beliefs "
        "to construct a cognitive model of the patient. "
        "Only output the intermediate beliefs as a string, nothing else."
    )

def get_conversational_styles_prompt(row):
    return f"""
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


async def process_patient_row(idx, row):
    # Step 1: Get the Situation
    sit_prompt = get_situation_prompt(row)
    #print(sit_prompt)
    situation_res = await call_llm(sit_prompt)
    print(situation_res)
    #row['situation'] = situation_res  # Update row for the next prompt

    # Step 2: Get Intermediate Beliefs (uses the situation generated above)
    ib_prompt = get_intermediate_beliefs_prompt(row)
    #print(ib_prompt)
    belief_res = await call_llm(ib_prompt)
    print(belief_res)
    #row['intermediate_beliefs'] = belief_res

    # Step 3: Get Styles
    styles_prompt = get_conversational_styles_prompt(row)
    styles_res = await call_llm(styles_prompt)
    print(styles_res)

    # Return a dictionary containing the results and the original index for tracking
    return {
        "index": idx,
        "situation": situation_res,
        "intermediate_beliefs": belief_res,
        "conversational_styles": styles_res
    }


async def main():
    # This creates a list of coroutines, one for each row in your DataFrame
    tasks = [process_patient_row(idx, row) for idx, row in df.iterrows()]

    print(f"Starting parallel processing for {len(df)} rows...")

    # 3. Gather preserves the original order of the 'tasks' list
    results = await asyncio.gather(*tasks)

    # 1. Convert the list of dictionaries into a new DataFrame
    results_df = pd.DataFrame(results)

    # 2. Save to CSV (index=False prevents an extra index column)
    results_df.to_csv("patient_processing_results.csv", index=False)

    print(f"Successfully saved {len(results_df)} results to patient_processing_results.csv")


if __name__ == "__main__":
    print("starting")
    asyncio.run(main())


# try:
#     print("calling")
#     result_text = call_llm(my_prompt)
#
#     # 3. Print the output
#     print("Model Output:")
#     print(result_text)
# except Exception as e:
#     print(f"An error occurred: {e}")

# Use the client to start the chat
# stream = client.chat(
#     model=MODEL,
#     messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
#     stream=True,
#     options=options # Pass the parameters here
# )
# for chunk in stream:
#     print(chunk['message']['content'], end='', flush=True)
