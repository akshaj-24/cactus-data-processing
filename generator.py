import re
import sys
import pandas as pd
# from langchain_ollama import OllamaLLM
# from langchain_core.prompts import ChatPromptTemplate
from ollama import Client
import traceback
import json
import requests
import aiohttp
import asyncio
from typing import NamedTuple

class LLMConfig(NamedTuple):
    temperature: float
    top_p: float
    top_k: int


client = Client(host='http://localhost:11434')

default_config = LLMConfig(temperature=0.4,
    top_p = 0.9,
    top_k = 40)

conversation_config = LLMConfig(temperature=0.4,
    top_p = 0.9,
    top_k = 40)

#base parameters
BASE_URL = "http://localhost:11434"
MODEL = "qwen3:32b"

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

async def call_llm(session, prompt: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system",
             "content": "You are an assistant that follows the instructions given by the user. Return ONLY JSON matching the schema."},
            {"role": "user", "content": prompt},
        ],
        "format": ascii_schema,
        "stream": False,
        "options": {"temperature": default_config.temperature,
                    "top_p": default_config.top_p,
                    "top_k": default_config.top_k},
    }

    async with session.post(f"{BASE_URL}/api/chat", json=payload, timeout=120) as resp:
        resp.raise_for_status()
        data = await resp.json()

    content = data["message"]["content"]
    obj = json.loads(content)
    return obj["text"]


async def situation(session, row):
    prompt = (
        # f"Patient story:\n{row.get('patient_context', '')}\n\n"
        f"Core beliefs:\n{row.get('core_beliefs', '')}\n\n"
        f"Automatic thoughts:\n{row.get('automatic_thoughts', '')}\n\n"
        "Based on the above, extract a concise, factual description of the situation as experienced by the patient. Do not include emotions, judgments, or interpretations. Only describe what happened. Write the situation in first person as the patient. "
        "Only output the situation as a string, nothing else."
    )
    llm_response = await call_llm(session, prompt)
    if llm_response:
        result = llm_response.strip()
    else:
        result = ""

    result = result.strip('\'"')
    print(result)
    return result

async def intermediate_beliefs(session, row):
    prompt = (
        f"Patient story:\n{row.get('patient_context', '')}\n\n"
        f"Core beliefs:\n{row.get('core_beliefs', '')}\n\n"
        f"Situation:\n{row.get('situation', '')}\n\n"
        f"Automatic thoughts:\n{row.get('automatic_thoughts', '')}\n\n"
        "Based on the above, generate a concise intermediate belief consistent with the thought and core beliefs to construct a cognitive model of the patient. "
        "Only output the intermediate beliefs as a string, nothing else."
    )
    llm_response = await call_llm(session, prompt)
    if llm_response:
        result = llm_response.strip()
    else:
        result = ""

    result = result.strip('\'"')
    print(result)
    return result

async def conversational_styles(session, row):
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
    llm_response = await call_llm(session, prompt)
    if llm_response:
        result = llm_response.strip()
    else:
        result = ""

    result = result.strip('\'"')
    styles_list = [s.strip() for s in result.split(",") if s.strip()]
    print(styles_list)
    return styles_list


# async def func_call(func, session, row):
#     try:
#         return await func(session, row)
#     except Exception as e:
#         with open(error_log_path, "a") as f:
#             f.write(f"Error in row {row.name}:\n{traceback.format_exc()}\n\n")
#         return None

async def func_call(func, session, row, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            result = await func(session, row)

            # Ensure the result is a non-empty string
            if result and str(result).strip():
                return result

            print(f"Empty result for {func.__name__} at row {row.name}. Retry {attempt}/{max_retries}")

        except Exception as e:
            error_msg = f"Error in row {row.name}, {func.__name__} (Attempt {attempt}):\n{traceback.format_exc()}\n\n"
            with open(error_log_path, "a") as f:
                f.write(error_msg)
            print(f"Exception in {func.__name__}. Retrying...")

        # Optional: Add a small delay between retries to let the LLM/connection reset
        await asyncio.sleep(1)

    return "FAILED_TO_GENERATE"  # Final sentinel value if all retries fail


# Run all functions
jsonl_path = "cactus_results.jsonl"
csv_path = "cactus_results.csv"
log_path = "cactus_progress.txt"

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


async def main():
    async with aiohttp.ClientSession() as session:
        for idx, row in df.iterrows():
            # --- Your LLM and assignment functions here ---

            row["situation"] = await func_call(situation, session, row)
            row["intermediate_beliefs"] = await func_call(intermediate_beliefs, session, row)
            row["conversational_styles"] = await func_call(conversational_styles, session, row)

            row_dict = row.to_dict()
            results.append(row_dict)

            # --- Save to JSONL ---
            with open(jsonl_path, "a") as f_jsonl:
                f_jsonl.write(json.dumps(row_dict) + "\n")

            # --- Save to CSV ---
            pd.DataFrame([row_dict]).to_csv(csv_path, mode="a", header=(idx == 0), index=False)

            print(f"Processed row {row.name}/{len(df)}: {row_dict.get('patient_ID', '')}")


asyncio.run(main())

print("Processing complete.")

df2 = df.iloc[:30, :]
df2.to_excel("cactus_processed_30_samples.xlsx", index=False)
