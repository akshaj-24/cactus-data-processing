import re
import sys
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from ollama import Client
import traceback
import json
import requests
import aiohttp
import asyncio

client = Client(host='http://localhost:11434')

df = pd.read_json("hf://datasets/LangAGI-Lab/cactus/cactus.json")

df.drop(columns=["cbt_technique"], inplace=True)
df.drop(columns=["cbt_plan"], inplace=True)
df.drop(columns=["dialogue"], inplace=True)
df.drop(columns=["attitude"], inplace=True)

cols_except = [col for col in df.columns if col not in ["patterns"]]
df = df.drop_duplicates(subset=cols_except, keep="first")


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

df = df.reindex(columns=desired_order)

def adjust_context(row):
    context = row["patient_context"]

    # Name
    name_match = re.search(r"Name:\s*([^\n]+)", str(context))
    name = name_match.group(1).strip() if name_match else None
    context = re.sub(r"Name:\s*[^\n]+\n?", "", str(context))

    # Age
    age_match = re.search(r"Age:\s*(\d+)", context)
    age = int(age_match.group(1)) if age_match else None
    context = re.sub(r"Age:\s*\d+\n?", "", context)

    # Gender
    gender_match = re.search(r"Gender:\s*([^\n]+)", context)
    gender_raw = gender_match.group(1).strip().lower() if gender_match else None
    if gender_raw == "male":
        gender = "m"
    elif gender_raw == "female":
        gender = "f"
    else:
        gender = None
    context = re.sub(r"Gender:\s*[^\n]+\n?", "", context)

    # Marital Status
    marital_match = re.search(r"Marital Status:\s*([^\n]+)", context)
    marital_raw = marital_match.group(1).strip().lower() if marital_match else None
    if marital_raw == "single":
        marital_status = "single"
    elif marital_raw == "married":
        marital_status = "married"
    else:
        marital_status = "not specified"
    
    context = re.sub(r"Marital Status:\s*[^\n]+\n?", "", context)
    context = re.sub(r"Family Details:\s*[^\n]+\n?", "", context)
    context = re.sub(r"2\. Presenting Problem\s*", "", context, flags=re.DOTALL)
    context = re.sub(r"3\. Reason for Seeking Counseling\s*", "", context, flags=re.DOTALL)
    context = re.sub(r"4\. Past History \(including medical history\)\s*", "", context, flags=re.DOTALL)
    context = re.sub(r"5\. Academic/occupational functioning level:?\s*", "", context, flags=re.DOTALL)
    context = re.sub(r"6\. Social Support System\s*", "", context, flags=re.DOTALL)
    
    # Patient living details
    education = re.search(r"Education:\s*([^\n]+)", context)
    education = education.group(1).strip() if education else None
    context = re.sub(r"Education:\s*[^\n]+\n?", "", context)
    
    # Patient living details
    occupation = re.search(r"Occupation:\s*([^\n]+)", context)
    occupation = occupation.group(1).strip() if occupation else None
    context = re.sub(r"Occupation:\s*[^\n]+\n?", "", context)
    

    return pd.Series({
        "patient_name": name,
        "patient_age": age,
        "patient_gender": gender,
        "patient_marital_status": marital_status,
        "patient_education": education,
        "patient_occupation": occupation,
        "patient_context": context.strip()
    })
    
    
df[["patient_name",
    "patient_age",
    "patient_gender",
    "patient_marital_status",
    "patient_education",
    "patient_occupation",
    "patient_context"]] = df.apply(adjust_context, axis=1)

print("Adjusted patient context and extracted demographics.")

BASE_URL = "http://localhost:11434"
MODEL = "qwen3:32b"
temperature = 0.4
top_p = 0.9
top_k = 40

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

async def call_llm(session, prompt: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are an assistant that follows the instructions given by the user. Return ONLY JSON matching the schema."},
            {"role": "user", "content": prompt},
        ],
        "format": ascii_schema,
        "stream": False,
        "options": {"temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k},
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

temperature=0.6
top_p=0.95
top_k=20

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

error_log_path = "error_log.txt"

async def func_call(func, session, row):
    try:
        return await func(session, row)
    except Exception as e:
        with open(error_log_path, "a") as f:
            f.write(f"Error in row {row.name}:\n{traceback.format_exc()}\n\n")
        return None
    

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
            pd.DataFrame([row_dict]).to_csv(csv_path, mode="a", header=(idx==0), index=False)

            print(f"Processed row {row.name}/{len(df)}: {row_dict.get('patient_ID', '')}")

asyncio.run(main())

print("Processing complete.")

df2 = df.iloc[:30, :]
df2.to_excel("cactus_processed_30_samples.xlsx", index=False)