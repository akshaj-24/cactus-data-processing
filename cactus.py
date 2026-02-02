import re
import pandas as pd
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from ollama import Client
import traceback
import json
import requests
import os


client = Client(host='http://localhost:11434')


df = pd.read_json("hf://datasets/LangAGI-Lab/cactus/cactus.json")


df.drop(columns=["cbt_technique"], inplace=True)
df.drop(columns=["cbt_plan"], inplace=True)
df.drop(columns=["dialogue"], inplace=True)
df.drop(columns=["attitude"], inplace=True)
df["conversational_styles"]=None #LLM
df["patient_ID"]=None
df["patient_name"]=None #regex
df["patient_age"]=None #regex
df["patient_gender"]=None #regex
df["patient_marital_status"]=None #regex
df["patient_education"]=None # regex + LLM categorization
df["patient_occupation"]=None # regex + LLM categorization
df["intermediate_beliefs"]=None # LLM
df["situation"]=None # LLM
# df["patient_living_details"]=None #regex
# df["past_history"]=None #LLM
# df["reason_for_seeking_help"]=None #LLM
# df["functioning_level"]=None #LLM
# df["support_system"]=None #LLM
# df["presenting_problem"]=None #LLM



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
# "patient_living_details",
# "presenting_problem",
# "reason_for_seeking_help",
# "past_history",
# "functioning_level",
# "support_system",
"patient_context",
# "disorder_symptoms",
"conversational_styles",
"core_beliefs",
"situation",
"intermediate_beliefs",
"automatic_thoughts",
]


df = df.reindex(columns=desired_order)

cols_except = [col for col in df.columns if col not in ["patient_ID", "core_beliefs"]]
df.drop_duplicates(subset=cols_except, keep="first", inplace=True)

df.reset_index(drop=True, inplace=True)

df["patient_ID"] = df.index + 1000

# Extract name age marital status gender from the patient_context column
# for row in df:
# Read regex in format Name: <name> and output name
# Insert name into patient_name column
# Remove Name: <name> from patient_context
# Read regex in format Age: <age> and output age
# Insert age into patient_age column
# Remove Age: <age> from patient_context
# Read regex in format Gender: <gender> and output gender.lower()
# If male then insert 'm', if female then insert 'f', else NULL
# Remove Gender: <gender> from patient_context
# Read regex in format Marital Status: <marital status> and output marital status.lower()
# If single then insert 'single', if married then insert 'married', else 'not specified'
# Remove Marital Status: <marital status> from patient_context
def extract_and_clean(row):
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


    # # Patient living details
    # details = re.search(r"Family Details:\s*([^\n]+)", context)
    # details = details.group(1).strip() if details else None
    context = re.sub(r"Family Details:\s*[^\n]+\n?", "", context)

    # # Presenting problem
    # presenting_problem = None
    # match = re.search(r"2\. Presenting Problem\s*(.*?)(?=\n\s*3\.)", context, re.DOTALL)
    # if match:
    # presenting_problem = match.group(1).strip()
    context = re.sub(r"2\. Presenting Problem\s*", "", context, flags=re.DOTALL)

    # # Reason for seeking help: between "3. Reason for Seeking Counseling" and "4."
    # reason_for_seeking_help = None
    # match = re.search(r"3\. Reason for Seeking Counseling\s*(.*?)(?=\n\s*4\.)", context, re.DOTALL)
    # if match:
    # reason_for_seeking_help = match.group(1).strip()
    context = re.sub(r"3\. Reason for Seeking Counseling\s*", "", context, flags=re.DOTALL)


    # # Past history: between "4. Past History (including medical history)" and "5."
    # past_history = None
    # match = re.search(r"4\. Past History \(including medical history\)\s*(.*?)(?=\n\s*5\.)", context, re.DOTALL)
    # if match:
    # past_history = match.group(1).strip()
    context = re.sub(r"4\. Past History \(including medical history\)\s*", "", context, flags=re.DOTALL)


    # # Functioning level: between "5. Academic/occupational functioning level:" and "6."
    # functioning_level = None
    # match = re.search(r"5\. Academic/occupational functioning level:?\s*(.*?)(?=\n\s*6\.)", context, re.DOTALL)
    # if match:
    # functioning_level = match.group(1).strip()
    context = re.sub(r"5\. Academic/occupational functioning level:?\s*", "", context, flags=re.DOTALL)


    # # Support system: between "6. Social Support System" and end or next section
    # support_system = None
    # match = re.search(r"6\. Social Support System\s*(.*)", context, re.DOTALL)
    # if match:
    # support_system = match.group(1).strip()
    context = re.sub(r"6\. Social Support System\s*", "", context, flags=re.DOTALL)

    # Patient living details
    education = re.search(r"Education:\s*([^\n]+)", context)
    education = education.group(1).strip() if education else None
    context = re.sub(r"Education:\s*[^\n]+\n?", "", context)

    # Patient living details
    occupation = re.search(r"Occupation:\s*([^\n]+)", context)
    occupation = occupation.group(1).strip() if occupation else None
    context = re.sub(r"Occupation:\s*[^\n]+\n?", "", context)


    print("done")
    return pd.Series({
    "patient_name": name,
    "patient_age": age,
    "patient_gender": gender,
    "patient_marital_status": marital_status,
    # "patient_living_details": details,
    # "presenting_problem": presenting_problem,
    # "reason_for_seeking_help": reason_for_seeking_help,
    # "past_history": past_history,
    # "functioning_level": functioning_level,
    # "support_system": support_system,
    "patient_education": education,
    "patient_occupation": occupation,
    "patient_context": context.strip()
    })


# Remove duplicates where every field is same
cols_except = [col for col in df.columns if col not in ["patient_ID", "core_beliefs"]]
df = df.drop_duplicates(subset=cols_except, keep="first")



# model1 = OllamaLLM(base_url="localhost:11434", model="qwen3:32b", temperature=0.4)


# def call_llm(prompt: str) -> str:
# msgs = ChatPromptTemplate.from_messages([
# ("system", "/no_think You are a helpful assistant that extracts and categorizes information accurately according to the instructions provided by the user."),
# ("user", prompt)
# ]).format_messages()


# response = model1.invoke(msgs)

# # print(response)
# return response


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

import ollama

client = ollama.Client(host="http://localhost:11434")


def call_llm(prompt: str) -> str:
    
    system = """
You are a clinical NLP annotation assistant helping construct CBT cognitive profiles from therapy intake text.

Scope and definitions:
- Use CBT terms in the standard way: situations are observable events; automatic thoughts are interpretations; intermediate beliefs are rules/assumptions/attitudes (often “If…then…”, “I should…”, “To be accepted, I must…”), influenced by core beliefs/schemas. [web:2]
- Do not diagnose, give treatment advice, or add clinical interpretations that are not explicitly supported by the provided text. [web:20]

Output discipline:
- Follow the user’s requested output format exactly (e.g., plain string, comma-separated labels, or JSON). [web:23][web:28]
- If required information is missing, output an empty string ("") or null (as specified by the user) rather than guessing. [web:18]
- Do not include explanations unless explicitly requested.
- Never include quotes around outputs unless requested.

Faithfulness:
- Prefer extraction over invention when the task is “extract”.
- When the task is “generate/hypothesize” (e.g., intermediate belief), keep it concise, consistent with the given core beliefs + automatic thoughts, and generalizable across situations. [web:2]
    
Respond in English with your answer. Do not include any other keys or commentary.   
    """
    
    resp = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ],
        #format=schema,
        stream=False,
        options={
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_ctx": 8192},
    )
    #print(resp)
    resp = resp["message"]["content"]
    print(resp)
    match = re.search(r"</think>\s*\n([\s\S]*)", resp)
    match = match.group(1).strip()
    print(match)
    if not match:
        print("\n\n\n-----------------------------------------------------")
        print(resp)
        print("-----------------------------------------------------\n\n\n")
        return "NULL"
    print(match)
    return match




def situation(row):
    prompt = f"""
You are extracting "Situation + impact" for a CBT cognitive profile.

FORMAT:
- Sentence 1 (Event): what happened (who/what/where/when), first person, observable.
- Sentence 2 (Impact): a brief statement of the negative impact using ONLY consequences explicitly mentioned (symptoms/impairment/behavioral fallout). If none, output only sentence 1. [web:38][web:44]

BANNED in Sentence 1:
- emotions, interpretations, judgments, motives, diagnoses.

ALLOWED in Sentence 2:
- symptom/impairment language when explicitly present (e.g., "I had a panic attack", "I avoided class", "I couldn’t sleep", "I missed work"). [web:38]

INPUT:
Patient story:
{row.get('patient_context', '')}

Automatic thoughts:
{row.get('automatic_thoughts', '')}

Core beliefs (context only):
{row.get('core_beliefs', '')}

OUTPUT:
Return only the final string (1–2 sentences).

"""
    llm_response = call_llm(prompt)
    result = (llm_response or "").strip().strip('\'"')
    return result


def intermediate_beliefs(row):
    prompt = f"""
You are generating ONE "Intermediate Belief" for a CBT cognitive model.

DEFINITION (Intermediate belief):
A rule/assumption/attitude that helps produce the automatic thought in the given situation and is shaped by the core belief. [page:1][page:2]
Often phrased as:
- "If X happens, then Y."
- "To be OK/accepted/safe, I must/should ..."
- "If I am not ___, then ___." [page:2]

STRICT RULES:
- Output exactly ONE sentence.
- Must be generalizable across situations (not a restatement of the specific event). [page:2]
- Do NOT include emotions, diagnoses, or advice.
- Use first person.
- Prefer one of these forms: "If..., then..." OR "To..., I must..." OR "I should..."
- Keep it <= 25 words.

INPUT:
Patient story:
{row.get('patient_context', '')}

Situation:
{row.get('situation', '')}

Automatic thoughts (closest-to-surface cognition):
{row.get('automatic_thoughts', '')} [page:2]

Core beliefs (deeper cognition):
{row.get('core_beliefs', '')} [page:2]

OUTPUT (return only the intermediate belief sentence, no quotes, no labels):
"""
    llm_response = call_llm(prompt)
    result = (llm_response or "").strip().strip('\'"')
    return result





# Assign disorder
# for row in df:
# Read disorder symptoms column
# Using LLM prompt using function call_llm(prompt: str) -> str
# assign disorder into disorder_symptoms column

# def disorder(row):
# prompt = (
# f"Patient story:\n{row.get('patient_context', '')}\n\n"\
# f"Patient Core Beliefs:\n{row.get('core_beliefs', '')}\n\n"\
# f"Patient Automatic Thoughts:\n{row.get('automatic_thoughts', '')}\n\n"\
# f"Patient age: {row.get('patient_age', '')}\n\n"\
# "Based on the above story and profile, identify the most likely psychological disorder or symptom that the patient has. "
# "Only output the disorder name as a string, nothing else."
# )
# llm_response = call_llm(prompt)
# # Extract everything after </think>
# match = re.search(r"</think>\s*\n(.*)", llm_response, re.DOTALL)
# if match:
#     result = match.group(1).strip()
# else:
#     result = ""


# result = result.strip('\'"')
# print(row.get('patient_context', ''))
# print(row.get('core_beliefs', ''))
# print(row.get('automatic_thoughts', ''))
# print(result)
# print("-----------------------------------------\n\n")
# return result








ALLOWED_STYLES = [
    "anxious","guilty","self-blaming","ruminative","avoidant","reassurance-seeking","shame-based",
    "socially vigilant","sensitive-to-rejection","catastrophic-framing","pessimistic","rigid-all-or-nothing",
    "overexplaining","detail-heavy","hesitant","guarded","emotionally intense","hopeless","low-energy",
    "tearful","self-critical","people-pleasing","conflict-avoidant"
]

def conversational_styles(row):
    prompt = f"""
You are labeling "conversational style" in a therapy intake transcript.

DEFINITION:
Conversational style = HOW the patient communicates (tone, pacing, interaction pattern, structure), not diagnosis,
not symptom severity, and not cognitive distortion names. [page:2]

ALLOWED STYLES (choose only from this list):
{", ".join(ALLOWED_STYLES)}

RULES:
- Choose 3 to 7 styles.
- Must be from the allowed list exactly (case-insensitive match).
- No duplicates.
- Output must be ONLY a comma-separated list (no tags, no quotes, no extra text).

INPUT:
Patient story:
{row.get('patient_context', '')}

Automatic thoughts:
{row.get('automatic_thoughts', '')}

Core beliefs:
{row.get('core_beliefs', '')}

Patient info (optional context):
Age: {row.get('patient_age', '')}, Gender: {row.get('patient_gender', '')},
Education: {row.get('patient_education', '')}, Occupation: {row.get('patient_occupation', '')}

OUTPUT:
<comma-separated list>
"""
    llm_response = call_llm(prompt) or ""
    raw = llm_response.strip().strip('\'"')

    # Optional: post-filter to guarantee allowed-only
    # (strongly recommended in production)
    styles = []
    for s in [x.strip().lower() for x in raw.split(",")]:
        if s in ALLOWED_STYLES and s not in styles:
            styles.append(s)
    return styles[:7]



error_log_path = "error_log.txt"
progress_csv = "cactus_progress.csv"

# Initialize progress file with headers if it doesn't exist
if not os.path.exists(progress_csv):
    df.head(0).to_csv(progress_csv, index=False)

def save_cactus(idx):
    if idx % 10 == 0:
        df.to_csv(progress_csv, mode='a', index=False)
        print(df.head())



def safe_apply(func, row, idx, total):
    try:
        result = func(row)
        save_cactus(idx)
        print(f"Saved row {idx + 1}/{total}")
        return result
    except Exception as e:
        with open(error_log_path, "a") as f:
            f.write(f"Error in row {row.name}:\n{traceback.format_exc()}\n\n")
        print(f"Error in row {idx + 1}, check error_log.txt")
        return None 


# Process extract_and_clean first
df[["patient_name", "patient_age", "patient_gender", "patient_marital_status", "patient_education", "patient_occupation", "patient_context"]] = df.apply(lambda row: safe_apply(extract_and_clean, row, row.name, len(df)), axis=1)
print(df)
df.to_csv("cactus_processed_partial.csv", mode='w', index=True)

# Process situation
df["situation"] = df.apply(lambda row: safe_apply(situation, row, row.name, len(df)), axis=1)

# Process intermediate_beliefs
df["intermediate_beliefs"] = df.apply(lambda row: safe_apply(intermediate_beliefs, row, row.name, len(df)), axis=1)

# Process conversational_styles
df["conversational_styles"] = df.apply(lambda row: safe_apply(conversational_styles, row, row.name, len(df)), axis=1)
# df["disorder_symptoms"] = df.apply(lambda row: safe_apply(disorder, row), axis=1)


df.to_json("cactus_processed_full.json", orient="records", lines=True)
df.to_csv("cactus_processed_full.csv", index=False)
df.to_excel("cactus_processed_full.xlsx", index=False)


df2 = df.iloc[:30, :]
df2.to_excel("cactus_processed_30_samples.xlsx", index=False)

