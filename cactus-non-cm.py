import pandas as pd
import re

df = pd.read_json("hf://datasets/LangAGI-Lab/cactus/cactus.json")

df.drop(columns=["cbt_technique"], inplace=True)
df.drop(columns=["cbt_plan"], inplace=True)
df.drop(columns=["dialogue"], inplace=True)
df.drop(columns=["attitude"], inplace=True)

cols_except = [col for col in df.columns if col not in ["patterns"]]
df = df.drop_duplicates(subset=cols_except, keep="first")

df.reset_index(drop=True, inplace=True)

df["conversational_styles"]=None #LLM
df["patient_ID"]=df.index+1000 #auto assign from 1000
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

df.to_csv("cactus-non-cm-processed.csv", index=False)
df.to_json("cactus-non-cm-processed.json", orient="records", lines=True)