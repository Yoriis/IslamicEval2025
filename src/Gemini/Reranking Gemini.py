import pandas as pd
import numpy as np
import ast
import os
import json
from google import generativeai as genai
from google.generativeai import types
import time
from tqdm import tqdm

# list of ks
top_ks = [70, 80, 100]
# load qrels data
qrels_dev = pd.read_csv('../data/QH-QA-25_Subtask2_ayatec_v1.3_dev.tsv', sep='\t', names=['q_id', 'question'])
qrels_dev_ans = pd.read_csv("../data/qrels/QH-QA-25_Subtask2_ayatec_v1.3_qrels_dev.gold", sep='\t', names=['q_id', 'nbr', 'verse_id', 'nbr_2'])

# load quran
quran = pd.read_csv('../data/Thematic_QPC/QH-QA-25_Subtask2_QPC_v1.1.tsv', sep='\t', names=['verse_id', 'verse'])

# load hadith
hadiths = []
with open("../data/Sahih-Bukhari/QH-QA-25_Subtask2_Sahih-Bukhari_v1.0.jsonl", 'r', encoding="utf-8") as file:
    for line in file:
        try:
            clean_line = ast.literal_eval(line)
            hadiths.append(clean_line)
        except (ValueError, SyntaxError) as e:
            print(f"⚠️ Skipping line due to parsing issue: {line}")

# Load into DataFrame
hadiths = pd.DataFrame(hadiths)

# load the retrieved docs
results_dir = "retreival results"
files_topk = []
for result in os.listdir(results_dir):
    filename = result.split("_")[-1].split('.')[0]
    files_topk.append(filename)
    print(f"variable {filename} created")
    globals()[filename] = pd.read_csv(f"{results_dir}/{result}")

# order of gold_ids gets scrambled in retreived file so we need to use the correct order
def reorder_gold_ids(file):
    for i, row in file.iterrows():
        # get a list of verse_ids
        verse_ids = qrels_dev_ans[qrels_dev_ans['q_id'] == row['question_id']]['verse_id'].tolist()
        # assign the list ot the cell
        file.at[i, 'gold_ids'] = verse_ids

    return file

def prep_data(answers, quran, hadiths):
    """
    function that takes the dataframe of the retreival results and 
    aggregates the ids in "predicted_ids" to the text of the corresponding verse or hadith

    Args:
    answers (Pandas DataFrame): the saved dataframe from the retreival steps
    quran (Pandas DataFrame): contains verse_id and verse
    hadiths (Pandas DataFrame): contains hadith_id and hadith (and other columns)

    Returns:
    preped_qs (list): a list of dicts where each dict contains the questions, some meta data and the retreived docs: ids aggregated with text 
    """
    preped_qs = []
    # answers is a df where the list of ids for each q is under 'predicted_ids'
    for _, ans in answers.iterrows():
        predicted_ids = ast.literal_eval(ans['predicted_ids'])
        predicted_verses = []
        for id in predicted_ids:
            try:
                predicted_verses.append(f"{id} - {quran['verse'].loc[quran['verse_id'] == id.strip()].values[0]}")
            except Exception as e:
                # if it's skipped it's hadith data
                predicted_verses.append(f"{id} - {hadiths['hadith'].loc[hadiths['hadith_id'] == int(id.strip())].values[0]}")
        preped_qs.append({
            'q_id': ans['question_id'],
            'question': ans['question_text'],
            'gold_ids': ans['gold_ids'],
            'predicted_ids': predicted_verses,
        })
    return preped_qs

# Configure the API
genai.configure(api_key="<GEMINI2.5-API-KEY>") 

# Load the model
model = genai.GenerativeModel("gemini-2.5-flash") 

# a safe generation function
def safe_generate(prompt, temperature=0.0):
    try:
        response = model.generate_content(prompt, generation_config={"temperature": temperature})
        return response.text.strip()
    except Exception as e:
        print(f"Error: {e}. Retrying after 60s...")
        time.sleep(60)
        try:
            response = model.generate_content(prompt, generation_config={"temperature": temperature})
            return response.text.strip()
        except Exception as e2:
            print(f"Second failure: {e2}")
            return ""

# filtering and reranking the retrieved documents using gemini
def filter_results_gemini(preped_data):
    results = []
    for row in tqdm(preped_data, total=len(preped_data)):
        q_id = row['q_id']
        question = row['question']
        context = row['predicted_ids']
        gold = row['gold_ids']

        
        PROMPT = f"""
        Given a question in Modern Standard Arabic (MSA) and a list of Quranic and Hadith verses (each with an associated ID), identify the IDs of the verses that contain the answer to the question.
        Instructions:
                - Return only the IDs of the extremely relevant verses in a list, ordered from most relevant to least relevant.
                - Do not explain your answer or provide verse text.
                - If the answer is not found in any verse, or you are unsure, you must return [-1].
                - Use the verse ID exactly as provided (e.g., if the verse ID is 23:14-16, return [23:14-16], not 23:14).
                - Format your response strictly as a Python list, like: [2:1-4, 4:5-7] or [-1].

        Question: {question}
        Verses: {context}
        """.strip()


        #print(len(context))

        # Generate output
        try:
            model_output = safe_generate(PROMPT, temperature=0.2)
        except Exception as e:
            model_output = f"[Error: {e}]"

        # Store results
        results.append({
            'q_id': q_id,
            'question': question,
            'predicted': context,
            'gold_answer_ids': gold,
            'model_output': model_output
        })

    return results

for file in files_topk:
    # create saving path
    save_dir = 'filtration results'
    os.makedirs(save_dir, exist_ok=True)  # ensures directory exists
    results_path = os.path.join(save_dir, f'filtration_{file}.json')

    # convert filename string to variable
    data = globals()[file]
    data = reorder_gold_ids(data)

    # aggregate the predicted_ids with the text
    prepped_data = prep_data(data, quran, hadiths)

    # prompt gemini to filter the data
    results = filter_results_gemini(prepped_data)

    # save the results
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)