import os
import sys
from functools import partial
import re
import string
from collections import defaultdict
import pandas as pd
import random
from tqdm import tqdm
import json
import tarfile
import gzip
from typing import List, Dict, Tuple
import numpy as np

# Tafseer Pre-processing

def read_tafseer(file_path):
    with open(file_path, ) as f:
        data = f.readlines()
    tafser_data = [e.split("|") for e in data if len(e.split("|")) == 3]

    tafseer_dict = defaultdict(dict)
    for surah, aya, tafser_text in tafser_data:
        tafseer_dict[surah][aya] = tafser_text
    return tafseer_dict

def read_docs_file(docs_file):
    doc_df = pd.read_csv(docs_file, sep="\t", names=["passage_id", "passage"])
    doc_df["passage_id"] = doc_df["passage_id"].astype(str)
    return doc_df

arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
english_punctuations = string.punctuation
punctuations_list = arabic_punctuations + english_punctuations

arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)



def remove_diacritics(text):
    text = re.sub(arabic_diacritics, '', text)
    return text


def remove_punctuations(text):
    translator = str.maketrans('', '', punctuations_list)
    return text.translate(translator)


def clean_text(text):
    text = remove_punctuations(text)
    text = remove_diacritics(text)

    return text

def expand_passage(doc_no, external):
    surah, aya = doc_no.split(":")
    s, e = aya.split("-")

    return clean_text(" ".join([external[surah][str(idx)] for idx in range(int(s), int(e) + 1)]))

jalalayn = read_tafseer("ar.jalalayn.txt")

full_data = read_docs_file("..\data\QH-QA-25_Subtask2_QPC_v1.1.tsv")

full_data["jalalayn"] = full_data["passage_id"].apply(partial(expand_passage, external=jalalayn), )
full_data["jalalayn_q"] = [f"j-{idx:04d}" for idx in range(full_data.shape[0])]

query_data = pd.concat([full_data[["jalalayn_q", "jalalayn"]].rename(columns={"jalalayn_q": "question_id", "jalalayn": "question"}), ])
qrel_df = pd.concat([full_data[["jalalayn_q", "passage_id"]].rename(columns={"jalalayn_q": "question_id", }), ])

qrel_df["label"] = "1"
qrel_df[["question_id", "passage_id", "label"]].to_csv(f"tafseer-qrel.tsv", sep="\t", index=False, header=False)
query_data[["question_id", "question"]].to_csv(f"tafseer-query.tsv", sep="\t", index=False, header=False)

full_data[["passage_id", "passage"]].to_csv(f"tafseer_docs.tsv", sep="\t", index=False, header=False)

qrels = pd.read_csv("tafseer-qrel.tsv", sep="\t", names=["question_id", "passage_id", "label"])
queries = pd.read_csv("tafseer-query.tsv", sep="\t", names=["question_id", "question"])
docs = pd.read_csv("tafseer_docs.tsv", sep="\t", names=["passage_id", "passage"])

train_tafseer = qrels.merge(queries, on="question_id").merge(docs, on="passage_id")

train_tafseer = train_tafseer[["question_id", "question", "passage_id", "passage", "label"]]

train_tafseer.head()

def process_tafseer_with_negatives(train_tafseer, num_negatives=3):
    # (passage_id, passage)
    all_passages = train_tafseer[["passage_id", "passage"]].drop_duplicates()
    passage_dict = dict(zip(all_passages["passage_id"], all_passages["passage"]))
    all_passage_ids = list(passage_dict.keys())

    # group up positives
    positives_grouped = train_tafseer[train_tafseer["label"] == 1].groupby("question_id")
    processed_rows = []

    for qid, group in tqdm(positives_grouped, desc="Processing Tafseer with negatives"):
        question = group["question"].iloc[0]
        positive_pids = set(group["passage_id"])

        # Add positives
        for _, row in group.iterrows():
            processed_rows.append({
                "question_id": row["question_id"],
                "passage_id": row["passage_id"],
                "question": row["question"],
                "passage": row["passage"],
                "label": 1
            })

        # Sample negatives
        available_negatives = [pid for pid in all_passage_ids if pid not in positive_pids]
        sampled_negatives = random.sample(
            available_negatives,
            k=min(num_negatives, len(available_negatives))
        )

        for neg_pid in sampled_negatives:
            processed_rows.append({
                "question_id": qid,
                "passage_id": neg_pid,
                "question": question,
                "passage": passage_dict[neg_pid],
                "label": 0
            })

    return pd.DataFrame(processed_rows)

train_tafseer = pd.read_csv("tafseer-qrel.tsv", sep="\t", names=["question_id", "passage_id", "label"])\
    .merge(pd.read_csv("tafseer-query.tsv", sep="\t", names=["question_id", "question"]), on="question_id")\
    .merge(pd.read_csv("tafseer_docs.tsv", sep="\t", names=["passage_id", "passage"]), on="passage_id")

train_tafseer = train_tafseer[["question_id", "question", "passage_id", "passage", "label"]]
train_tafseer_balanced = process_tafseer_with_negatives(train_tafseer, num_negatives=3)

train_tafseer_balanced.info()

train_tafseer_balanced.to_csv("tafseer_balanced.csv", index=False)

def convert_tydi_qa_to_dataframe(file_path: str,
                                include_unanswerable: bool = True,
                                language_filter: str = None) -> pd.DataFrame:

    data_rows = []
    def get_language(qa_item, file_lang=None):
        q_id = qa_item.get('id', '')
        if '-' in q_id:
            lang_map = {
                'ar': 'arabic', 'en': 'english', 'bn': 'bengali', 'fi': 'finnish',
                'id': 'indonesian', 'ja': 'japanese', 'ko': 'korean', 'ru': 'russian',
                'sw': 'swahili', 'te': 'telugu', 'th': 'thai'
            }
            return lang_map.get(q_id.split('-')[0].lower(), 'unknown')
        if file_lang:
            return file_lang.lower()
        question_text = qa_item.get('question', '')
        if any('\u0600' <= char <= '\u06FF' for char in question_text):
            return 'arabic'
        return 'unknown'

    def process_data(data, file_language=None):
        for article in data.get('data', []):
            for paragraph in article.get('paragraphs', []):
                context = paragraph.get('context', '')
                for qa in paragraph.get('qas', []):
                    language = get_language(qa, file_language)
                    if language_filter and language != language_filter.lower():
                        continue
                    answers = qa.get('answers', [])
                    is_impossible = qa.get('is_impossible', False)
                    if is_impossible or len(answers) == 0:
                        if not include_unanswerable:
                            continue
                        label = 0
                        answer_text = ""
                    else:
                        label = 1
                        answer_text = answers[0].get('text', '')
                    data_rows.append({
                        'question_id': qa.get('id', ''),
                        'question_text': qa.get('question', ''),
                        'context': context,
                        'answer': answer_text,
                        'label': label,
                        'language': language
                    })

    if file_path.endswith(('.tgz', '.tar.gz')):
        with tarfile.open(file_path, 'r:gz') as tar:
            for member in tar.getmembers():
                if member.isfile() and member.name.endswith('.json'):
                    file_language = 'arabic' if 'arabic' in member.name.lower() else None
                    with tar.extractfile(member) as f:
                        data = json.loads(f.read().decode('utf-8'))
                        process_data(data, file_language)
    elif file_path.endswith('.jsonl.gz'):
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    process_data(json.loads(line), None)
    elif file_path.endswith('.json'):
        with open(file_path, 'r', encoding='utf-8') as f:
            process_data(json.load(f), None)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

    return pd.DataFrame(data_rows)


def create_negative_samples(df: pd.DataFrame, num_negatives_per_question: int = 5,
                             random_seed: int = 42) -> pd.DataFrame:
    random.seed(random_seed)
    np.random.seed(random_seed)

    positive_samples = df[df['label'] == 1].copy()
    all_contexts = df['context'].unique().tolist()
    negatives = []

    for _, row in positive_samples.iterrows():
        available_contexts = [ctx for ctx in all_contexts if ctx != row['context']]
        if len(available_contexts) >= num_negatives_per_question:
            sampled = random.sample(available_contexts, num_negatives_per_question)
        else:
            sampled = random.choices(available_contexts, k=num_negatives_per_question)
        for i, ctx in enumerate(sampled):
            negatives.append({
                'question_id': f"{row['question_id']}_neg_{i+1}",
                'question_text': row['question_text'],
                'context': ctx,
                'answer': "",
                'label': 0,
                'language': row['language']
            })

    negative_df = pd.DataFrame(negatives)
    return pd.concat([positive_samples, negative_df], ignore_index=True)


def create_triplet_format(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={'question_text': 'question', 'context': 'passage'})


def create_and_save_arabic_tydiqa_dataset(train_file: str,
                                          dev_file: str,
                                          save_path: str = "tydiqa_arabic_triplets.tsv",
                                          include_unanswerable: bool = True,
                                          num_negatives_per_question: int = 5,
                                          random_seed: int = 42) -> pd.DataFrame:

    train_df = convert_tydi_qa_to_dataframe(train_file, include_unanswerable, 'arabic')
    dev_df = convert_tydi_qa_to_dataframe(dev_file, include_unanswerable, 'arabic')
    combined_df = pd.concat([train_df.assign(source='train'), dev_df.assign(source='dev')], ignore_index=True)

    if num_negatives_per_question > 0:
        combined_df = create_negative_samples(combined_df, num_negatives_per_question, random_seed)

    triplet_df = create_triplet_format(combined_df)
    triplet_df.to_csv(save_path, sep='\t', index=False, encoding="utf-8")
    return triplet_df

triplets = create_and_save_arabic_tydiqa_dataset(
    train_file='tydiqa-goldp-v1.1-train.json',
    dev_file='tydiqa-goldp-v1.1-dev.tgz',
    save_path='tydiqa_arabic_train.csv',
    include_unanswerable=False,
    num_negatives_per_question=3,
)

# QUQA

quqa = pd.read_excel("QUQA.xlsx")

def process_quqa(quqa_df, num_negatives=3):
    # passage_id
    quqa_df["passage_id"] = (
        quqa_df["Chapter_Number"].astype(str) + ":" +
        quqa_df["Verses_Number_Start"].astype(str) + "-" +
        quqa_df["Verses_Number_End"].astype(str)
    )

    quqa_df = quqa_df[quqa_df["Quran_Full_Verse_Answer"].notna()]

    # Get all unique verses
    all_verses = quqa_df[["passage_id", "Quran_Full_Verse_Answer"]].drop_duplicates()
    verse_dict = dict(zip(all_verses["passage_id"], all_verses["Quran_Full_Verse_Answer"]))

    all_passage_ids = list(verse_dict.keys())


    processed_rows = []

    for _, row in tqdm(quqa_df.iterrows(), total=len(quqa_df), desc="Building pairs"):
        question = row["Question_Text"]
        pos_pid = row["passage_id"]
        pos_passage = row["Quran_Full_Verse_Answer"]
        qid = row["Question_Id"]

        processed_rows.append({
            "question_id": qid,
            "passage_id": pos_pid,
            "question": question,
            "passage": pos_passage,
            "label": 1
        })

        # Sample negatives
        negatives = random.sample(
            [pid for pid in all_passage_ids if pid != pos_pid],
            k=min(num_negatives, len(all_passage_ids) - 1)
        )
        for neg_pid in negatives:
            processed_rows.append({
                "question_id": qid,
                "passage_id": neg_pid,
                "question": question,
                "passage": verse_dict[neg_pid],
                "label": 0
            })

    return pd.DataFrame(processed_rows)

train_quqa = process_quqa(quqa)

train_quqa.to_csv("train_quqa.csv", index = False)

# HAQA

haqa = pd.read_csv("HAQA.csv")

def process_haqa(haqa_df, num_negatives=5):
    haqa_df = haqa_df.copy()

    haqa_df = haqa_df[haqa_df["Hadith_Full_Answer"].notna()].reset_index(drop=True)
    haqa_df["passage_id"] = ["HADITH#{:05d}".format(i) for i in range(len(haqa_df))]

    passage_dict = dict(zip(haqa_df["passage_id"], haqa_df["Hadith_Full_Answer"]))
    all_passage_ids = list(passage_dict.keys())

    processed_rows = []

    for _, row in tqdm(haqa_df.iterrows(), total=len(haqa_df), desc="Processing HAQA"):
        question = row["Question_Text"]
        pos_pid = row["passage_id"]
        pos_passage = row["Hadith_Full_Answer"]
        qid = str(row["Question_Id"])

        processed_rows.append({
            "question_id": qid,
            "passage_id": pos_pid,
            "question": question,
            "passage": pos_passage,
            "label": 1
        })

        negatives = random.sample(
            [pid for pid in all_passage_ids if pid != pos_pid],
            k=min(num_negatives, len(all_passage_ids) - 1)
        )

        for neg_pid in negatives:
            processed_rows.append({
                "question_id": qid,
                "passage_id": neg_pid,
                "question": question,
                "passage": passage_dict[neg_pid],
                "label": 0
            })

    return pd.DataFrame(processed_rows)

haqa_train = process_haqa(haqa, num_negatives=5)

haqa_train.to_csv("haqa_train.csv", index=False)