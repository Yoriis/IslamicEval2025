#!/usr/bin/env python3
"""
Data Preprocessing Script for Islamic Eval 2025
Converts the data.ipynb notebook to a standalone Python script
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import ast
import json
import os

def load_data():
    """Load questions and passages data"""
    print("Loading data...")
    
    questions_df = pd.read_csv("data/Task Data/data/combined_questions_with_passages_train.tsv", sep="\t")
    questions_df["relevant_passages"] = questions_df["relevant_passages"].apply(ast.literal_eval)
    
    quran_df = pd.read_csv("data/Task Data/Thematic_QPC/QH-QA-25_Subtask2_QPC_v1.1.tsv", sep="\t", names=["id", "passage"], header=None)
    
    hadith_passages = []
    with open("data/Task Data/Sahih-Bukhari/QH-QA-25_Subtask2_Sahih-Bukhari_v1.0.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            item = ast.literal_eval(line.strip())
            hadith_passages.append({
                "passage": item["hadith"],
                "id": str(item["hadith_id"])
            })
    
    hadith_df = pd.DataFrame(hadith_passages)
    
    all_passages_df = pd.concat([quran_df, hadith_df], ignore_index=True)
    all_passages = all_passages_df.to_dict(orient="records")
    
    print(f"Loaded {len(questions_df)} questions and {len(all_passages)} passages")
    
    return questions_df, all_passages

def create_index_mappings(all_passages):
    """Create index to ID mappings"""
    index_to_id = {i: p["id"] for i, p in enumerate(all_passages)}
    id_to_index = {v: i for i, v in index_to_id.items()}
    return index_to_id, id_to_index

def encode_embeddings(questions_df, all_passages):
    """Encode questions and passages using the model"""
    print("Loading model and encoding embeddings...")
    
    model = SentenceTransformer("NAMAA-Space/AraModernBert-Base-STS", trust_remote_code=True)
    
    question_embeddings = model.encode(
        questions_df["question_text"].tolist(), 
        show_progress_bar=True, 
        convert_to_numpy=True
    )
    
    passage_texts = [p["passage"] for p in all_passages]
    qh_embeddings = model.encode(
        passage_texts, 
        show_progress_bar=True, 
        convert_to_numpy=True
    )
    
    print("Embeddings encoded successfully")
    return model, question_embeddings, qh_embeddings

def generate_finetune_datasets(questions_df, all_passages, index_to_id, id_to_index, 
                              question_embeddings, qh_embeddings):
    """Generate fine-tuning datasets with different top-k values"""
    
    top_k_values = [60, 70, 80, 90]
    output_dir = "finetune_data_ara_60-90"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating datasets in {output_dir}")
    
    print("Building positive-only dataset...")
    finetune_pos_only = []
    
    for i, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Building Positives Only"):
        gold_ids = set(row["relevant_passages"])
        for pid in gold_ids:
            if pid in id_to_index:
                finetune_pos_only.append({
                    "question": row["question_text"],
                    "passage_id": pid,
                    "label": 1
                })
    
    pos_only_df = pd.DataFrame(finetune_pos_only)
    pos_only_df.to_csv(f"{output_dir}/finetune_dataset_posonly.tsv", sep="\t", index=False)
    print(f"Saved positive-only dataset: {len(pos_only_df)} samples")
    
    for k in top_k_values:
        print(f"Building dataset with top-{k}...")
        finetune_data = []
        
        for i, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc=f"Building Dataset top-{k}"):
            q_embed = question_embeddings[i]
            similarities = cosine_similarity([q_embed], qh_embeddings)[0]
            top_indices = np.argsort(similarities)[::-1][:k]
            top_ids = [index_to_id[idx] for idx in top_indices]
            
            gold_ids = set(row["relevant_passages"])
            
            for pid in gold_ids:
                if pid in id_to_index:
                    finetune_data.append({
                        "question": row["question_text"],
                        "passage_id": pid,
                        "label": 1
                    })
            
            for pid in top_ids:
                if pid not in gold_ids:
                    finetune_data.append({
                        "question": row["question_text"],
                        "passage_id": pid,
                        "label": 0
                    })
        
        finetune_df = pd.DataFrame(finetune_data)
        finetune_df.to_csv(f"{output_dir}/finetune_dataset_top{k}.tsv", sep="\t", index=False)
        print(f"Saved top-{k} dataset: {len(finetune_df)} samples")

def main():
    """Main function to run the data preprocessing pipeline"""
    print("Starting Data Preprocessing for Islamic Eval 2025")
    print("=" * 60)
    
    try:
        questions_df, all_passages = load_data()
        
        index_to_id, id_to_index = create_index_mappings(all_passages)
        
        model, question_embeddings, qh_embeddings = encode_embeddings(questions_df, all_passages)
        
        generate_finetune_datasets(questions_df, all_passages, index_to_id, id_to_index,
                                 question_embeddings, qh_embeddings)
        
        print("\nData preprocessing completed successfully!")
        print("Generated datasets:")
        print("   - Positive-only dataset")
        print("   - Top-60 dataset")
        print("   - Top-70 dataset") 
        print("   - Top-80 dataset")
        print("   - Top-90 dataset")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        raise

if __name__ == "__main__":
    main()
