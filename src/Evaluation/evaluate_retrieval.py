#!/usr/bin/env python3
"""
Model Evaluation Script - Islamic Eval 2025
Evaluates fine-tuned models using retrieval metrics, converted from fine_tune.ipynb
"""

import os
import ast
import gc
import faiss
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def load_data():
    """Load development data and passages"""
    print("Loading evaluation data...")
    
    dev_df = pd.read_csv("data/Task Data/data/combined_questions_with_passages_dev.tsv", sep="\t")
    
    quran_passages = []
    with open("data/Task Data/Thematic_QPC/QH-QA-25_Subtask2_QPC_v1.1.tsv", "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                quran_passages.append({"text": parts[1], "source": "quran", "id": parts[0]})
    
    hadith_passages = []
    with open("data/Task Data/Sahih-Bukhari/QH-QA-25_Subtask2_Sahih-Bukhari_v1.0.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = ast.literal_eval(line.strip())
                hadith_passages.append({"text": item['hadith'], "source": "hadith", "id": item['hadith_id']})
            except Exception as e:
                print(f"Skipping invalid line: {e}")
    
    all_passages = quran_passages + hadith_passages
    print(f"Loaded total passages: {len(all_passages)}")
    
    return dev_df, all_passages

def build_faiss_index(embeddings, similarity_type="cosine"):
    """Build FAISS index for similarity search"""
    if similarity_type == "cosine":
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
    elif similarity_type == "l2":
        index = faiss.IndexFlatL2(embeddings.shape[1])
    else:
        raise ValueError("Invalid similarity_type")
    index.add(embeddings)
    return index

def evaluate_retrieval_by_id_with_model(df, model, index, all_passages, similarity_type="cosine", top_k=10):
    """Evaluate retrieval performance using passage IDs"""
    precision_list, recall_list, f1_list = [], [], []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            gold_ids = set(ast.literal_eval(row["relevant_passages"]))
        except:
            continue
        
        if gold_ids == {"-1"}:
            continue

        # Encode query
        query_emb = model.encode([row["question_text"]],
                               convert_to_numpy=True,
                               normalize_embeddings=(similarity_type == "cosine"))
        
        # Search
        D, I = index.search(query_emb, top_k)
        predicted_ids = set(str(all_passages[i]["id"]) for i in I[0])

        # Calculate metrics
        tp = len(gold_ids & predicted_ids)
        precision = tp / len(predicted_ids) if predicted_ids else 0
        recall = tp / len(gold_ids) if gold_ids else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return {
        "precision": np.mean(precision_list),
        "recall": np.mean(recall_list),
        "f1": np.mean(f1_list)
    }

def evaluate_model_checkpoint(checkpoint_path, dev_df, all_passages, model_name, loss_type, data_config):
    """Evaluate a single model checkpoint"""
    print(f"Evaluating checkpoint: {checkpoint_path}")
    
    try:
        model = SentenceTransformer(checkpoint_path)
        
        texts = [p["text"] for p in all_passages]
        raw_embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            batch_size=64,
            show_progress_bar=True
        )
        
        results = []
        similarity_types = ["cosine", "l2"]
        top_k_values = [20, 30, 40, 50, 60, 70, 80]
        
        for sim_type in similarity_types:
            print(f"   Similarity: {sim_type}")
            
            embeddings = (
                raw_embeddings / np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
                if sim_type == "cosine" else raw_embeddings.copy()
            )
            
            index = build_faiss_index(embeddings, sim_type)
            
            for k in top_k_values:
                print(f"     Evaluating Top-{k}")
                metrics = evaluate_retrieval_by_id_with_model(
                    dev_df, model, index, all_passages, sim_type, top_k=k
                )
                
                results.append({
                    "model_name": model_name,
                    "loss_type": loss_type,
                    "data_config": data_config,
                    "checkpoint": os.path.basename(checkpoint_path),
                    "similarity_type": sim_type,
                    "top_k": k,
                    **metrics
                })
        
        return results
        
    except Exception as e:
        print(f"     Failed on {checkpoint_path}: {e}")
        return []
    
    finally:
        if 'model' in locals():
            del model
        torch.cuda.empty_cache()
        gc.collect()

def main():
    """Main function to evaluate all fine-tuned models"""
    print("Starting Model Evaluation for Islamic Eval 2025")
    print("=" * 70)
    
    dev_df, all_passages = load_data()
    
    base_dir = "results"
    
    if not os.path.exists(base_dir):
        print(f"Base directory not found: {base_dir}")
        print("Please run fine-tuning scripts first or update the base_dir path")
        return
    
    print(f"Evaluating models in: {base_dir}")
    
    results = []
    
    for model_name in sorted(os.listdir(base_dir)):
        model_path = os.path.join(base_dir, model_name)
        
        if not os.path.isdir(model_path):
            continue
        
        if "contrastive" in model_name:
            loss_type = "contrastive"
        elif "cosine_similarity" in model_name:
            loss_type = "cosine_similarity"
        elif "hybrid" in model_name:
            loss_type = "hybrid"
        elif "mnrl" in model_name:
            loss_type = "mnrl"   
        else:
            continue

        if "posonly" in model_name:
            data_config = "posonly"
            top_k_config = None
        elif "top" in model_name:
            top_k_config = int(model_name.split("_top")[-1]) 
            data_config = f"top{top_k_config}"
        else:
            data_config = "unknown"
            top_k_config = None

        print(f"\nModel: {model_name} | Loss: {loss_type} | Config: {data_config}")

        for ckpt_dir in sorted(os.listdir(model_path)):
            ckpt_path = os.path.join(model_path, ckpt_dir)
            
            if not os.path.isdir(ckpt_path) or not ckpt_dir.startswith("checkpoint-"):
                continue

            # Evaluate checkpoint
            checkpoint_results = evaluate_model_checkpoint(
                ckpt_path, dev_df, all_passages, model_name, loss_type, data_config
            )
            
            results.extend(checkpoint_results)
    
    if results:
        df_results = pd.DataFrame(results)
        output_file = "model_evaluation_results.csv"
        df_results.to_csv(output_file, index=False)
        print(f"\nAll models evaluated. Results saved to: {output_file}")
        
        print(f"\nEvaluation Summary:")
        print(f"   Total evaluations: {len(results)}")
        print(f"   Models evaluated: {len(set(r['model_name'] for r in results))}")
        print(f"   Checkpoints evaluated: {len(set(r['checkpoint'] for r in results))}")
        
        best_f1 = df_results.loc[df_results['f1'].idxmax()]
        print(f"\nBest F1 Score: {best_f1['f1']:.4f}")
        print(f"   Model: {best_f1['model_name']}")
        print(f"   Loss: {best_f1['loss_type']}")
        print(f"   Config: {best_f1['data_config']}")
        print(f"   Similarity: {best_f1['similarity_type']}")
        print(f"   Top-K: {best_f1['top_k']}")
        
    else:
        print("No evaluation results generated")

if __name__ == "__main__":
    main()
