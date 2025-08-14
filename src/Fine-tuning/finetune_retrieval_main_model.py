#!/usr/bin/env python3
"""
Fine-tuning Script for Main Model - Islamic Eval 2025
Fine-tunes the main model over different data sources
"""

import os
import pandas as pd
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.losses import ContrastiveLoss, CosineSimilarityLoss, MultipleNegativesRankingLoss
import ast

def load_passages():
    """Load Quran and Hadith passages"""
    print("Loading passages...")
    
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
    
    passage_lookup = dict(zip(all_passages_df["id"], all_passages_df["passage"]))
    
    print(f"Loaded {len(all_passages)} passages")
    return passage_lookup

def load_dataset(dataset_path):
    """Load and format dataset for training"""
    print(f"Loading dataset: {dataset_path}")
    
    questions_df = pd.read_csv(dataset_path, sep="\t")
    questions_df["question"] = questions_df["question"].astype(str)
    questions_df["passage_id"] = questions_df["passage_id"].astype(str)
    questions_df["label"] = questions_df["label"].astype(int)
    
    return questions_df

def format_samples(questions_df, passage_lookup):
    """Format samples for training"""
    formatted_samples = []
    
    for _, row in questions_df.iterrows():
        passage_text = passage_lookup.get(row["passage_id"], "[PASSAGE_NOT_FOUND]")
        formatted_samples.append({
            "text1": row["question"].strip(),
            "text2": passage_text.strip(),
            "label": int(row["label"])
        })
    
    return formatted_samples

def fine_tune_model(model_name, dataset, loss_name, loss_class, output_dir, training_args):
    """Fine-tune a single model with specified loss"""
    print(f"Fine-tuning {model_name} using {loss_name} loss...")
    
    try:
        model = SentenceTransformer(model_name)
        
        if loss_name == "mnrl":
            positives = [s for s in dataset if s["label"] == 1]
            current_train_dataset = Dataset.from_list(
                [{"text1": s["text1"], "text2": s["text2"]} for s in positives]
            )
            loss = loss_class(model=model)
        elif loss_name == "hybrid":
            current_train_dataset = Dataset.from_list(dataset)
            loss = loss_class(model=model, initial_weights=(0.5, 0.5), margin=0.5)
        else:
            current_train_dataset = Dataset.from_list(dataset)
            loss = loss_class(model=model)
        
        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=current_train_dataset,
            loss=loss,
        )
        
        trainer.train()
        
        if loss_name == "hybrid":
            w_contrastive, w_cosine = loss.get_weights()
            print(f"Learned weights: Contrastive={w_contrastive:.4f}, Cosine={w_cosine:.4f}")
        
        print(f"Successfully fine-tuned {model_name} with {loss_name} loss")
        return True
        
    except Exception as e:
        print(f"Error fine-tuning {model_name} with {loss_name} loss: {e}")
        return False

def main():
    """Main function to run main model fine-tuning over different data sources"""
    print("Starting Main Model Fine-tuning for Islamic Eval 2025")
    print("=" * 70)
    
    passage_lookup = load_passages()
    
    dataset_configs = [
        # From finetune_data/
        # ("finetune_data", "finetune_dataset_old_posonly.tsv", "old_posonly"),
        # ("finetune_data_ara", "finetune_dataset_top20.tsv", "old_top20"),
        # ("finetune_data_ara", "finetune_dataset_top25.tsv", "old_top25"),
        # ("finetune_data_ara", "finetune_dataset_top30.tsv", "old_top30"),
        # ("finetune_data_ara", "finetune_dataset_top35.tsv", "old_top35"),
        # ("finetune_data", "finetune_dataset_old_top40.tsv", "old_top40"),
        # ("finetune_data", "finetune_dataset_old_top45.tsv", "old_top45"),
        # ("finetune_data", "finetune_dataset_old_top50.tsv", "old_top50"),

        # From finetune_data_ara/
        # ("finetune_data_ara", "finetune_dataset_top40.tsv", "ara_top40"),
        ("finetune_data_final_ara_60-90", "finetune_dataset_top60.tsv", "ara_top60"),
        ("finetune_data_final_ara_60-90", "finetune_dataset_top70.tsv", "ara_top70"),
        ("finetune_data_final_ara_60-90", "finetune_dataset_top80.tsv", "ara_top80"),
        ("finetune_data_final_ara_60-90", "finetune_dataset_top90.tsv", "ara_top90"),
        # ("finetune_data_ara_75-100", "finetune_dataset_top95.tsv", "ara_top95"),
        # ("finetune_data_ara_75-100", "finetune_dataset_top100.tsv", "ara_top100"),
    ]
    
    # Models to fine-tune
    models_to_finetune = [
        "yoriis/NAMAA-retriever-tydi-tafseer-quqa-haqa-cos",
        # "yoriis/NAMAA-retriever-tydi-tafseer-quqa-haqa"
    ]
    
    # Loss types
    loss_types = {
        "contrastive": ContrastiveLoss,
        "cosine_similarity": CosineSimilarityLoss,
        #"mnrl": MultipleNegativesRankingLoss,
        # "hybrid": HybridLoss,  # Uncomment if HybridLoss is available
    }
    
    training_args = SentenceTransformerTrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=8,
        learning_rate=2e-5,
        warmup_steps=100,
        bf16=True,
        logging_steps=200,
        save_strategy="epoch",
    )
    
    base_output = "results"
    os.makedirs(base_output, exist_ok=True)
    
    results = []
    
    for folder, file_name, tag in dataset_configs:
        dataset_path = os.path.join(folder, file_name)
        
        if not os.path.exists(dataset_path):
            print(f"Dataset not found: {dataset_path}, skipping...")
            continue
            
        print(f"\nProcessing dataset: {dataset_path}")
        
        questions_df = load_dataset(dataset_path)
        formatted_samples = format_samples(questions_df, passage_lookup)
        
        for model_name in models_to_finetune:
            model_tag = model_name.split("/")[-1]
            
            for loss_name, loss_class in loss_types.items():
                print(f"\nFine-tuning {model_name} on {tag} using {loss_name} loss...")
                
                output_dir = os.path.join(base_output, f"{model_tag}_{loss_name}_{tag}")
                training_args.output_dir = output_dir
                
                success = fine_tune_model(
                    model_name, formatted_samples, loss_name, loss_class, 
                    output_dir, training_args
                )
                
                results.append({
                    "model": model_name,
                    "dataset": tag,
                    "loss": loss_name,
                    "output_dir": output_dir,
                    "success": success
                })
                
                import gc
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
    
    print("\n" + "=" * 70)
    print("Fine-tuning Summary")
    print("=" * 70)
    
    successful = sum(1 for r in results if r["success"])
    total = len(results)
    
    print(f"Successful: {successful}/{total}")
    print(f"Failed: {total - successful}/{total}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{base_output}/main_model_fine_tuning_results.csv", index=False)
    print(f"\nResults saved to: {base_output}/main_model_fine_tuning_results.csv")
    
    if total - successful > 0:
        print("\nFailed fine-tuning attempts:")
        for result in results:
            if not result["success"]:
                print(f"   - {result['model']} with {result['loss']} loss on {result['dataset']}")
    
    print("\nMain model fine-tuning completed!")
    print(f"Checkpoints saved in: {base_output}")

if __name__ == "__main__":
    main()
