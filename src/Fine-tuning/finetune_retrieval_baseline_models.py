#!/usr/bin/env python3
"""
Fine-tuning Script for Baseline Models - Islamic Eval 2025
Fine-tunes baseline models using both cosine and contrastive similarity losses
"""

import os
import pandas as pd
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments
)
from sentence_transformers.losses import ContrastiveLoss, CosineSimilarityLoss
import ast

BASELINE_MODELS = [
    "NAMAA-Space/AraModernBert-Base-STS",
    "silma-ai/silma-embeddding-sts-v0.1",
    "omarelshehy/Arabic-Retrieval-v1.0",
    "omarelshehy/Arabic-STS-Matryoshka-V2",
    "Omartificial-Intelligence-Space/GATE-AraBert-v1",
    "ALJIACHI/bte-base-ar",
    "mohamed2811/Muffakir_Embedding",
    "silma-ai/silma-embeddding-matryoshka-v0.1",
    "Omartificial-Intelligence-Space/Arabic-Triplet-Matryoshka-V2",
    "AhmedZaky1/arabic-bert-sts-matryoshka",
    "Alibaba-NLP/gte-multilingual-base",
    "Omartificial-Intelligence-Space/Arabert-all-nli-triplet-Matryoshka",
    "AhmedZaky1/arabic-bert-nli-matryoshka",
    "AhmedZaky1/DIMI-embedding-v2",
    "ibm-granite/granite-embedding-278m-multilingual",
    "omarelshehy/arabic-english-sts-matryoshka-v2.0",
    "OmarAlsaabi/e5-base-mlqa-finetuned-arabic-for-rag",
    "intfloat/multilingual-e5-base",
    "ibm-granite/granite-embedding-107m-multilingual",
    "Abdelkareem/zaraah_jina_v3",
    "AhmedZaky1/DIMI-embedding-v4",
    "Snowflake/snowflake-arctic-embed-m-v2.0",
    "Abdelkareem/abjd",
    "Abdelkareem/ara-qwen3-18",
    "Omartificial-Intelligence-Space/Arabic-labse-Matryoshka",
    "sentence-transformers/LaBSE",
    "Omartificial-Intelligence-Space/Arabic-MiniLM-L12-v2-all-nli-triplet",
    "mixedbread-ai/mxbai-embed-large-v1",
    "metga97/Modern-EgyBert-Base",
    "metga97/Modern-EgyBert-Embedding",
    "sentence-transformers/all-mpnet-base-v2",
    "sentence-transformers/all-MiniLM-L6-v2"
]

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
    
    try:
        questions_df = pd.read_csv(dataset_path, sep=",")
    except:
        try:
            questions_df = pd.read_csv(dataset_path, sep="\t")
        except:
            questions_df = pd.read_csv(dataset_path)
    
    required_columns = ["question", "passage_id", "label"]
    if not all(col in questions_df.columns for col in required_columns):
        raise ValueError(f"Dataset must contain columns: {required_columns}")
    
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
        
        if loss_name == "contrastive":
            loss = loss_class(model=model)
        else:
            loss = loss_class(model=model)
        
        trainer = SentenceTransformerTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            loss=loss,
        )
        
        trainer.train()
        
        print(f"Successfully fine-tuned {model_name} with {loss_name} loss")
        return True
        
    except Exception as e:
        print(f"Error fine-tuning {model_name} using {loss_name} loss: {e}")
        return False

def main():
    """Main function to run baseline model fine-tuning"""
    print("Starting Baseline Model Fine-tuning for Islamic Eval 2025")
    print("=" * 70)
    
    passage_lookup = load_passages()
    
    dataset_configs = [
        ("", "finetune_data.csv", "finetune_data"),
    ]
    
    loss_types = {
        "contrastive": ContrastiveLoss,
        "cosine_similarity": CosineSimilarityLoss,
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
    
    base_output = "baseline_models_finetuned"
    os.makedirs(base_output, exist_ok=True)
    
    results = []
    
    folder, file_name, tag = dataset_configs[0]
    dataset_path = file_name if not folder else os.path.join(folder, file_name)
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        print("Please ensure finetune_data.csv exists in the current directory")
        return
        
    print(f"\nProcessing dataset: {dataset_path}")
    
    questions_df = load_dataset(dataset_path)
    formatted_samples = format_samples(questions_df, passage_lookup)
    
    train_dataset = Dataset.from_list(formatted_samples)
    
    print(f"Loaded {len(formatted_samples)} training samples")
    
    for model_name in BASELINE_MODELS:
        model_tag = model_name.split("/")[-1]
        
        for loss_name, loss_class in loss_types.items():
            print(f"\nFine-tuning {model_name} using {loss_name} loss...")
            
            output_dir = os.path.join(base_output, f"{model_tag}_{loss_name}_{tag}")
            training_args.output_dir = output_dir
            
            success = fine_tune_model(
                model_name, train_dataset, loss_name, loss_class, 
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
    results_df.to_csv(f"{base_output}/fine_tuning_results.csv", index=False)
    print(f"\nResults saved to: {base_output}/fine_tuning_results.csv")
    
    if total - successful > 0:
        print("\nFailed fine-tuning attempts:")
        for result in results:
            if not result["success"]:
                print(f"   - {result['model']} with {result['loss']} loss on {result['dataset']}")
    
    print("\nBaseline model fine-tuning completed!")

if __name__ == "__main__":
    main()
