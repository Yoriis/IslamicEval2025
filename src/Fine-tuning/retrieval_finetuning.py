
import os
import ast
import gc
import faiss
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
from huggingface_hub import login
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import ContrastiveLoss, CosineSimilarityLoss
from datasets import Dataset
from huggingface_hub import HfApi, login


tydi_train = pd.read_csv("tydiqa_arabic_train.csv", sep='\t', encoding="utf-8")
tafseer_train = pd.read_csv("tafsir_train.csv")
quqa_train = pd.read_csv("train_quqa.csv")
haqa_train = pd.read_csv("haqa_train.csv")

def standardize_dataset_format(df: pd.DataFrame, dataset_name: str = "unknown") -> pd.DataFrame:
    print(f"Standardizing dataset: {dataset_name}")
    print(f"Original shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    standardized_data = []

    if 'question' in df.columns and 'passage' in df.columns:
        # Format 1: question, passage, label, passage_id columns
        for _, row in df.iterrows():
            standardized_data.append({
                "text1": str(row["question"]).strip(),
                "text2": str(row["passage"]).strip(),
                "label": int(row["label"]),
                "passage_id": str(row.get("passage_id", row.get("question_id", f"unknown_{len(standardized_data)}")))
            })

    elif 'question' in df.columns and 'passage_id' in df.columns:
        # Format 2: Need to lookup passages separately
        print("Warning: This format requires separate passage lookup!")
        print("   Please provide passage lookup or use create_passage_lookup() function")
        return df

    else:
        raise ValueError(f"Unsupported dataset format. Columns: {list(df.columns)}")

    result_df = pd.DataFrame(standardized_data)
    print(f"Standardized shape: {result_df.shape}")
    print(f"Positive samples: {(result_df['label'] == 1).sum()}")
    print(f"Negative samples: {(result_df['label'] == 0).sum()}")

    return result_df

tydi_train = standardize_dataset_format(tydi_train, "tydi")
tafseer_train = standardize_dataset_format(tafseer_train, "tafseer")
quqa_train = standardize_dataset_format(quqa_train, "quqa")
haqa_train = standardize_dataset_format(haqa_train, "haqa")


from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import pandas as pd
import os

def fine_tune(
    df: pd.DataFrame,
    model_name: str = "NAMAA-Space/AraModernBert-Base-ST",
    loss_function: str = "cosine",
    output_dir: str = "finetuned/",
    batch_size: int = 4,
    epochs: int = 3,
    warmup_steps: int = 100,
    show_progress_bar: bool = True,
    ft_model = None,
):
    print(f"\nFine-tuning `{model_name}` with {loss_function} loss on {len(df)} samples")

    train_samples = [
        InputExample(texts=[row["text1"], row["text2"]], label=float(row["label"]))
        for _, row in df.iterrows()
    ]

    # Load model
    if ft_model:
      model = ft_model
    else:
      model = SentenceTransformer(model_name)

    # Choose loss
    if loss_function == "cosine":
        train_loss = losses.CosineSimilarityLoss(model)
    elif loss_function == "contrastive":
        train_loss = losses.ContrastiveLoss(model)
    else:
        raise ValueError(f"Unsupported loss: {loss_function}")

    # DataLoader
    train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=batch_size)

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Train
    model.fit(
      train_objectives=[(train_dataloader, train_loss)],
      epochs=epochs,
      warmup_steps=warmup_steps,
      output_path=output_dir,
      show_progress_bar=show_progress_bar,
      use_amp=True,
    )

    print(f"Done! Model saved to: {output_dir}")

    return model


BATCH_SIZE = 8
NUM_EPOCHS = 4
SAVE_DIR = "saved_models"
HF_ORG = "HF_ORG"
HF_TOKEN = "HF_TOKEN"

login(token=HF_TOKEN)


def train_model(loss_function):
  nickname = "cos" if loss_function == "cosine" else "cont"
  i = 1

  # TYDI
  model = fine_tune(tydi_train, model_name = "NAMAA-Space/AraModernBert-Base-STS", loss_function=loss_function, output_dir=f"finetuned/step_{i}/")
  model.push_to_hub(f"{HF_ORG}/NAMAA-retriever-{loss_function}-{i}")
  model = SentenceTransformer("yoriis/NAMAA-retriever-cosine-1")
  i += 1

  # TAFSEER
  model = fine_tune(tafseer_train, model_name = "NAMAA-Space/AraModernBert-Base-STS", loss_function=loss_function, output_dir=f"finetuned/step_{i}/", ft_model = model)
  model.push_to_hub(f"{HF_ORG}/NAMAA-retriever-{loss_function}-{i}")

  i += 1

  # QUQA
  model = fine_tune(quqa_train, model_name = "NAMAA-Space/AraModernBert-Base-STS", loss_function=loss_function, output_dir=f"finetuned/step_{i}/", ft_model = model)
  model.push_to_hub(f"{HF_ORG}/NAMAA-retriever-{loss_function}-{i}")

  # HAQA
  model = fine_tune(haqa_train, model_name = "NAMAA-Space/AraModernBert-Base-STS", loss_function=loss_function, output_dir=f"finetuned/step_{i}/", ft_model = model)
  model.push_to_hub(f"{HF_ORG}/NAMAA-retriever-{loss_function}-final")

  return model

model_cos = train_model("cosine")

model_cont = train_model("contrastive")