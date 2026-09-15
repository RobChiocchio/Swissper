import gc
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Audio, Dataset, DatasetDict, load_from_disk
from transformers import WhisperProcessor

# Mapping all German-speaking cantons to the 8 SwissDial regions
CANTON_TO_REGION = {
    # Core 8 regions
    "AG": "AG",  # Aargau
    "BE": "BE",  # Bern
    "BS": "BS",  # Basel-Stadt
    "GR": "GR",  # Graubünden
    "LU": "LU",  # Lucerne
    "SG": "SG",  # St. Gallen
    "VS": "VS",  # Wallis
    "ZH": "ZH",  # Zurich
    # Northwestern Switzerland -> BS
    "BL": "BS",  # Basel-Landschaft (Dataset explicitly states BS is labeled as BL)
    "SO": "BE",  # Solothurn (Bernese/Northwestern transition)
    # Central Switzerland (Innerschweiz) -> LU
    "NW": "LU",  # Nidwalden
    "OW": "LU",  # Obwalden
    "SZ": "LU",  # Schwyz
    "UR": "LU",  # Uri
    "ZG": "LU",  # Zug
    # Eastern Switzerland (Ostschweiz) -> SG
    "AI": "SG",  # Appenzell Innerrhoden
    "AR": "SG",  # Appenzell Ausserrhoden
    "GL": "SG",  # Glarus
    "TG": "SG",  # Thurgau
    # Schaffhausen & Fribourg
    #"SH": "ZH",  # Schaffhausen
    "SH": "SG", # TODO: is this valid?
    "FR": "BE",  # Fribourg (German-speaking region)
}

LABEL2ID = {
    "AG": 0, "BE": 1, "BS": 2, "GR": 3,
    "LU": 4, "SG": 5, "VS": 6, "ZH": 7
}

# MODEL_ID = "Flix-AI/flix-swissgerman-full"
DATA_DIR = Path("ASGDTS")
AUDIO_DIR = DATA_DIR / "clips"
TSV_PATH = DATA_DIR / "all.tsv"
OUTPUT_DIR = Path("./dataset/asgdts-preprocessed")

def parse_canton(accent_str: str) -> str | None:
    """Extract 2-letter canton code from accent column."""
    if not isinstance(accent_str, str):
        return None
    match = re.search(r"\b([A-Z]{2})\b", accent_str.strip())
    return match.group(1) if match else None


def load_raw_dataset(data_dir: Path) -> Dataset:
    df = pd.read_csv(TSV_PATH, sep="\t", dtype=str)

    records = []
    skipped_unmapped = 0
    skipped_missing_audio = 0

    for _, row in df.iterrows():
        clip_rel_path = row["path"]
        audio_path = AUDIO_DIR / clip_rel_path

        if not audio_path.exists():
            skipped_missing_audio += 1
            continue

        canton = parse_canton(row.get("accent", ""))
        region = CANTON_TO_REGION.get(canton) if canton else None
        label = LABEL2ID.get(region) if region else None

        if label is None:
            skipped_unmapped += 1
            continue

        records.append(
            {
                "audio": str(audio_path),
                "text": row.get("sentence", ""),
                "client_id": row.get("client_id", ""),
                "canton": canton,
                "dialect_region": region,
                "labels": label,
            }
        )

    print(
        f"Loaded {len(records)} samples. "
        f"Skipped {skipped_unmapped} unmapped dialects, "
        f"{skipped_missing_audio} missing audio files."
    )

    return Dataset.from_list(records).cast_column(
        "audio", Audio(sampling_rate=16000)
    )

def main():
    raw_dataset = load_raw_dataset(DATA_DIR)
    dataset = DatasetDict({ "test": raw_dataset })
    dataset.save_to_disk(OUTPUT_DIR)

if __name__ == "__main__":
    main()


    from collections import Counter, defaultdict


    ds = load_from_disk(OUTPUT_DIR)
    ds.set_format("torch")
    splits = ds.keys() if hasattr(ds, "keys") else ["dataset"]

    ID2LABEL = {0: "AG", 1: "BE", 2: "BS", 3: "GR", 4: "LU", 5: "SG", 6: "VS", 7: "ZH"}

    for split in splits:
        data = ds[split] if hasattr(ds, "keys") else ds
        labels = data["labels"]
        
        # Handle both PyTorch tensors and standard lists
        if hasattr(labels, "tolist"):
            labels = labels.tolist()
            
        counts = Counter(l.item() for l in labels)
        total = len(labels)
        
        print(f"=== {split.upper()} SPLIT (Total: {total}) ===")
        for label_id in range(8):
            name = ID2LABEL[label_id]
            count = counts.get(label_id, 0)
            pct = (count / total * 100) if total > 0 else 0.0
            print(f"Label {label_id} ({name}): {count:6d} samples ({pct:5.1f}%)")
        print()