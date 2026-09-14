import os
import re
import gc
import shutil
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from datasets import Dataset, Audio, DatasetDict, load_from_disk, concatenate_datasets
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from transformers import WhisperProcessor
from collections import Counter, defaultdict

CANTON_TO_REGION = {
    "AG": "AG", # Aargau
    "BE": "BE", # Bern
    "BS": "BS", # Basel-Stadt
    "GR": "GR", # Graubünden (Grisons)
    "LU": "LU", # Lucerne
    "SG": "SG", # St. Gallen
    "VS": "VS", # Wallis (Valais)
    "ZH": "ZH", # Zurich

    # Northwestern Switzerland -> BS
    "BL": "BS", # Basel-Landschaft

    # Innerschweiz / Central Switzerland -> LU
    "NW": "LU", # Nidwalden
    "SZ": "LU", # Schwyz
    "UR": "LU", # Uri

    # Ostschweiz / Eastern Switzerland -> SG
    "GL": "SG", # Glarus
    "SH": "SG", # Schaffhausen

    # Schaffhausen -> ZH 
    #"SH": "ZH", # (Robyn thinks this may be a bad merge)
}

LABEL2ID = {
    "AG": 0, "BE": 1, "BS": 2, "GR": 3,
    "LU": 4, "SG": 5, "VS": 6, "ZH": 7
}

MODEL_ID = "Flix-AI/flix-swissgerman-full"
DATA_DIR = Path("ArchiMob")
AUDIO_DIR = DATA_DIR / "audio_segmented_anonymized"
METADATA_PATH = DATA_DIR / "Metadata.txt"
OUTPUT_DIR = Path("./bucket/archimob-preprocessed")

def parse_metadata(metadata_path):
    df = pd.read_csv(metadata_path, sep="\t", dtype=str)
    df.columns = df.columns.str.strip()

    metadata = {}

    for _, row in df.iterrows():
        doc_id = row["DocID"].strip()
        speaker_id = row["SpeakerID"].strip()
        dialect_area = row["Dialect area"].strip()

        match = re.match(r"^([A-Z]{2})", dialect_area)
        canton = match.group(1) if match else None

        region = CANTON_TO_REGION.get(canton)
        label = LABEL2ID.get(region)

        metadata[doc_id] = {
            "speaker_id": speaker_id,
            "dialect_area": dialect_area,
            "canton": canton,
            "region": region,
            "label": label,
        }

    return metadata

def parse_xml_transcripts(xml_path: Path, speaker_id: str) -> list[dict]:
    """Return interviewee utterances with their audio segment IDs."""
    if not xml_path.exists():
        return []

    tree = ET.parse(xml_path)
    root = tree.getroot()

    XML_ID = "{http://www.w3.org/XML/1998/namespace}id"
    target_who = f"person_db#{speaker_id}"

    records = []

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]

        if tag != "u":
            continue

        if elem.attrib.get("who") != target_who:
            continue

        utt_id = elem.attrib.get(XML_ID)
        start = elem.attrib.get("start")

        if not utt_id or not start:
            continue

        # media_pointers#d1007-T3 -> d1007_T3
        if not start.startswith("media_pointers#"):
            continue

        segment_id = start.removeprefix("media_pointers#").replace("-", "_", 1)

        # Extract actual token text.
        words = []

        for child in elem.iter():
            child_tag = child.tag.split("}")[-1]

            if child_tag == "w":
                text = (child.text or "").strip()
                if text:
                    words.append(text)

            elif child_tag == "vocal":
                desc = child.find(".//{*}desc")
                if desc is not None and desc.text:
                    words.append(desc.text.strip())

            elif child_tag == "gap":
                # Could alternatively omit it entirely.
                words.append("...")

        text = " ".join(words)
        text = re.sub(r"\s+", " ", text).strip()

        records.append({
            "utt_id": utt_id,
            "segment_id": segment_id,
            "text": text,
        })

    return records

def load_archimob_dataset(data_dir: Path) -> Dataset:
    metadata = parse_metadata(METADATA_PATH)

    records = []

    for folder in sorted(AUDIO_DIR.iterdir()):
        if not folder.is_dir():
            continue

        # 1082_1 -> 1082
        doc_id = folder.name.split("_")[0]

        if doc_id not in metadata:
            print(f"WARNING: no metadata for {doc_id}")
            continue

        meta = metadata[doc_id]

        if meta["label"] is None:
            print(
                f"WARNING: no region mapping for "
                f"{doc_id}: {meta['dialect_area']}"
            )
            continue

        # ArchiMob XML is based on the base DocID.
        xml_path = data_dir / f"{doc_id}.xml"

        utterances = parse_xml_transcripts(
            xml_path,
            meta["speaker_id"],
        )

        wavs = {
            wav.stem: wav
            for wav in folder.glob("*.wav")
        }

        for utt in utterances:
            wav = wavs.get(utt["segment_id"])

            if wav is None: # not sure that I care about this for now, just skip
                # print(
                #     f"WARNING: missing audio "
                #     f"{utt['segment_id']} for {doc_id}"
                # )
                continue

            records.append({
                "audio": str(wav),
                "text": utt["text"],
                "utt_id": utt["utt_id"],
                "segment_id": utt["segment_id"],
                "doc_id": doc_id,
                "speaker_id": meta["speaker_id"],
                "dialect_area": meta["dialect_area"],
                "canton": meta["canton"],
                "dialect_region": meta["region"],
                "labels": meta["label"],
            })

    print(f"Loaded {len(records)} interviewee segments.")

    return Dataset.from_list(records).cast_column(
        "audio",
        Audio(sampling_rate=16000),
    )

# def create_grouped_split(dataset: Dataset, group_column="doc_id", test_size=0.15, seed=42) -> DatasetDict:
#     groups = np.array(dataset[group_column])
#     gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
#     train_indices, val_indices = next(gss.split(X=np.zeros(len(groups)), groups=groups))

#     return DatasetDict({
#         "train": dataset.select(train_indices),
#         "validation": dataset.select(val_indices)
#     })

def split_archimob(
    dataset: Dataset, 
    group_col="doc_id", 
    label_col="labels", 
    test_size=0.15, 
    seed=42,
    allow_clip_split_for_singletons=True
) -> DatasetDict:
    
    np.random.seed(seed)
    labels = np.array(dataset[label_col])
    groups = np.array(dataset[group_col])
    
    # Map label -> list of unique doc_ids
    label_to_docs = defaultdict(set)
    for doc, lbl in zip(groups, labels):
        label_to_docs[lbl].add(doc)
    
    train_docs = set()
    val_docs = set()
    clip_level_val_indices = []
    
    for lbl, docs in label_to_docs.items():
        docs = sorted(list(docs))
        np.random.shuffle(docs)
        
        if len(docs) == 1:
            doc = docs[0]
            if allow_clip_split_for_singletons:
                # Clip-level split for singletons to ensure val coverage
                doc_indices = np.where(groups == doc)[0]
                np.random.shuffle(doc_indices)
                n_val = max(1, int(len(doc_indices) * test_size))
                clip_level_val_indices.extend(doc_indices[:n_val])
                # Rest remains in train implicitly
            else:
                # Strict group isolation: Force singletons to TRAIN
                train_docs.add(doc)
        else:
            # At least 2 docs: Force at least 1 doc into val, rest proportional
            n_val_docs = max(1, int(round(len(docs) * test_size)))
            val_docs.update(docs[:n_val_docs])
            train_docs.update(docs[n_val_docs:])
            
    # Build index masks
    all_indices = np.arange(len(dataset))
    
    val_mask = np.isin(groups, list(val_docs))
    if allow_clip_split_for_singletons and clip_level_val_indices:
        val_mask[clip_level_val_indices] = True
        
    train_indices = all_indices[~val_mask]
    val_indices = all_indices[val_mask]
    
    return DatasetDict({
        "train": dataset.select(train_indices),
        "validation": dataset.select(val_indices)
    })

def process_split_in_shards(split_dataset: Dataset, processor: WhisperProcessor, split_name: str, shard_size=3000) -> Dataset:
    """Processes features in fixed-size shards to keep PyArrow memory bounded."""
    total_samples = len(split_dataset)
    num_shards = int(np.ceil(total_samples / shard_size))
    shard_dir = OUTPUT_DIR / f"temp_{split_name}_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard_paths = []

    print(f"\nProcessing '{split_name}' split ({total_samples} samples) in {num_shards} shards...")

    for i in range(num_shards):
        shard_path = shard_dir / f"shard_{i}"
        shard_paths.append(shard_path)

        if shard_path.exists():
            print(f"  Shard {i+1}/{num_shards} already processed. Skipping.")
            continue

        start_idx = i * shard_size
        end_idx = min((i + 1) * shard_size, total_samples)
        sub_ds = split_dataset.select(range(start_idx, end_idx))

        def extract_features(batch):
            audios = [sample["array"] for sample in batch["audio"]]
            inputs = processor(audios, sampling_rate=16000, return_tensors="np")
            return {
                "input_features": inputs.input_features, # .astype(np.float16)
                "labels": batch["labels"],
            }

        print(f"  Mapping shard {i+1}/{num_shards} ({end_idx - start_idx} samples)...")
        mapped_shard = sub_ds.map(
            extract_features,
            batched=True,
            batch_size=64,
            remove_columns=sub_ds.column_names,
            num_proc=4,
        )

        mapped_shard.save_to_disk(shard_path)

        # Force garbage collection to reclaim PyArrow memory arenas
        del sub_ds, mapped_shard
        gc.collect()

    print(f"Re-loading and combining shards for '{split_name}'...")
    shards = [load_from_disk(p) for p in shard_paths]
    combined_dataset = concatenate_datasets(shards)

    # Clean up temporary shard directories
    shutil.rmtree(shard_dir)
    return combined_dataset

def main():
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    raw_dataset = load_archimob_dataset(DATA_DIR)

    # Perform group split by doc_id
    #raw_splits = create_grouped_split(raw_dataset, group_column="doc_id")
    raw_splits = split_archimob(raw_dataset, group_col="doc_id")

    # Process train and validation splits independently in memory-bounded shards
    train_dataset = process_split_in_shards(raw_splits["train"], processor, split_name="train", shard_size=3000)
    val_dataset = process_split_in_shards(raw_splits["validation"], processor, split_name="validation", shard_size=3000)

    # Recombine splits
    final_dict = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset
    })

    final_dict.set_format("torch")
    final_dict.save_to_disk(OUTPUT_DIR)
    print(f"Preprocessed dataset saved successfully to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()

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