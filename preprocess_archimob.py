import os
import re
from pathlib import Path
import pandas as pd
from datasets import Dataset, Audio
from transformers import WhisperProcessor

CANTON_TO_REGION = {
    "AG": "AG",
    "BE": "BE",
    "BS": "BS",
    "GR": "GR",
    "LU": "LU",
    "SG": "SG",
    "VS": "VS",
    "ZH": "ZH",

    # Central Switzerland -> LU
    "NW": "LU",
    "SZ": "LU",
    "UR": "LU",

    # Northwestern Switzerland -> BS
    "BL": "BS",

    # Eastern Switzerland -> SG
    "GL": "SG",

    # Schaffhausen -> ZH
    "SH": "ZH",
}

LABEL2ID = {
    "AG": 0, "BE": 1, "BS": 2, "GR": 3,
    "LU": 4, "SG": 5, "VS": 6, "ZH": 7
}

MODEL_ID = "Flix-AI/flix-swissgerman-full"
DATA_DIR = Path("ArchiMob")
AUDIO_DIR = DATA_DIR / "audio_segmented_anonymized"
METADATA_PATH = DATA_DIR / "Metadata.txt"
OUTPUT_DIR = "./bucket/archimob-preprocessed"


def parse_metadata(metadata_path: Path) -> dict:
    """Parses Metadata.txt and maps DocIDs to target label integers."""
    df = pd.read_csv(metadata_path, sep="\t", dtype=str)
    df.columns = [col.strip() for col in df.columns]

    doc_to_label = {}
    for _, row in df.iterrows():
        doc_id = str(row["DocID"]).strip()
        dialect_str = str(row["Dialect area"]).strip()

        match = re.match(r"^([A-Z]{2})", dialect_str)
        if match:
            canton = match.group(1)
            mapped_region = CANTON_TO_REGION.get(canton)
            if mapped_region and mapped_region in LABEL2ID:
                doc_to_label[doc_id] = LABEL2ID[mapped_region]
            else:
                doc_to_label[doc_id] = None
        else:
            doc_to_label[doc_id] = None

    return doc_to_label


def load_archimob_dataset() -> Dataset:
    """Scans folders and pairs audio files directly with region labels."""
    doc_to_label = parse_metadata(METADATA_PATH)

    audio_paths = []
    labels = []
    doc_ids = []

    for folder_name in os.listdir(AUDIO_DIR):
        folder_path = AUDIO_DIR / folder_name
        if not folder_path.is_dir():
            continue

        base_doc_id = folder_name.split("_")[0]
        label = doc_to_label.get(base_doc_id)

        if label is None:
            continue

        for wav_file in folder_path.glob("*.wav"):
            audio_paths.append(str(wav_file))
            labels.append(label)
            doc_ids.append(base_doc_id)

    print(f"Loaded {len(audio_paths)} matching audio segments.")

    ds = Dataset.from_dict({
        "audio": audio_paths,
        "labels": labels,
        "doc_id": doc_ids,
    })

    return ds.cast_column("audio", Audio(sampling_rate=16000))


def main():
    processor = WhisperProcessor.from_pretrained(MODEL_ID)

    raw_dataset = load_archimob_dataset()

    def extract_features(batch):
        audios = [sample["array"] for sample in batch["audio"]]
        inputs = processor(
            audios,
            sampling_rate=16000,
            return_tensors="np",
        )
        return {
            "input_features": inputs.input_features,
            "labels": batch["labels"],
        }

    processed_dataset = raw_dataset.map(
        extract_features,
        batched=True,
        batch_size=64,
        remove_columns=raw_dataset.column_names,
        num_proc=4,
    )

    processed_dataset.set_format("torch")
    processed_dataset.save_to_disk(OUTPUT_DIR)
    print(f"Dataset saved successfully to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()