import os
import re
from pathlib import Path
import xml.etree.ElementTree as ET
import pandas as pd
from datasets import Dataset, Audio
from transformers import WhisperProcessor

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

    # InnerSchweiz / Central Switzerland -> LU
    "NW": "LU", # Nidwalden
    "SZ": "LU", # Schwyz
    "UR": "LU", # Uri

    # Ostschweiz / Eastern Switzerland -> SG
    "GL": "SG", # Glarus
    "SH": "ZH", # Schaffhausen

    # Schaffhausen -> ZH 
    # "SH": "ZH", # (Robyn thinks this may be a bad merge)
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

def main():
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    raw_dataset = load_archimob_dataset(DATA_DIR)

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

    # Extract log-mel features and drop raw text/audio structures
    processed_dataset = raw_dataset.map(
        extract_features,
        batched=True,
        batch_size=64,
        remove_columns=raw_dataset.column_names,
        num_proc=4,
    )

    processed_dataset.set_format("torch")

    # Perform group split by doc_id
    split_dataset = create_grouped_split(processed_dataset, group_column="doc_id")
    print(f"Train split: {len(split_dataset['train'])} samples")
    print(f"Validation split: {len(split_dataset['validation'])} samples")

    split_dataset.save_to_disk(OUTPUT_DIR)
    print(f"Preprocessed dataset saved successfully to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()