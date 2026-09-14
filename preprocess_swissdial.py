from transformers import WhisperForConditionalGeneration, WhisperForAudioClassification, WhisperProcessor
import torch

import datasets
from datasets import load_dataset, load_from_disk, Audio

model_id = "Flix-AI/flix-swissgerman-full"

processor = WhisperProcessor.from_pretrained(model_id)

labels = {
    0: "AG",
    1: "BE",
    2: "BS",
    3: "GR",
    4: "LU",
    5: "SG",
    6: "VS",
    7: "ZH",
}

id2label = labels
label2id = {v: k for k, v in labels.items()}

def extract_features(batch):
    audios = []

    for sample in batch["audio"]:
        audio = sample.get_all_samples()
        waveform = audio.data.squeeze().numpy()
        audios.append(waveform)

    inputs = processor(
        audios,
        sampling_rate=16000,
        return_tensors="np",
    )

    labels = [
        #label2id[code.upper()]
        code for code in batch["dialect_code"]
    ]

    return {
        "input_features": inputs.input_features,
        "labels": labels,
    }

class FastFeatureCollator:
    def __call__(self, features):
        input_features = torch.stack([
            torch.from_numpy(f["input_features"])
            for f in features
        ]).to(torch.bfloat16)

        labels = torch.tensor(
            [f["labels"] for f in features],
            dtype=torch.long,
        )

        return {
            "input_features": input_features,
            "labels": labels,
        }

def main():
    raw_dataset = load_dataset("audiofolder", data_dir="./SwissDial", split="train")
    raw_dataset = raw_dataset.rename_column("label", "dialect_code")
    raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=16000))


    from collections import Counter


    counts = Counter(raw_dataset["dialect_code"])
    total = len(raw_dataset)

    for label, count in sorted(counts.items()):
        print(
            id2label[label],
            count,
            f"{count / total:.1%}",
        )

    raw_dataset = raw_dataset.map(
        extract_features,
        batched=True,
        batch_size=64,
        remove_columns=raw_dataset.column_names,
        num_proc=6,
    )

    raw_dataset.set_format("torch") # untested

    raw_dataset.save_to_disk("./bucket/swissdial")

if __name__ == "__main__":
    main()
