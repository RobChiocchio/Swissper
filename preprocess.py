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
        label2id[code.upper()]
        for code in batch["dialect_code"]
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
    train_dataset = load_dataset("i4ds/swiss-german-city-sentences_train", split="train")
    eval_dataset = load_dataset("i4ds/swiss-german-city-sentences_val", split="train")

    train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
    eval_dataset = eval_dataset.cast_column("audio", Audio(sampling_rate=16000))

    train_dataset = train_dataset.map(
        extract_features,
        batched=True,
        batch_size=64,
        remove_columns=train_dataset.column_names,
        num_proc=6,
    )
    
    eval_dataset = eval_dataset.map(
        extract_features,
        batched=True,
        batch_size=64,
        remove_columns=eval_dataset.column_names,
        num_proc=6,
    )
    
    train_dataset.save_to_disk("./bucket/preprocessed_swiss_train")
    eval_dataset.save_to_disk("./bucket/preprocessed_swiss_eval")


if __name__ == "__main__":
    main()
