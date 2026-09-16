from pathlib import Path
from collections import Counter
from datasets import load_dataset, load_from_disk, Audio

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

AUDIO_DIR = Path("./SwissDial_16k")  # Point to pre-resampled 16kHz audio
OUTPUT_DIR = Path("./dataset/swiss-dial-preprocessed")

def main():
    raw_dataset = load_dataset("audiofolder", data_dir=AUDIO_DIR, split="train")
    raw_dataset = raw_dataset.rename_column("label", "labels")
    raw_dataset = raw_dataset.cast_column("audio", Audio(sampling_rate=16000))


    from collections import Counter


    counts = Counter(raw_dataset["labels"])
    total = len(raw_dataset)

    for label, count in sorted(counts.items()):
        print(
            id2label[label],
            count,
            f"{count / total:.1%}",
        )

    raw_dataset.save_to_disk(OUTPUT_DIR)

if __name__ == "__main__":
    main()
    # TODO: load transcripts