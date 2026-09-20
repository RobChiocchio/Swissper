import gradio as gr

#from transformers import WhisperForAudioClassification, WhisperProcessor, WhisperFeatureExtractor
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from peft import PeftModel

# import torch

# import datasets
# from datasets import load_dataset, load_from_disk, Audio

from transformers import pipeline

# adapter_id = "RobChio/swissgerman-dialect-classifier-tiny"
#base_model_id = "Flix-AI/flix-swissgerman-full"
# base_model_id = "openai/whisper-tiny"
adapter_id = "RobChio/swissgerman-dialect-classifier-xls-r"
base_model_id = "facebook/wav2vec2-xls-r-300m"

# model = WhisperForAudioClassification.from_pretrained(model_id)
# feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model_id)
# processor = WhisperProcessor.from_pretrained(base_model_id)

# processor = WhisperProcessor.from_pretrained(model_id) # set to original
# model = WhisperForAudioClassification.from_pretrained(model_id)

# model = AutoModelForAudioClassification.from_pretrained(model_id)

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

base_model = AutoModelForAudioClassification.from_pretrained(
    base_model_id,
    torch_dtype="auto",
    num_labels=8,
    label2id=label2id,
    id2label=id2label,
)

peft_model = PeftModel.from_pretrained(
    base_model,
    adapter_id,
)

model = peft_model.merge_and_unload()

feature_extractor = AutoFeatureExtractor.from_pretrained(base_model_id)

# Load a pre-trained audio classification model pipeline
pipe = pipeline("audio-classification", model=model, feature_extractor=feature_extractor)

def classify_audio(filepath):
    preds = pipe(filepath)
    # Format predictions as a dictionary for Gradio's Label component
    return {p["label"]: p["score"] for p in preds}

# Set up the Gradio UI
demo = gr.Interface(
    fn=classify_audio,
    inputs=gr.Audio(type="filepath", label="Upload or Record Audio"),
    outputs=gr.Label(num_top_classes=8, label="Predictions"),
    title="Audio Classification with Transformers",
    description="Upload an audio file to classify its content using a fine-tuned transformer model."
)

if __name__ == "__main__":
    demo.launch(share=False)