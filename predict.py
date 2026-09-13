import gradio as gr

from transformers import WhisperForAudioClassification, WhisperProcessor, WhisperFeatureExtractor

# import torch

# import datasets
# from datasets import load_dataset, load_from_disk, Audio

from transformers import pipeline

model_id = "RobChio/swissgerman-dialect-classifier"
base_model_id = "Flix-AI/flix-swissgerman-full"

model = WhisperForAudioClassification.from_pretrained(model_id)
feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model_id)
processor = WhisperProcessor.from_pretrained(base_model_id)

# processor = WhisperProcessor.from_pretrained(model_id) # set to original
# model = WhisperForAudioClassification.from_pretrained(model_id)


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
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="Audio Classification with Transformers",
    description="Upload an audio file to classify its content using a fine-tuned transformer model."
)

if __name__ == "__main__":
    demo.launch()