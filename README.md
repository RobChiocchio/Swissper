# Swiss German Dialect Classifier

<!--
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-lg-dark.svg)](https://huggingface.co/RobChio/swissgerman-dialect-classifier)
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-lg-dark.svg)](https://huggingface.co/RobChio/swissgerman-dialect-classifier-xls-r)
[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-lg-dark.svg)](https://huggingface.co/RobChio/swissgerman-dialect-classifier-tiny)
-->

## [swissgerman-dialect-classifier-xls-r](https://github.com/RobChiocchio/Swissper/blob/main/swissgerman-dialect-classifier-xls-r.ipynb)

This model is a fine-tuned version of [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m) for Swiss German dialect classification.
It achieves the following results on the evaluation set:
- Loss: 2.3748
- F1: 0.4512
- Accuracy: 0.4292

## Training and evaluation data

* [SwissDial Dataset](https://mtc.ethz.ch/publications/open-source/swiss-dial.html) (3 hours per dialect, ~25 hours)
* [The ArchiMob Corpus](https://www.spur.uzh.ch/en/research/projects-all/lab-projects/ArchiMob0.html) (43 interviews, ~70 hours)
* [All Swiss German Dialects Test Set](https://www.cs.technik.fhnw.ch/i4ds-datasets) (13 hours)

~15% of the ArchiMob corpus was held out for validation. No speakers appeared in both the train and validation splits.

### Preprocessing

* Re-encoded to 16khz
* Re-labeled to match convention
* For Whisper, previously pre-processed to mel-spectograms (filesize too big)

### Labels

The region labels are from SwissDial:

* 0 AG Aargau
* 1 BE Bern
* 2 BS Basel
* 3 GR Graubünden
* 4 LU Lucerne
* 5 SG St. Gallen
* 6 VS Wallis (Valais)
* 7 ZH Zurich

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0005
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 32
- optimizer: Use OptimizerNames.ADAMW_TORCH_FUSED with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 500
- num_epochs: 5
- label_smoothing_factor: 0.1

### Training results

| Training Loss | Epoch  | Step | Validation Loss | F1     | Accuracy |
|:-------------:|:------:|:----:|:---------------:|:------:|:--------:|
| 1.4528        | 0.4631 | 1000 | 2.4189          | 0.3165 | 0.3058   |
| 1.2588        | 0.9261 | 2000 | 2.6695          | 0.4099 | 0.3231   |
| 1.1551        | 1.3890 | 3000 | 2.3748          | 0.4512 | 0.4292   |
| 1.1238        | 1.8520 | 4000 | 2.7390          | 0.3865 | 0.2923   |
| 1.0970        | 2.3149 | 5000 | 2.8112          | 0.3664 | 0.3085   |
| 1.0865        | 2.7780 | 6000 | 2.4348          | 0.4452 | 0.4023   |


### Framework versions

- PEFT 0.21.0
- Transformers 5.17.0
- Pytorch 2.11.0+cu128
- Datasets 5.0.1
- Tokenizers 0.23.2

## What's next
* Get and train with [STT4SG-350 and SDS-200](https://swissnlp.org/home/activities/datasets/) (see Akeret for issues)
* Get access to data from [Swiss German Dialects Across Time and Space (SDATS)](https://www.csls.unibe.ch/research/csls_projects/swiss_german_dialects_across_time_and_space_sdats/index_eng.html)
* Select new optimal [regional divisions](https://dialektkarten.ch/dmviewer/swg/index.en.html#algo=WARD&app=cluster&dataset=TOT&nclu=4&sim=JRIW&sim2=JRIW&version=v3) and re-label data

### After
* Transcription task
* Train a Whisper adapter for each region for ASR, swap adapters by region
* Hierarchical stacking of LoRA adapters
* [GLoRIA](https://doi.org/10.1109/ICASSP55912.2026.11462008)
* Transcribe dialectual Swiss German instead of "translating" to High German

