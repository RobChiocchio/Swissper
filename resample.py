import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import torchaudio

#SRC_DIR = Path("ArchiMob/audio_segmented_anonymized")
#DST_DIR = Path("ArchiMob/audio_segmented_16k")
SRC_DIR = Path("SwissDial")
DST_DIR = Path("SwissDial_16k")
TARGET_SR = 16000

def process_file(src_path: Path) -> None:
    rel_path = src_path.relative_to(SRC_DIR)
    dst_path = DST_DIR / rel_path
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    if dst_path.exists():
        return

    waveform, sr = torchaudio.load(src_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != TARGET_SR:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=TARGET_SR)
        waveform = resampler(waveform)

    torchaudio.save(dst_path, waveform, TARGET_SR, encoding="PCM_S", bits_per_sample=16)

def main():
    wav_files = list(SRC_DIR.rglob("*.wav"))
    print(f"Found {len(wav_files)} WAV files across subfolders.")

    num_workers = min(32, os.cpu_count() or 4)
    print(f"Resampling using {num_workers} parallel workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(executor.map(process_file, wav_files))

    print(f"Resampling complete. Files written to {DST_DIR}")

if __name__ == "__main__":
    main()