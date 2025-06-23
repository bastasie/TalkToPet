# Machine Learning Pipeline for Animal Vocalization Classification

"""A simple pipeline to train and evaluate classifiers on animal vocalization data.

The dataset should be organized with one subdirectory per species label,
containing audio files (wav or mp3). Example structure:

    dataset_root/
        wolf/
            audio1.wav
            audio2.wav
        dolphin/
            audio1.wav
            audio2.wav

This script extracts mel spectrogram features, trains a classifier,
and saves the resulting model with joblib.
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple

import joblib
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def load_audio_files(dataset_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load audio files and corresponding labels."""
    features = []
    labels = []
    label_names = []

    for label_dir in sorted(dataset_path.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        label_names.append(label)
        for audio_file in label_dir.glob("*.wav"):
            y, sr = librosa.load(audio_file, sr=None)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=sr / 2)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            features.append(mel_db.flatten())
            labels.append(label)
        for audio_file in label_dir.glob("*.mp3"):
            y, sr = librosa.load(audio_file, sr=None)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=sr / 2)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            features.append(mel_db.flatten())
            labels.append(label)
    return np.array(features), np.array(labels), label_names


def train_classifier(X: np.ndarray, y: np.ndarray) -> RandomForestClassifier:
    """Train a RandomForest classifier on the extracted features."""
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X, y)
    return clf


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an animal vocalization classifier")
    parser.add_argument("dataset", type=Path, help="Path to dataset root directory")
    parser.add_argument("model_out", type=Path, help="Path to save trained model")
    args = parser.parse_args()

    X, y, label_names = load_audio_files(args.dataset)
    if len(X) == 0:
        raise RuntimeError("Dataset directory does not contain any audio files")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = train_classifier(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=label_names))

    joblib.dump(clf, args.model_out)
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    main()
