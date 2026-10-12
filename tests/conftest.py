import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeEmbedder:
    """Fast bag-of-words embedder, so tests do not download a model."""

    name = "fake-hash-embedder"
    dim = 256

    def encode(self, texts, kind="document"):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for row, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                out[row, h % self.dim] += 1.0
        return out


class FakeLLM:
    def __init__(self, reply="Cough and fever [1]."):
        self.reply = reply
        self.calls = []

    def generate(self, system, prompt):
        self.calls.append((system, prompt))
        return self.reply


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def sample_pages():
    return [
        (1, "Synthetic records\n\nPatient 1: Name: Ana Diaz\nAge: 40\nDisease: Tuberculosis\n"
            "Symptoms: Cough, Fever\nTreatment: Antibiotics\n\nPatient 2: Name: Ben Ray\nAge: 55\n"
            "Disease: Diabetes\nSymptoms: Fatigue"),
        (2, "Treatment: Insulin\nFollow-up: 2 week(s)\n\nPatient 3: Name: Cy Wu\nAge: 30\n"
            "Disease: Asthma\nSymptoms: Shortness of breath\nTreatment: Inhaler"),
    ]
