"""Create a small PDF of FAKE patient records for demos and tests.

All names and values are randomly generated. No real patient data.

Usage:
    python scripts/make_sample_pdf.py            # writes data/sample_medical_records.pdf
    python scripts/make_sample_pdf.py out.pdf 40 # custom path and number of patients
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

FIRST = ["Jason", "Maria", "Curtis", "Aisha", "Ravi", "Emily", "Noah", "Priya", "Lucas", "Sara",
         "Daniel", "Meera", "Omar", "Grace", "Arjun", "Hannah", "Kevin", "Lina", "Victor", "Zoe"]
LAST = ["Fleming", "Lopez", "Carter", "Khan", "Iyer", "Brooks", "Patel", "Nguyen", "Silva", "Reed",
        "Moore", "Das", "Hassan", "Price", "Menon", "Wells", "Ortiz", "Kim", "Shaw", "Bell"]
DISEASES = {
    "Tuberculosis": ["Cough", "Chest pain", "Fever", "Sweating", "Loss of appetite", "Fatigue",
                     "Shortness of breath"],
    "Pneumonia": ["Cough", "Fever", "Shortness of breath", "Chest pain", "Fatigue"],
    "Diabetes": ["Fatigue", "Dizziness", "Nausea", "Headache"],
    "Hypertension": ["Headache", "Dizziness", "Chest pain", "Nausea"],
    "Influenza": ["Fever", "Runny nose", "Sore throat", "Muscle ache", "Headache"],
    "Asthma": ["Shortness of breath", "Cough", "Chest pain"],
    "Arthritis": ["Joint pain", "Muscle ache", "Fatigue"],
}
TREATMENTS = {
    "Tuberculosis": ["Antibiotics", "Corticosteroids", "Physiotherapy"],
    "Pneumonia": ["Antibiotics", "Oxygen therapy", "Corticosteroids"],
    "Diabetes": ["Insulin", "Metformin", "Diet plan"],
    "Hypertension": ["ACE inhibitors", "Beta blockers", "Diet plan"],
    "Influenza": ["Antivirals", "Rest", "Fluids"],
    "Asthma": ["Inhaler", "Corticosteroids"],
    "Arthritis": ["Painkillers", "Physiotherapy"],
}


def make_records(n: int, seed: int = 7) -> list[str]:
    rng = random.Random(seed)
    records = []
    for i in range(1, n + 1):
        disease = rng.choice(list(DISEASES))
        symptoms = ", ".join(rng.sample(DISEASES[disease], k=min(3, len(DISEASES[disease]))))
        treatment = ", ".join(rng.sample(TREATMENTS[disease], k=2))
        temp = round(rng.uniform(97.5, 104.0), 1) if disease in {"Tuberculosis", "Pneumonia", "Influenza"} \
            else round(rng.uniform(97.5, 99.5), 1)
        records.append(
            f"Patient {i}: Name: {rng.choice(FIRST)} {rng.choice(LAST)}<br/>"
            f"Age: {rng.randint(18, 85)}<br/>"
            f"Disease: {disease}<br/>"
            f"Symptoms: {symptoms}<br/>"
            f"Checkup: Blood Pressure - {rng.randint(105, 160)}/{rng.randint(65, 100)} mmHg, "
            f"Heart Rate - {rng.randint(60, 110)} bpm, Temperature - {temp} °F<br/>"
            f"Treatment: {treatment}<br/>"
            f"Follow-up: {rng.choice([1, 2, 3, 4, 6])} week(s)"
        )
    return records


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sample_medical_records.pdf")
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    out.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    story = [Paragraph("Synthetic Medical Records (fake data for testing)", styles["Title"]), Spacer(1, 12)]
    for record in make_records(count):
        story += [Paragraph(record, styles["Normal"]), Spacer(1, 10)]
    SimpleDocTemplate(str(out), pagesize=A4).build(story)
    print(f"Wrote {count} fake patient records to {out}")


if __name__ == "__main__":
    main()
