# Inferencia independiente para Traveler Insights UPTC 2026
# Uso: python inferencia.py --modelo modelo_final --entrada submission.csv --salida submission_envio.csv
import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def limpiar(t):
    t = re.sub(r"https?://\S+|www\.\S+", " ", str(t))
    return re.sub(r"\s+", " ", t).strip()


class Clasificador:
    def __init__(self, ruta, device=None):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(ruta)
        self.model = AutoModelForSequenceClassification.from_pretrained(ruta).to(self.device).eval()
        with open(os.path.join(ruta, "config_inferencia.json"), encoding="utf-8") as f:
            self.max_len = json.load(f)["max_len"]

    @torch.no_grad()
    def probs(self, textos, batch_size=64):
        textos = [limpiar(t) for t in textos]
        ids = self.tokenizer(textos, truncation=True, max_length=self.max_len)["input_ids"]
        orden = np.argsort([len(x) for x in ids])
        pad = self.tokenizer.pad_token_id
        salida = np.zeros((len(ids), self.model.config.num_labels), dtype=np.float32)
        for i in range(0, len(orden), batch_size):
            idx = orden[i:i + batch_size]
            largo = max(len(ids[j]) for j in idx)
            x = torch.full((len(idx), largo), pad, dtype=torch.long)
            m = torch.zeros((len(idx), largo), dtype=torch.long)
            for r, j in enumerate(idx):
                x[r, :len(ids[j])] = torch.tensor(ids[j])
                m[r, :len(ids[j])] = 1
            logits = self.model(input_ids=x.to(self.device), attention_mask=m.to(self.device)).logits
            salida[idx] = torch.softmax(logits.float(), dim=-1).cpu().numpy()
        return salida


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelo", required=True)
    ap.add_argument("--entrada", required=True)
    ap.add_argument("--salida", required=True)
    a = ap.parse_args()
    df = pd.read_csv(a.entrada)
    clf = Clasificador(a.modelo)
    p = clf.probs(df["Comentario"].tolist())
    pd.DataFrame({"ID": df["ID"], "Sentimiento": p.argmax(1)}).to_csv(a.salida, index=False)
    print("Archivo escrito:", a.salida, "con", len(df), "filas")


if __name__ == "__main__":
    main()
