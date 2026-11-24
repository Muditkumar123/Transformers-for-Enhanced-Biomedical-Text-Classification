# MediParse: Transformers for Enhanced Biomedical Text Classification

Code and reproduction for our paper:

> M. Kumar, Shubham, L. M. Saini and P. Mishra, "Transformers for Enhanced Biomedical Text Classification,"
> *2025 4th International Conference on Innovative Mechanisms for Industry Applications (ICIMIA)*, IEEE, pp. 1205–1212, 2025.
> DOI: [10.1109/ICIMIA67127.2025.11200801](https://doi.org/10.1109/ICIMIA67127.2025.11200801)

The task is sequential sentence classification of PubMed RCT abstracts: label every sentence as
BACKGROUND, OBJECTIVE, METHODS, RESULTS or CONCLUSIONS. The paper compares eight models, from a TF-IDF baseline up to
our proposed **PubMed-BERT + character + positional ("tribrid") model (Model 6)**.

This repository contains:

- `medi_parse_tf.py`: the original TensorFlow/Keras notebook (`original/Medi_Parse_paper_code.ipynb`) as one runnable script, same architectures and training settings.
- `medi_parse_torch.py`: a PyTorch port of the same eight models.
- Results for both frameworks on **PubMed 20k RCT** (the paper's setting) and the **full PubMed 200k RCT**.

## Results

Validation set, weighted F1 (the paper's metric). Test-set numbers and precision/recall are in `results/<framework>/<dataset>/RESULTS.md`.

| # | Model | Paper (20k) | TF 20k | PyTorch 20k | TF 200k | PyTorch 200k |
|---|---|---|---|---|---|---|
| 6 | **BERT + char + positional (proposed)** | **0.87** | **0.884** | **0.882** | **0.898** | **0.899** |
| 7 | USE + LSTM self-attention + position | ≈0.85 | 0.843 | 0.873 | 0.858 | 0.885 |
| 5 | Tribrid: USE + char + positional | ≈0.84 | 0.818 | 0.844 | 0.859 | 0.876 |
| 1 | Conv1D, token embeddings | ≈0.78 | 0.786 | 0.782 | 0.835 | 0.835 |
| 4 | Hybrid: USE + char BiLSTM | ≈0.73 | 0.731 | 0.771 | 0.779 | 0.819 |
| 2 | USE feature extractor | ≈0.71 | 0.712 | 0.759 | 0.774 | 0.823 |
| 0 | TF-IDF + Multinomial NB | ≈0.70 | 0.699 | 0.699 | 0.739 | 0.739 |
| 3 | Conv1D, character embeddings | ≈0.65 | 0.648 | 0.641 | 0.744 | 0.741 |

"Paper" values for Models 0–5 and 7 are read off Fig. 4; Model 6 is from Table 1.

Model 6 per-class F1 (paper Table 1 vs. this reproduction, validation set):

| Class | Paper | TF 20k | PyTorch 20k | TF 200k | PyTorch 200k |
|---|---|---|---|---|---|
| BACKGROUND | 0.77 | 0.79 | 0.78 | 0.76 | 0.76 |
| CONCLUSIONS | 0.89 | 0.91 | 0.91 | 0.92 | 0.91 |
| METHODS | 0.91 | 0.93 | 0.93 | 0.94 | 0.95 |
| OBJECTIVE | 0.63 | 0.62 | 0.62 | 0.71 | 0.71 |
| RESULTS | 0.90 | 0.92 | 0.92 | 0.93 | 0.93 |
| Accuracy | 0.87 | 0.89 | 0.89 | 0.90 | 0.90 |

Findings:

- **The paper's result reproduces.** On 20k the TF code gives Model 6 F1 0.884 / accuracy 88.7% (paper: 0.87), and the model ranking matches Fig. 4.
- **The two frameworks agree** wherever they use the same kind of encoder (Models 0, 1, 3, 6: within 0.01 F1 on both datasets).
- **The PyTorch Models 2, 4, 5 and 7 score 0.02–0.05 higher.** These are the models built on the Universal Sentence Encoder, which has no PyTorch release; the port uses distiluse (see below), which gives better sentence vectors for this task. The TF numbers are the faithful ones.
- **More data helps every model.** The full 200k set adds 1–10 F1 points; Model 6 reaches 0.898 F1 / 89.9% accuracy in TF and 0.899 / 90.0% in PyTorch, and the hardest class (OBJECTIVE, often confused with BACKGROUND) improves most (0.62 → 0.71).

Figures for each run (F1 comparison as in Fig. 4, all metrics as in Fig. 5, Model 6 confusion matrix as in Fig. 6) are in
`results/<framework>/<dataset>/fig*.png`.

## Models

All models follow the notebook exactly; dimensions are those in the paper (character embeddings 25-d, char BiLSTM 2×24, etc.).

| # | Inputs | Architecture |
|---|---|---|
| 0 | text | TF-IDF → Multinomial Naive Bayes |
| 1 | tokens | TextVectorization (68k vocab) → Embedding 128 → Conv1D(64, k=5) → GlobalAvgPool → softmax |
| 2 | sentence | USE (frozen, 512-d) → Dense 128 → softmax |
| 3 | characters | char vectorizer → Embedding 25 → Conv1D(64, k=5) → GlobalMaxPool → softmax |
| 4 | sentence + chars | [USE → Dense 128] ⊕ [char Embedding 25 → BiLSTM 24] → Dropout → Dense 128 → Dropout → softmax |
| 5 | sentence + chars + line_number + total_lines | [USE → Dense 128] ⊕ [char BiLSTM] → Dense 256 → Dropout, ⊕ [one-hot line_number → Dense 32] ⊕ [one-hot total_lines → Dense 32] → softmax; label smoothing 0.2 |
| 6 | same as 5 | as Model 5, with USE replaced by **PubMed-BERT** (`pooled_output`, frozen) |
| 7 | sentence + line_number | USE → LSTM query/value (gelu) → additive attention → BatchNorm + residual, × Dense 64(USE), ⊕ line features → Dense 64 → softmax; label smoothing 0.1 |

Training follows the notebook: Adam (lr 1e-3), batch size 32; Models 1–4 train 3 epochs on 10% of the batches, Model 5 trains
2 such epochs, Model 6 trains 5 full epochs, Model 7 trains 5 full epochs with ReduceLROnPlateau. Models 5–7 keep the
checkpoint with the best validation accuracy.

## Running it

```bash
./download.sh                     # PubMed RCT data + pretrained encoders into data/ and models/
```

TensorFlow (original code). Needs Python 3.10 and TF 2.15:

```bash
conda create -n mediparse-tf python=3.10 -y && conda activate mediparse-tf
pip install -r requirements-tf.txt
python medi_parse_tf.py --dataset 20k
python medi_parse_tf.py --dataset 200k --cache-encoders
python make_report.py --framework tf --dataset 20k
```

PyTorch:

```bash
conda create -n mediparse-torch python=3.10 -y && conda activate mediparse-torch
pip install -r requirements-torch.txt
python medi_parse_torch.py --dataset 20k
python medi_parse_torch.py --dataset 200k
python make_report.py --framework torch --dataset 20k
```

`--models 6,7` runs a subset. Every finished model is saved to `results/.../model_<n>.json`; a rerun skips models that already
have a result, so an interrupted run resumes where it stopped.

Runtimes on one shared A100 40 GB: TF 20k ≈ 1 h 50 min (Model 6 alone 90 min), PyTorch 20k ≈ 12 min, PyTorch 200k ≈ 2 h 10 min, TF 200k ≈ 4.5 h with `--cache-encoders` (BERT encoding 75 min, Model 6 training 2 h).

## Differences from the notebook

Behaviour is kept the same except for the following:

1. **`--cache-encoders` (TF, used for 200k only).** The notebook runs the frozen USE/BERT inside the model, so BERT does a forward pass on every batch of every epoch.
   On 200k that is about 18 h for Model 6. Because the encoders are frozen (`trainable=False`, `training=False`), their output for a
   sentence never changes, so the flag computes it once and trains on the cached vectors. The math is the same. The 20k TF run uses the notebook's inline setup.
2. **Checkpoints store weights only** (`.weights.h5`), and the best weights are reloaded. The notebook saved full SavedModels, which is fragile with TF-Hub layers.
3. **Model 7's line one-hot size is computed** from the data (31 on 20k) instead of hard-coded, so the 200k data works too. The
   notebook's `train_model(..., callbacks=[CALLBACKS])` call passes `callbacks` twice; the script passes it once.
4. **Fresh embedding layers per model.** The notebook reuses one `char_embed` layer object across Models 3–5, so later models
   start from the earlier model's trained weights. Here each model initialises its own, so results do not depend on run order.
5. **Metrics** are those of the notebook's `calculate_results` (accuracy, weighted precision/recall/F1). They are now computed on both the validation and the test split.

PyTorch port specifics:

| TF component | PyTorch replacement |
|---|---|
| Universal Sentence Encoder v4 (TF-Hub) | `sentence-transformers/distiluse-base-multilingual-cased-v2` (512-d, distilled from USE) |
| `experts/bert/pubmed/2` (TF-Hub) | `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract`, `pooler_output` (BERT-base pretrained on PubMed abstracts, with trained NSP pooler) |
| `TextVectorization` | same standardisation (lowercase, strip ASCII punctuation, whitespace split), vocabulary by frequency, 0 = pad, 1 = OOV |
| Keras masking in the char BiLSTM | packed sequences |
| Keras initialisers / Adam ε / BatchNorm momentum | set to Keras defaults (Glorot, orthogonal recurrent, forget bias 1, ε = 1e-7, momentum 0.99) |

The frozen encoders' outputs are cached in both frameworks (fp16 in PyTorch). Model 7's LSTM with `activation='gelu'` runs on a
length-1 sequence from a zero state, so it is implemented exactly as a single gated step.

## Repository layout

```
medi_parse_tf.py       TensorFlow/Keras implementation (original notebook as a script)
medi_parse_torch.py    PyTorch implementation
common.py              data parsing, metrics, result files (shared)
make_report.py         tables + Fig. 4/5/6-style plots from results/
download.sh            data and pretrained encoders
original/              the original Colab notebook
results/{tf,torch}/{20k,200k}/   per-model JSON, RESULTS.md, summary CSVs, figures
logs/                  full training logs of the runs reported above (TF 200k ran in four resumed
                       attempts: models 0–1 in attempt1, 2–4 in attempt2, 5 in attempt3, 6–7 in tf_200k.log)
```

## Data

PubMed 200k RCT (Dernoncourt & Lee, 2017), [github.com/Franck-Dernoncourt/pubmed-rct](https://github.com/Franck-Dernoncourt/pubmed-rct),
in the `numbers_replaced_with_at_sign` version, with its official train/dev/test split. The 20k subset is the paper's setting.

## Citation

```bibtex
@inproceedings{kumar2025mediparse,
  author    = {Kumar, Mudit and Shubham and Saini, Lalit Mohan and Mishra, Parth},
  title     = {Transformers for Enhanced Biomedical Text Classification},
  booktitle = {2025 4th International Conference on Innovative Mechanisms for Industry Applications (ICIMIA)},
  pages     = {1205--1212},
  year      = {2025},
  publisher = {IEEE},
  doi       = {10.1109/ICIMIA67127.2025.11200801}
}
```
