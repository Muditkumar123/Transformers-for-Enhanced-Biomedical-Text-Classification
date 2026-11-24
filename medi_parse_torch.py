"""PyTorch port of medi_parse_tf.py (same 8 models, same training regimen).

TF-Hub encoders have no PyTorch release, so the closest PyTorch/HF checkpoints stand in:
  Universal Sentence Encoder v4  -> sentence-transformers/distiluse-base-multilingual-cased-v2 (512-d, distilled from USE)
  TF-Hub experts/bert/pubmed/2   -> microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract (BERT-base on PubMed abstracts)
Both encoders are frozen in the notebook, so their sentence vectors are computed once and cached.

Usage:  python medi_parse_torch.py --dataset 20k|200k [--models 0,1,...,7]
"""
import argparse, copy, math, os, random, string, time
from collections import Counter
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["20k", "200k"], default="20k")
ap.add_argument("--models", default="0,1,2,3,4,5,6,7")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

from common import ROOT, load_data, Results, line_number_one_hot
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "hf"))
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RUN = [int(m) for m in args.models.split(",")]
LOCAL = ROOT / "models" / "hf"  # pre-downloaded copies (see README); otherwise pulled from the HF hub
USE_REPO = str(LOCAL / "distiluse") if (LOCAL / "distiluse").exists() else "sentence-transformers/distiluse-base-multilingual-cased-v2"
BERT_REPO = str(LOCAL / "biomedbert") if (LOCAL / "biomedbert").exists() else "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
BS = 32

t0 = time.time()
(train_df, val_df, test_df), (y_train, y_val, y_test), class_names = load_data(args.dataset)
num_classes = len(class_names)
R = Results("torch", args.dataset, y_val, y_test, class_names)
splits = {"train": train_df, "val": val_df, "test": test_df}
sents = {k: d["text"].tolist() for k, d in splits.items()}

# ---------------- Keras TextVectorization equivalents ----------------
# default standardize = lowercase + delete ASCII punctuation, then whitespace split; 0 = pad, 1 = [UNK]
STRIP = str.maketrans("", "", string.punctuation)


def build_vocab(token_lists, max_tokens):
    return {w: i + 2 for i, (w, _) in enumerate(Counter(t for ts in token_lists for t in ts).most_common(max_tokens - 2))}


def vectorize(token_lists, vocab, seq_len):
    out = np.zeros((len(token_lists), seq_len), np.int32)
    for i, ts in enumerate(token_lists):
        ids = [vocab.get(t, 1) for t in ts[:seq_len]]
        out[i, :len(ids)] = ids
    return torch.from_numpy(out)


words = {k: [s.translate(STRIP).split() for s in v] for k, v in sents.items()}
chars = {k: [[c for c in s.translate(STRIP) if not c.isspace()] for s in v] for k, v in sents.items()}  # == split_chars + vectorizer
seq_len = int(np.percentile([len(s.split()) for s in sents["train"]], 95))
char_len = int(np.percentile([len(s) for s in sents["train"]], 95))
tok_vocab = build_vocab(words["train"], 68000)
char_vocab = build_vocab(chars["train"], len(string.ascii_lowercase + string.digits + string.punctuation) + 2)
V_TOK, V_CHR = len(tok_vocab) + 2, len(char_vocab) + 2
print(f"[prep] seq_len={seq_len} char_len={char_len} vocab={V_TOK} chars={V_CHR}")

data = {k: {"tok": vectorize(words[k], tok_vocab, seq_len), "chr": vectorize(chars[k], char_vocab, char_len),
            "ln": F.one_hot(torch.tensor(d["line_number"].values).clamp(max=14), 15).float()
                  * (torch.tensor(d["line_number"].values) < 15).float()[:, None],  # tf.one_hot: out of range -> zeros
            "tl": F.one_hot(torch.tensor(d["total_lines"].values).clamp(max=19), 20).float()
                  * (torch.tensor(d["total_lines"].values) < 20).float()[:, None],
            "y": torch.tensor(y)}
        for (k, d), y in zip(splits.items(), (y_train, y_val, y_test))}
l7 = line_number_one_hot(train_df, val_df, test_df)
l7[0] = np.abs(l7[0] - 0.01)  # notebook's apply_smoothing (train only)
for k, m in zip(splits, l7):
    data[k]["l7"] = torch.from_numpy(m)
del words, chars


# ---------------- frozen pretrained sentence encoders (cached) ----------------
@torch.no_grad()
def encode(kind):
    from transformers import AutoModel, AutoTokenizer
    paths = {k: ROOT / ".cache" / "emb" / f"{args.dataset}_{kind}_{k}.npy" for k in splits}
    if all(p.exists() for p in paths.values()):
        for k, p in paths.items():
            data[k][kind] = torch.from_numpy(np.load(p))
        return
    repo = USE_REPO if kind == "use" else BERT_REPO
    tok = AutoTokenizer.from_pretrained(repo)
    enc = AutoModel.from_pretrained(repo).to(dev).eval().half()
    if kind == "use":  # sentence-transformers head: mean pooling -> Dense(768->512, tanh)
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        dense_file = Path(repo) / "2_Dense" / "model.safetensors"
        w = load_file(dense_file if dense_file.exists() else hf_hub_download(repo, "2_Dense/model.safetensors"))
        dense = nn.Linear(768, 512).to(dev).half()
        dense.load_state_dict({"weight": w["linear.weight"], "bias": w["linear.bias"]})
    for k, p in paths.items():
        texts = sents[k]
        order = np.argsort([len(t) for t in texts])  # length-sorted batches = less padding
        out = np.zeros((len(texts), 512 if kind == "use" else 768), np.float16)
        for i in range(0, len(texts), 512):
            idx = order[i:i + 512]
            b = tok([texts[j] for j in idx], padding=True, truncation=True, max_length=128, return_tensors="pt").to(dev)
            o = enc(**b)
            if kind == "use":
                m = b["attention_mask"].unsqueeze(-1).half()
                v = torch.tanh(dense((o.last_hidden_state * m).sum(1) / m.sum(1)))
            else:
                v = o.pooler_output  # == TF-Hub BERT 'pooled_output'
            out[idx] = v.float().cpu().numpy()
        p.parent.mkdir(parents=True, exist_ok=True); np.save(p, out)
        data[k][kind] = torch.from_numpy(out)
        print(f"[encode] {kind} {k}: {out.shape} ({time.time()-t0:.0f}s)", flush=True)
    del enc; torch.cuda.empty_cache()


# ---------------- Keras-default initialisation, so layers start from the same distributions ----------------
def keras_init(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, -0.05, 0.05)
        elif isinstance(m, nn.LSTM):
            for name, p in m.named_parameters():
                if "weight_ih" in name: nn.init.xavier_uniform_(p)
                elif "weight_hh" in name: nn.init.orthogonal_(p)
                else:
                    nn.init.zeros_(p)
                    if "bias_ih" in name: p.data[m.hidden_size:2 * m.hidden_size] = 1.0  # unit_forget_bias
    return module


class CharBiLSTM(nn.Module):
    """Embedding(25) -> Bidirectional(LSTM(24)); with mask_zero the LSTM skips padding like Keras."""
    def __init__(self, mask_zero=True):
        super().__init__()
        self.mask_zero = mask_zero
        self.emb = nn.Embedding(V_CHR, 25)
        self.lstm = nn.LSTM(25, 24, batch_first=True, bidirectional=True)

    def forward(self, c):
        x = self.emb(c)
        if self.mask_zero:
            lens = (c != 0).sum(1).clamp(min=1).cpu()
            x = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(x)
        return torch.cat([h[0], h[1]], 1)


class Model1(nn.Module):  # Conv1D + token embeddings (Conv1D drops the Keras mask -> plain average)
    keys = ["tok"]
    def __init__(self):
        super().__init__()
        self.emb, self.conv, self.out = nn.Embedding(V_TOK, 128), nn.Conv1d(128, 64, 5, padding=2), nn.Linear(64, num_classes)
    def forward(self, t):
        return self.out(F.relu(self.conv(self.emb(t).transpose(1, 2))).mean(2))


class Model2(nn.Module):  # USE -> Dense(128) -> softmax
    keys = ["use"]
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(512, 128), nn.ReLU(), nn.Linear(128, num_classes))
    def forward(self, u):
        return self.net(u)


class Model3(nn.Module):  # Conv1D + char embeddings + global max pool
    keys = ["chr"]
    def __init__(self):
        super().__init__()
        self.emb, self.conv, self.out = nn.Embedding(V_CHR, 25), nn.Conv1d(25, 64, 5, padding=2), nn.Linear(64, num_classes)
    def forward(self, c):
        return self.out(F.relu(self.conv(self.emb(c).transpose(1, 2))).amax(2))


class Model4(nn.Module):  # hybrid USE token + char BiLSTM
    keys = ["use", "chr"]
    def __init__(self):
        super().__init__()
        self.tok, self.char = nn.Linear(512, 128), CharBiLSTM()
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(128 + 48, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, num_classes))
    def forward(self, u, c):
        return self.head(torch.cat([F.relu(self.tok(u)), self.char(c)], 1))


class Tribrid(nn.Module):  # Model 5 (USE) / Model 6 (PubMed BERT): token + char + line_number + total_lines
    def __init__(self, emb_key, emb_dim, char_mask):
        super().__init__()
        self.keys = ["ln", "tl", emb_key, "chr"]
        self.tok, self.char = nn.Linear(emb_dim, 128), CharBiLSTM(char_mask)
        self.z = nn.Sequential(nn.Linear(128 + 48, 256), nn.ReLU(), nn.Dropout(0.5))
        self.ln, self.tl, self.out = nn.Linear(15, 32), nn.Linear(20, 32), nn.Linear(32 + 32 + 256, num_classes)
    def forward(self, ln, tl, e, c):
        z = self.z(torch.cat([F.relu(self.tok(e)), self.char(c)], 1))
        return self.out(torch.cat([F.relu(self.ln(ln)), F.relu(self.tl(tl)), z], 1))


class LSTMStep(nn.Module):
    """Keras LSTM(activation='gelu') over a length-1 sequence from zero state (the notebook's dim-expanded USE vector).
    With h0=c0=0 the recurrent kernel and forget gate have no effect, so this is exact."""
    def __init__(self, d_in, units):
        super().__init__()
        self.k = nn.Linear(d_in, 4 * units)
        nn.init.xavier_normal_(self.k.weight); nn.init.zeros_(self.k.bias)
    def forward(self, x):
        i, _, g, o = self.k(x).chunk(4, -1)  # Keras gate order i, f, c, o
        c = torch.sigmoid(i) * F.gelu(g)
        return (torch.sigmoid(o) * F.gelu(c)).unsqueeze(1)


class AdditiveAttention(nn.Module):  # keras.layers.AdditiveAttention(use_scale=True)
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.empty(dim).uniform_(-math.sqrt(6 / (1 + dim)), math.sqrt(6 / (1 + dim))))
    def forward(self, q, v):
        scores = (self.scale * torch.tanh(q.unsqueeze(2) + v.unsqueeze(1))).sum(-1)
        return torch.softmax(scores, -1) @ v


class Model7(nn.Module):  # USE + LSTM self-attention block + current-line features
    keys = ["use", "l7"]
    def __init__(self, n_lines):
        super().__init__()
        self.q, self.v, self.att = LSTMStep(512, 64), LSTMStep(512, 64), AdditiveAttention(64)
        self.qbn, self.abn = nn.BatchNorm1d(64, eps=1e-3, momentum=0.01), nn.BatchNorm1d(64, eps=1e-3, momentum=0.01)
        self.dense, self.line, self.h, self.out = nn.Linear(512, 64), nn.Linear(n_lines, 32), nn.Linear(96, 64), nn.Linear(64, num_classes)
        for l in (self.dense, self.line, self.h, self.out):
            nn.init.xavier_normal_(l.weight); nn.init.zeros_(l.bias)
    def forward(self, u, lines):
        q, v = self.q(u), self.v(u)
        block = self.qbn(q.mean(1)) + self.abn(self.att(q, v).mean(1))
        x = block * F.gelu(self.dense(u))
        line = F.dropout(F.gelu(self.line(lines)), 0.2, self.training)
        return self.out(F.gelu(self.h(torch.cat([x, line], 1))))


# ---------------- Keras-style fit / evaluate ----------------
def batch_stream(n, shuffle):
    """Endless batch iterator: like Keras with steps_per_epoch, later epochs continue where the last one stopped."""
    while True:
        order = torch.randperm(n) if shuffle else torch.arange(n)
        for i in range(0, n, BS):
            yield order[i:i + BS]


def batch(split, keys, idx):
    return [data[split][k][idx].to(dev).float() if data[split][k].dtype in (torch.float16, torch.float32)
            else data[split][k][idx].to(dev).long() for k in keys]


@torch.no_grad()
def evaluate(model, split, steps=None, smoothing=0.0):
    model.eval()
    n = len(data[split]["y"]) if steps is None else min(steps * BS, len(data[split]["y"]))
    loss = correct = 0.0; preds = []
    for i in range(0, n, 1024):
        idx = torch.arange(i, min(i + 1024, n))
        logits = model(*batch(split, model.keys, idx)); y = data[split]["y"][idx].to(dev)
        loss += F.cross_entropy(logits, y, label_smoothing=smoothing, reduction="sum").item()
        p = logits.argmax(1); correct += (p == y).sum().item(); preds.append(p.cpu())
    return loss / n, correct / n, torch.cat(preds).numpy()


def fit(model, epochs, steps_frac=None, val_frac=None, smoothing=0.0, best_by_val_acc=False, shuffle=False, reduce_lr=False):
    model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, eps=1e-7)  # Keras Adam defaults
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.1, patience=1, threshold=1e-4, threshold_mode="abs") \
        if reduce_lr else None  # torch patience=1 == Keras patience=2 (Keras reduces once wait >= patience)
    n_batches = math.ceil(len(data["train"]["y"]) / BS)
    steps = int(steps_frac * n_batches) if steps_frac else n_batches
    val_steps = int(val_frac * math.ceil(len(data["val"]["y"]) / BS)) if val_frac else None
    stream, best, best_acc = batch_stream(len(data["train"]["y"]), shuffle), None, -1
    for ep in range(epochs):
        model.train(); t = time.time(); tl = tc = seen = 0
        for _ in range(steps):
            idx = next(stream)
            logits = model(*batch("train", model.keys, idx)); y = data["train"]["y"][idx].to(dev)
            loss = F.cross_entropy(logits, y, label_smoothing=smoothing)
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item() * len(idx); tc += (logits.argmax(1) == y).sum().item(); seen += len(idx)
        vl, va, _ = evaluate(model, "val", val_steps, smoothing)
        print(f"  epoch {ep+1}/{epochs}: loss={tl/seen:.4f} acc={tc/seen:.4f} val_loss={vl:.4f} val_acc={va:.4f} "
              f"({time.time()-t:.0f}s)", flush=True)
        if sched: sched.step(vl)
        if best_by_val_acc and va > best_acc:
            best_acc, best = va, copy.deepcopy(model.state_dict())
    if best is not None:
        model.load_state_dict(best)  # ModelCheckpoint(save_best_only, monitor=val_accuracy) + load_model
    return model


def finish(mid, name, model, t):
    R.save(mid, name, evaluate(model, "val")[2], evaluate(model, "test")[2], time.time() - t)


def wanted(mid):
    return mid in RUN and not R.done(mid)


if wanted(0):
    t = time.time()
    m0 = Pipeline([("tf-idf", TfidfVectorizer()), ("clf", MultinomialNB())]).fit(sents["train"], y_train)
    R.save(0, "baseline", m0.predict(sents["val"]), m0.predict(sents["test"]), time.time() - t)
if any(wanted(m) for m in (2, 4, 5, 7)):
    encode("use")
if wanted(6):
    encode("bert")

# Models 1-5: 10% of batches per epoch (unshuffled, continuing), 10% of val batches for monitoring — as in the notebook
if wanted(1):
    t = time.time(); finish(1, "custom_token_embed_conv1d", fit(keras_init(Model1()), 3, 0.1, 0.1), t)
if wanted(2):
    t = time.time(); finish(2, "pretrained_token_embed", fit(keras_init(Model2()), 3, 0.1, 0.1), t)
if wanted(3):
    t = time.time(); finish(3, "custom_char_embed_conv1d", fit(keras_init(Model3()), 3, 0.1, 0.1), t)
if wanted(4):
    t = time.time(); finish(4, "hybrid_char_token_embed", fit(keras_init(Model4()), 3, 0.1, 0.1), t)
if wanted(5):
    t = time.time()
    finish(5, "tribrid_pos_char_token_embed",
           fit(keras_init(Tribrid("use", 512, True)), 2, 0.1, 0.1, smoothing=0.2, best_by_val_acc=True), t)
if wanted(6):  # full epochs, 15% of val batches for checkpoint selection; notebook's char embedding has no mask here
    t = time.time()
    finish(6, "Transformer-BERT model",
           fit(keras_init(Tribrid("bert", 768, False)), 5, None, 0.15, smoothing=0.2, best_by_val_acc=True), t)
if wanted(7):
    t = time.time()
    torch.manual_seed(args.seed)
    finish(7, "Self Attention model",
           fit(Model7(data["train"]["l7"].shape[1]), 5, None, None, smoothing=0.1, best_by_val_acc=True,
               shuffle=True, reduce_lr=True), t)

print(f"[done] total {(time.time()-t0)/60:.1f} min. Run: python make_report.py --framework torch --dataset {args.dataset}")
