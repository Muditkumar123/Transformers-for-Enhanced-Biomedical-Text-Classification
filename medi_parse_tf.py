"""Recreation of "Transformers for Enhanced Biomedical Text Classification" (ICIMIA 2025).

Ported from the authors' Colab notebook (Medi_Parse_paper_code.ipynb) to a runnable script.
Architectures and training settings follow the notebook; see README for the few deviations.

Usage:  python medi_parse_tf.py --dataset 20k      (PubMed 20k RCT)
        python medi_parse_tf.py --dataset 200k     (PubMed 200k RCT, full)
"""
import argparse, json, os, random, string, time
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["20k", "200k"], default="20k")
ap.add_argument("--models", default="0,1,2,3,4,5,6,7", help="comma-separated model ids to run")
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--cache-encoders", action="store_true",
                help="run the frozen USE/BERT encoders once and train on their cached outputs "
                     "(same math as the notebook, used for the 200k run where inline BERT takes ~18h)")
args = ap.parse_args()

from common import ROOT, load_data, Results, line_number_one_hot
os.environ.setdefault("TFHUB_CACHE_DIR", str(ROOT / ".cache" / "tfhub"))
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text  # noqa: F401  registers ops needed by the BERT preprocessor
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)  # GPUs on this server are shared
random.seed(args.seed); np.random.seed(args.seed); tf.random.set_seed(args.seed)
RUN = [int(m) for m in args.models.split(",")]

def hub_src(local, url):  # prefer a pre-extracted copy in models/tfhub (see README), else download from TF-Hub
    return str(ROOT / "models" / "tfhub" / local) if (ROOT / "models" / "tfhub" / local).exists() else url


USE_URL = hub_src("use4", "https://tfhub.dev/google/universal-sentence-encoder/4")
BERT_PRE_URL = hub_src("bert_preprocess3", "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
BERT_URL = hub_src("bert_pubmed2", "https://tfhub.dev/google/experts/bert/pubmed/2")

t0 = time.time()
(train_df, val_df, test_df), (y_train, y_val, y_test), class_names = load_data(args.dataset)
num_classes = len(class_names)
R = Results("tf", args.dataset, y_val, y_test, class_names)
save_result = R.save
CKPT = R.ckpt
train_sentences, val_sentences, test_sentences = (d["text"].tolist() for d in (train_df, val_df, test_df))
# same column order as the notebook's OneHotEncoder (both sort classes alphabetically)
train_labels_one_hot, val_labels_one_hot, test_labels_one_hot = (
    tf.one_hot(y, num_classes).numpy() for y in (y_train, y_val, y_test))


def predict(model, ds):
    return np.argmax(model.predict(ds, verbose=0), axis=1)


def ds(x, y, bs=32):
    with tf.device("/CPU:0"):  # keep source arrays in host RAM; at 200k they do not fit as GPU constants
        return tf.data.Dataset.from_tensor_slices((x, y)).batch(bs).prefetch(tf.data.AUTOTUNE)


def wanted(mid):
    return mid in RUN and not R.done(mid)


def best_ckpt(mid):
    """Notebook saves the best model by val_accuracy; weights-only avoids SavedModel issues with hub layers."""
    path = str(CKPT / f"best_model_{mid}.weights.h5")
    return path, tf.keras.callbacks.ModelCheckpoint(path, monitor="val_accuracy", save_best_only=True,
                                                    save_weights_only=True, verbose=1)


def fit_short(model, train, val, epochs, callbacks=None):
    """Notebook trains models 1-5 on 10% of the batches per epoch and validates on 10%."""
    return model.fit(train, steps_per_epoch=int(0.1 * len(train)), epochs=epochs, validation_data=val,
                     validation_steps=int(0.1 * len(val)), callbacks=callbacks, verbose=2)


# ---------------- shared vectorizers / embeddings ----------------
sent_lens = [len(s.split()) for s in train_sentences]
output_seq_len = int(np.percentile(sent_lens, 95))
text_vectorizer = TextVectorization(max_tokens=68000, output_sequence_length=output_seq_len)
text_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_sentences).batch(4096))
token_vocab = text_vectorizer.get_vocabulary()


def split_chars(text):
    return " ".join(list(text))


train_chars, val_chars, test_chars = ([split_chars(s) for s in ss] for ss in (train_sentences, val_sentences, test_sentences))
output_seq_char_len = int(np.percentile([len(s) for s in train_sentences], 95))
alphabet = string.ascii_lowercase + string.digits + string.punctuation
char_vectorizer = TextVectorization(max_tokens=len(alphabet) + 2, output_sequence_length=output_seq_char_len,
                                    name="char_vectorizer")
char_vectorizer.adapt(tf.data.Dataset.from_tensor_slices(train_chars).batch(4096))
char_vocab = char_vectorizer.get_vocabulary()
print(f"[prep] seq_len={output_seq_len} char_len={output_seq_char_len} vocab={len(token_vocab)} chars={len(char_vocab)}")

_use = None


def use_layer():
    global _use
    if _use is None:
        _use = hub.KerasLayer(USE_URL, trainable=False, name="universal_sentence_encoder")
    return _use


def _inline_encoder(kind, name):
    """String input -> frozen TF-Hub encoder, exactly as in the notebook."""
    inp = layers.Input(shape=[], dtype=tf.string, name=name)
    if kind == "use":
        return inp, use_layer()(inp)
    bert_in = hub.KerasLayer(BERT_PRE_URL, name="bert_input_preprocess")(inp)
    return inp, hub.KerasLayer(BERT_URL, name="bert_layer")(bert_in, training=False)["pooled_output"]  # frozen


SENTS = {"train": train_sentences, "val": val_sentences, "test": test_sentences}
_enc_cache = {}


def encoded(kind, split):
    """Cached output of the frozen encoder for one split (only with --cache-encoders)."""
    path = ROOT / ".cache" / "emb" / f"tf_{args.dataset}_{kind}_{split}.npy"
    if not path.exists():
        if kind not in _enc_cache:
            if kind == "bert":  # WordPiece runs serially on CPU (~90 sent/s); move it into parallel tf.data workers
                txt = layers.Input(shape=[], dtype=tf.string)
                tokenizer = tf.keras.Model(txt, hub.KerasLayer(BERT_PRE_URL)(txt))
                ids = {k: layers.Input(shape=(128,), dtype=tf.int32, name=k)
                       for k in ("input_word_ids", "input_mask", "input_type_ids")}
                enc = tf.keras.Model(ids, hub.KerasLayer(BERT_URL)(ids, training=False)["pooled_output"])
                _enc_cache[kind] = (tokenizer, enc)
            else:
                _enc_cache[kind] = (None, tf.keras.Model(*_inline_encoder(kind, f"{kind}_text")))
        tokenizer, enc = _enc_cache[kind]
        with tf.device("/CPU:0"):
            batches = tf.data.Dataset.from_tensor_slices(SENTS[split]).batch(256)
            if tokenizer is not None:
                batches = batches.map(lambda b: tokenizer(b), num_parallel_calls=tf.data.AUTOTUNE)
            batches = batches.prefetch(4)
        # batch-by-batch to host memory: predict() concatenates on GPU and OOMs at 2.2M x 512
        x = np.concatenate([enc(b, training=False).numpy().astype(np.float16) for b in batches])
        path.parent.mkdir(parents=True, exist_ok=True); np.save(path, x)
        print(f"[encode] {kind} {split}: {x.shape}", flush=True)
    return np.load(path).astype(np.float32)


def text_input(kind, name):
    """(model input, sentence representation): inline encoder, or its cached output as a float input."""
    if args.cache_encoders:
        inp = layers.Input(shape=(512 if kind == "use" else 768,), dtype=tf.float32, name=name)
        return inp, inp
    return _inline_encoder(kind, name)


def text_feed(kind, split):
    return encoded(kind, split) if args.cache_encoders else SENTS[split]


def char_branch(name, mask_zero=True):
    char_inputs = layers.Input(shape=(1,), dtype=tf.string, name=name)
    x = layers.Embedding(len(char_vocab), 25, mask_zero=mask_zero, name=f"{name}_embed")(char_vectorizer(char_inputs))
    return char_inputs, layers.Bidirectional(layers.LSTM(24))(x)


# positional features shared by models 5 and 6
pos = {k: (tf.one_hot(d["line_number"].to_numpy(), 15).numpy(), tf.one_hot(d["total_lines"].to_numpy(), 20).numpy())
       for k, d in (("train", train_df), ("val", val_df), ("test", test_df))}


def tribrid_ds(split, sentences, chars, labels):
    return ds((pos[split][0], pos[split][1], sentences, chars), labels)


def tribrid_head(token_inputs, token_out, name):
    """Model 5/6 head: token+char -> Dense256 -> concat with line_number & total_lines dense features."""
    char_inputs, char_out = char_branch(f"{name}_char_input", mask_zero=(name == "m5"))
    ln_in = layers.Input(shape=(15,), dtype=tf.float32, name=f"{name}_line_number_input")
    tl_in = layers.Input(shape=(20,), dtype=tf.float32, name=f"{name}_total_lines_input")
    z = layers.Dense(256, activation="relu")(layers.Concatenate()([token_out, char_out]))
    z = layers.Dropout(0.5)(z)
    x = layers.Concatenate()([layers.Dense(32, activation="relu")(ln_in), layers.Dense(32, activation="relu")(tl_in), z])
    out = layers.Dense(num_classes, activation="softmax", name="output_layer")(x)
    model = tf.keras.Model([ln_in, tl_in, token_inputs, char_inputs], out)
    model.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.2),
                  optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
    return model


# ---------------- Model 0: TF-IDF + Multinomial NB baseline ----------------
if wanted(0):
    t = time.time()
    m0 = Pipeline([("tf-idf", TfidfVectorizer()), ("clf", MultinomialNB())]).fit(train_sentences, y_train)
    save_result(0, "baseline", m0.predict(val_sentences), m0.predict(test_sentences), time.time() - t)

train_ds, val_ds, test_ds = (ds(s, l) for s, l in ((train_sentences, train_labels_one_hot),
                                                    (val_sentences, val_labels_one_hot),
                                                    (test_sentences, test_labels_one_hot)))

# ---------------- Model 1: Conv1D + custom token embeddings ----------------
if wanted(1):
    inputs = layers.Input(shape=(1,), dtype=tf.string)
    x = layers.Embedding(len(token_vocab), 128, mask_zero=True, name="token_embedding")(text_vectorizer(inputs))
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling1D()(x)
    m1 = tf.keras.Model(inputs, layers.Dense(num_classes, activation="softmax")(x))
    m1.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
    t = time.time(); fit_short(m1, train_ds, val_ds, 3)
    save_result(1, "custom_token_embed_conv1d", predict(m1, val_ds), predict(m1, test_ds), time.time() - t)

# ---------------- Model 2: USE feature extraction ----------------
if wanted(2):
    inputs, emb = text_input("use", "m2_text")
    x = layers.Dense(128, activation="relu")(emb)
    m2 = tf.keras.Model(inputs, layers.Dense(num_classes, activation="softmax")(x))
    m2.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
    tr, va, te = (ds(text_feed("use", k), l) for k, l in (("train", train_labels_one_hot), ("val", val_labels_one_hot),
                                                           ("test", test_labels_one_hot)))
    t = time.time(); fit_short(m2, tr, va, 3)
    save_result(2, "pretrained_token_embed", predict(m2, va), predict(m2, te), time.time() - t)

# ---------------- Model 3: Conv1D + character embeddings ----------------
if wanted(3):
    inputs = layers.Input(shape=(1,), dtype="string")
    x = layers.Embedding(len(char_vocab), 25, mask_zero=True, name="char_embed")(char_vectorizer(inputs))
    x = layers.Conv1D(64, kernel_size=5, padding="same", activation="relu")(x)
    x = layers.GlobalMaxPool1D()(x)
    m3 = tf.keras.Model(inputs, layers.Dense(num_classes, activation="softmax")(x), name="model_3_conv1D_char_embedding")
    m3.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
    tr, va, te = (ds(c, l) for c, l in ((train_chars, train_labels_one_hot), (val_chars, val_labels_one_hot),
                                        (test_chars, test_labels_one_hot)))
    t = time.time(); fit_short(m3, tr, va, 3)
    save_result(3, "custom_char_embed_conv1d", predict(m3, va), predict(m3, te), time.time() - t)

# ---------------- Model 4: hybrid USE token + char BiLSTM ----------------
if wanted(4):
    token_inputs, emb = text_input("use", "token_input")
    token_out = layers.Dense(128, activation="relu")(emb)
    char_inputs, char_out = char_branch("char_input")
    x = layers.Dropout(0.5)(layers.Concatenate(name="token_char_hybrid")([token_out, char_out]))
    x = layers.Dropout(0.5)(layers.Dense(128, activation="relu")(x))
    m4 = tf.keras.Model([token_inputs, char_inputs], layers.Dense(num_classes, activation="softmax")(x),
                        name="model_4_token_char_embedding")
    m4.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"])
    tr, va, te = (ds((text_feed("use", k), c), l) for k, c, l in (("train", train_chars, train_labels_one_hot),
                                                                  ("val", val_chars, val_labels_one_hot),
                                                                  ("test", test_chars, test_labels_one_hot)))
    t = time.time(); fit_short(m4, tr, va, 3)
    save_result(4, "hybrid_char_token_embed", predict(m4, va), predict(m4, te), time.time() - t)

# ---------------- Model 5: tribrid USE token + char + positional ----------------
if wanted(5):
    token_inputs, emb = text_input("use", "token_inputs")
    token_out = layers.Dense(128, activation="relu")(emb)
    m5 = tribrid_head(token_inputs, token_out, "m5")
    tr = tribrid_ds("train", text_feed("use", "train"), train_chars, train_labels_one_hot)
    va = tribrid_ds("val", text_feed("use", "val"), val_chars, val_labels_one_hot)
    te = tribrid_ds("test", text_feed("use", "test"), test_chars, test_labels_one_hot)
    path, cb = best_ckpt(5)
    t = time.time(); fit_short(m5, tr, va, 2, [cb]); m5.load_weights(path)
    save_result(5, "tribrid_pos_char_token_embed", predict(m5, va), predict(m5, te), time.time() - t)

# ---------------- Model 6: PubMed BERT + char + positional (the paper's proposed model) ----------------
if wanted(6):
    token_inputs, pooled = text_input("bert", "token_inputs")  # BERT frozen with training=False, as in notebook
    token_out = layers.Dense(128, activation="relu")(pooled)
    m6 = tribrid_head(token_inputs, token_out, "m6")
    tr = tribrid_ds("train", text_feed("bert", "train"), train_chars, train_labels_one_hot)
    va = tribrid_ds("val", text_feed("bert", "val"), val_chars, val_labels_one_hot)
    te = tribrid_ds("test", text_feed("bert", "test"), test_chars, test_labels_one_hot)
    path, cb = best_ckpt(6)
    t = time.time()
    m6.fit(tr, epochs=5, steps_per_epoch=len(tr), validation_data=va,
           validation_steps=int(0.15 * len(va)), callbacks=[cb], verbose=2)
    m6.load_weights(path)
    save_result(6, "Transformer-BERT model", predict(m6, va), predict(m6, te), time.time() - t)

# ---------------- Model 7: USE + LSTM additive self-attention + line position ----------------
if wanted(7):
    tr_lines, va_lines, te_lines = line_number_one_hot(train_df, val_df, test_df)
    tr_lines = np.abs(tr_lines - 0.01)  # notebook's apply_smoothing (train only)
    n_lines = tr_lines.shape[1]  # 31 for 20k (hard-coded in notebook); computed so 200k works too

    def self_attention_block(x, units=64, act="gelu"):
        init = tf.keras.initializers.GlorotNormal()
        xe = layers.Lambda(lambda e: tf.expand_dims(e, axis=1), name="block_dim_expand")(x)
        q = layers.LSTM(units, activation=act, kernel_initializer=init, return_sequences=True, name="block_query_lstm")(xe)
        v = layers.LSTM(units, activation=act, kernel_initializer=init, go_backwards=True, return_sequences=True,
                        name="block_value_lstm")(xe)
        att = layers.AdditiveAttention(name="block_attention")([q, v])
        pool = layers.GlobalAveragePooling1D(name="block_average_pooler")
        qn = layers.BatchNormalization(name="block_query_batch_norm")(pool(q))
        an = layers.BatchNormalization(name="block_attention_batch_norm")(pool(att))
        return layers.Add(name="block_residual")([qn, an])

    init = tf.keras.initializers.GlorotNormal()
    abstract_input, emb = text_input("use", "abstract_text_input")
    current_line = layers.Input(shape=(n_lines,), dtype=tf.float32, name="abstract_current_line")
    att = layers.Multiply(name="mul_residual")([self_attention_block(emb),
                                                layers.Dense(64, activation="gelu", kernel_initializer=init)(emb)])
    line = layers.Dropout(0.2)(layers.Dense(32, activation="gelu", kernel_initializer=init)(current_line))
    x = layers.Dense(64, activation="gelu", kernel_initializer=init)(layers.Concatenate()([att, line]))
    m7 = tf.keras.Model([abstract_input, current_line],
                        layers.Dense(num_classes, activation="softmax", kernel_initializer=init)(x),
                        name="use_attention_model")
    m7.compile(loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
               optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=["accuracy"])
    with tf.device("/CPU:0"):
        tr = tf.data.Dataset.from_tensor_slices(((text_feed("use", "train"), tr_lines), train_labels_one_hot.astype(np.float32))) \
            .shuffle(1000, seed=args.seed).batch(32).prefetch(tf.data.AUTOTUNE)
    va = ds((text_feed("use", "val"), va_lines), val_labels_one_hot.astype(np.float32))
    te = ds((text_feed("use", "test"), te_lines), test_labels_one_hot.astype(np.float32))
    path, cb = best_ckpt(7)
    cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=2, factor=0.1, verbose=1), cb]
    t = time.time(); m7.fit(tr, epochs=5, validation_data=va, callbacks=cbs, verbose=2); m7.load_weights(path)
    save_result(7, "Self Attention model", predict(m7, va), predict(m7, te), time.time() - t)

print(f"[done] total {(time.time()-t0)/60:.1f} min. Run: python make_report.py --framework tf --dataset {args.dataset}")
