#!/bin/bash
# Fetch PubMed RCT data and the pretrained encoders into data/ and models/ (both git-ignored).
set -e
cd "$(dirname "$0")"
R=https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/master
for d in PubMed_20k_RCT_numbers_replaced_with_at_sign PubMed_200k_RCT_numbers_replaced_with_at_sign; do
  mkdir -p data/pubmed-rct/$d
  for f in dev.txt test.txt; do curl -fsSL -o data/pubmed-rct/$d/$f $R/$d/$f; done
done
curl -fsSL -o data/pubmed-rct/PubMed_20k_RCT_numbers_replaced_with_at_sign/train.txt $R/PubMed_20k_RCT_numbers_replaced_with_at_sign/train.txt
D=data/pubmed-rct/PubMed_200k_RCT_numbers_replaced_with_at_sign
curl -fsSL -o $D/train.zip $R/$D/train.zip && unzip -o -q $D/train.zip -d $D && rm $D/train.zip

# TF-Hub models (optional: medi_parse_tf.py downloads them itself if models/tfhub is missing)
for pair in "use4 google/universal-sentence-encoder/4" "bert_pubmed2 google/experts/bert/pubmed/2" \
            "bert_preprocess3 tensorflow/bert_en_uncased_preprocess/3"; do
  set -- $pair; mkdir -p models/tfhub/$1
  curl -fsSL "https://tfhub.dev/$2?tf-hub-format=compressed" | tar xz -C models/tfhub/$1
done

# Hugging Face models for the PyTorch port
HF=https://huggingface.co
DU=sentence-transformers/distiluse-base-multilingual-cased-v2
for f in config.json tokenizer.json tokenizer_config.json special_tokens_map.json vocab.txt model.safetensors \
         2_Dense/config.json 2_Dense/model.safetensors; do
  mkdir -p models/hf/distiluse/$(dirname $f); curl -fsSL -o models/hf/distiluse/$f $HF/$DU/resolve/main/$f
done
BB=microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract
mkdir -p models/hf/biomedbert
for f in config.json tokenizer_config.json vocab.txt pytorch_model.bin; do
  curl -fsSL -o models/hf/biomedbert/$f $HF/$BB/resolve/main/$f
done
# BiomedBERT ships only pytorch_model.bin; transformers>=5 needs torch>=2.6 to load it, so convert once to safetensors
python - <<'PY'
import torch
from safetensors.torch import save_file
sd = torch.load("models/hf/biomedbert/pytorch_model.bin", map_location="cpu", weights_only=True)
save_file({k.removeprefix("bert."): v.contiguous() for k, v in sd.items() if k.startswith("bert.")},
          "models/hf/biomedbert/model.safetensors", metadata={"format": "pt"})
PY
rm models/hf/biomedbert/pytorch_model.bin
echo done
