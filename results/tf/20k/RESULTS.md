# Results: tf, PubMed 20k RCT

## val split (weighted precision/recall/F1)

|   model | name                         |   accuracy |   precision |   recall |     f1 |   train_min |
|--------:|:-----------------------------|-----------:|------------:|---------:|-------:|------------:|
|       0 | baseline                     |     0.7218 |      0.7186 |   0.7218 | 0.6989 |      0.1233 |
|       1 | custom_token_embed_conv1d    |     0.7877 |      0.7848 |   0.7877 | 0.7855 |      0.6417 |
|       2 | pretrained_token_embed       |     0.7145 |      0.7149 |   0.7145 | 0.7116 |      0.9517 |
|       3 | custom_char_embed_conv1d     |     0.6568 |      0.6490 |   0.6568 | 0.6476 |      0.2433 |
|       4 | hybrid_char_token_embed      |     0.7338 |      0.7328 |   0.7338 | 0.7314 |      1.8033 |
|       5 | tribrid_pos_char_token_embed |     0.8196 |      0.8194 |   0.8196 | 0.8184 |      1.5283 |
|       6 | Transformer-BERT model       |     0.8871 |      0.8901 |   0.8871 | 0.8838 |     89.9150 |
|       7 | Self Attention model         |     0.8472 |      0.8511 |   0.8472 | 0.8433 |     16.5400 |

## test split (weighted precision/recall/F1)

|   model | name                         |   accuracy |   precision |   recall |     f1 |   train_min |
|--------:|:-----------------------------|-----------:|------------:|---------:|-------:|------------:|
|       0 | baseline                     |     0.7167 |      0.7124 |   0.7167 | 0.6925 |      0.1233 |
|       1 | custom_token_embed_conv1d    |     0.7824 |      0.7801 |   0.7824 | 0.7805 |      0.6417 |
|       2 | pretrained_token_embed       |     0.7082 |      0.7104 |   0.7082 | 0.7057 |      0.9517 |
|       3 | custom_char_embed_conv1d     |     0.6512 |      0.6454 |   0.6512 | 0.6416 |      0.2433 |
|       4 | hybrid_char_token_embed      |     0.7267 |      0.7260 |   0.7267 | 0.7241 |      1.8033 |
|       5 | tribrid_pos_char_token_embed |     0.8143 |      0.8142 |   0.8143 | 0.8130 |      1.5283 |
|       6 | Transformer-BERT model       |     0.8841 |      0.8864 |   0.8841 | 0.8809 |     89.9150 |
|       7 | Self Attention model         |     0.8446 |      0.8480 |   0.8446 | 0.8406 |     16.5400 |

## Table 1 (Model 6 per-class F1) vs paper

|             |   paper F1 |   ours val F1 |   ours test F1 |
|:------------|-----------:|--------------:|---------------:|
| BACKGROUND  |       0.77 |          0.79 |           0.80 |
| CONCLUSIONS |       0.89 |          0.91 |           0.91 |
| METHODS     |       0.91 |          0.93 |           0.92 |
| OBJECTIVE   |       0.63 |          0.62 |           0.63 |
| RESULTS     |       0.90 |          0.92 |           0.91 |
| ACCURACY    |       0.87 |          0.89 |           0.88 |
