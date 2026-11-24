# Results: torch, PubMed 20k RCT

## val split (weighted precision/recall/F1)

|   model | name                         |   accuracy |   precision |   recall |     f1 |   train_min |
|--------:|:-----------------------------|-----------:|------------:|---------:|-------:|------------:|
|       0 | baseline                     |     0.7218 |      0.7186 |   0.7218 | 0.6989 |      0.1150 |
|       1 | custom_token_embed_conv1d    |     0.7848 |      0.7809 |   0.7848 | 0.7821 |      0.1550 |
|       2 | pretrained_token_embed       |     0.7617 |      0.7605 |   0.7617 | 0.7585 |      0.0433 |
|       3 | custom_char_embed_conv1d     |     0.6533 |      0.6430 |   0.6533 | 0.6409 |      0.0900 |
|       4 | hybrid_char_token_embed      |     0.7736 |      0.7722 |   0.7736 | 0.7709 |      1.2900 |
|       5 | tribrid_pos_char_token_embed |     0.8454 |      0.8454 |   0.8454 | 0.8442 |      0.9283 |
|       6 | Transformer-BERT model       |     0.8851 |      0.8880 |   0.8851 | 0.8818 |      2.9833 |
|       7 | Self Attention model         |     0.8739 |      0.8734 |   0.8739 | 0.8732 |      2.4033 |

## test split (weighted precision/recall/F1)

|   model | name                         |   accuracy |   precision |   recall |     f1 |   train_min |
|--------:|:-----------------------------|-----------:|------------:|---------:|-------:|------------:|
|       0 | baseline                     |     0.7167 |      0.7124 |   0.7167 | 0.6925 |      0.1150 |
|       1 | custom_token_embed_conv1d    |     0.7808 |      0.7772 |   0.7808 | 0.7782 |      0.1550 |
|       2 | pretrained_token_embed       |     0.7554 |      0.7547 |   0.7554 | 0.7522 |      0.0433 |
|       3 | custom_char_embed_conv1d     |     0.6460 |      0.6380 |   0.6460 | 0.6329 |      0.0900 |
|       4 | hybrid_char_token_embed      |     0.7652 |      0.7649 |   0.7652 | 0.7627 |      1.2900 |
|       5 | tribrid_pos_char_token_embed |     0.8420 |      0.8425 |   0.8420 | 0.8409 |      0.9283 |
|       6 | Transformer-BERT model       |     0.8835 |      0.8850 |   0.8835 | 0.8801 |      2.9833 |
|       7 | Self Attention model         |     0.8703 |      0.8693 |   0.8703 | 0.8693 |      2.4033 |

## Table 1 (Model 6 per-class F1) vs paper

|             |   paper F1 |   ours val F1 |   ours test F1 |
|:------------|-----------:|--------------:|---------------:|
| BACKGROUND  |       0.77 |          0.78 |           0.78 |
| CONCLUSIONS |       0.89 |          0.91 |           0.91 |
| METHODS     |       0.91 |          0.93 |           0.93 |
| OBJECTIVE   |       0.63 |          0.62 |           0.62 |
| RESULTS     |       0.90 |          0.92 |           0.92 |
| ACCURACY    |       0.87 |          0.89 |           0.88 |
