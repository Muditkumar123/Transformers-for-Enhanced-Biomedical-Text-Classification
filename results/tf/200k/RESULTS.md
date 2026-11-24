# Results: tf, PubMed 200k RCT

## val split (weighted precision/recall/F1)

|   model | name                         |   accuracy |   precision |   recall |     f1 |   train_min |
|--------:|:-----------------------------|-----------:|------------:|---------:|-------:|------------:|
|       0 | baseline                     |     0.7498 |      0.7432 |   0.7498 | 0.7390 |      0.5733 |
|       1 | custom_token_embed_conv1d    |     0.8365 |      0.8393 |   0.8365 | 0.8347 |      2.2000 |
|       2 | pretrained_token_embed       |     0.7765 |      0.7744 |   0.7765 | 0.7740 |      0.7183 |
|       3 | custom_char_embed_conv1d     |     0.7512 |      0.7424 |   0.7512 | 0.7443 |      1.2200 |
|       4 | hybrid_char_token_embed      |     0.7825 |      0.7789 |   0.7825 | 0.7788 |      9.3267 |
|       5 | tribrid_pos_char_token_embed |     0.8613 |      0.8653 |   0.8613 | 0.8590 |      6.9633 |
|       6 | Transformer-BERT model       |     0.8986 |      0.8992 |   0.8986 | 0.8979 |    120.5117 |
|       7 | Self Attention model         |     0.8600 |      0.8634 |   0.8600 | 0.8584 |     49.6083 |

## test split (weighted precision/recall/F1)

|   model | name                         |   accuracy |   precision |   recall |     f1 |   train_min |
|--------:|:-----------------------------|-----------:|------------:|---------:|-------:|------------:|
|       0 | baseline                     |     0.7517 |      0.7457 |   0.7517 | 0.7415 |      0.5733 |
|       1 | custom_token_embed_conv1d    |     0.8349 |      0.8367 |   0.8349 | 0.8329 |      2.2000 |
|       2 | pretrained_token_embed       |     0.7777 |      0.7750 |   0.7777 | 0.7752 |      0.7183 |
|       3 | custom_char_embed_conv1d     |     0.7480 |      0.7392 |   0.7480 | 0.7410 |      1.2200 |
|       4 | hybrid_char_token_embed      |     0.7839 |      0.7796 |   0.7839 | 0.7799 |      9.3267 |
|       5 | tribrid_pos_char_token_embed |     0.8616 |      0.8640 |   0.8616 | 0.8595 |      6.9633 |
|       6 | Transformer-BERT model       |     0.8959 |      0.8958 |   0.8959 | 0.8951 |    120.5117 |
|       7 | Self Attention model         |     0.8583 |      0.8601 |   0.8583 | 0.8567 |     49.6083 |

## Table 1 (Model 6 per-class F1) vs paper

|             |   paper F1 |   ours val F1 |   ours test F1 |
|:------------|-----------:|--------------:|---------------:|
| BACKGROUND  |       0.77 |          0.76 |           0.76 |
| CONCLUSIONS |       0.89 |          0.92 |           0.91 |
| METHODS     |       0.91 |          0.94 |           0.94 |
| OBJECTIVE   |       0.63 |          0.71 |           0.71 |
| RESULTS     |       0.90 |          0.93 |           0.93 |
| ACCURACY    |       0.87 |          0.90 |           0.90 |
