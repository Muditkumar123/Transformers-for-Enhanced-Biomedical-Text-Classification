# Results: torch, PubMed 200k RCT

## val split (weighted precision/recall/F1)

|   model | name                         |   accuracy |   precision |   recall |     f1 |   train_min |
|--------:|:-----------------------------|-----------:|------------:|---------:|-------:|------------:|
|       0 | baseline                     |     0.7498 |      0.7432 |   0.7498 | 0.7390 |      0.5883 |
|       1 | custom_token_embed_conv1d    |     0.8367 |      0.8393 |   0.8367 | 0.8348 |      1.5283 |
|       2 | pretrained_token_embed       |     0.8237 |      0.8244 |   0.8237 | 0.8227 |      4.4567 |
|       3 | custom_char_embed_conv1d     |     0.7481 |      0.7383 |   0.7481 | 0.7408 |      3.7067 |
|       4 | hybrid_char_token_embed      |     0.8206 |      0.8199 |   0.8206 | 0.8186 |     11.3383 |
|       5 | tribrid_pos_char_token_embed |     0.8784 |      0.8833 |   0.8784 | 0.8763 |      8.3600 |
|       6 | Transformer-BERT model       |     0.9000 |      0.9008 |   0.9000 | 0.8990 |     53.9783 |
|       7 | Self Attention model         |     0.8857 |      0.8874 |   0.8857 | 0.8847 |     25.7500 |

## test split (weighted precision/recall/F1)

|   model | name                         |   accuracy |   precision |   recall |     f1 |   train_min |
|--------:|:-----------------------------|-----------:|------------:|---------:|-------:|------------:|
|       0 | baseline                     |     0.7517 |      0.7457 |   0.7517 | 0.7415 |      0.5883 |
|       1 | custom_token_embed_conv1d    |     0.8355 |      0.8369 |   0.8355 | 0.8333 |      1.5283 |
|       2 | pretrained_token_embed       |     0.8230 |      0.8228 |   0.8230 | 0.8218 |      4.4567 |
|       3 | custom_char_embed_conv1d     |     0.7445 |      0.7346 |   0.7445 | 0.7371 |      3.7067 |
|       4 | hybrid_char_token_embed      |     0.8204 |      0.8184 |   0.8204 | 0.8181 |     11.3383 |
|       5 | tribrid_pos_char_token_embed |     0.8765 |      0.8795 |   0.8765 | 0.8744 |      8.3600 |
|       6 | Transformer-BERT model       |     0.8978 |      0.8979 |   0.8978 | 0.8966 |     53.9783 |
|       7 | Self Attention model         |     0.8851 |      0.8859 |   0.8851 | 0.8839 |     25.7500 |

## Table 1 (Model 6 per-class F1) vs paper

|             |   paper F1 |   ours val F1 |   ours test F1 |
|:------------|-----------:|--------------:|---------------:|
| BACKGROUND  |       0.77 |          0.76 |           0.76 |
| CONCLUSIONS |       0.89 |          0.91 |           0.91 |
| METHODS     |       0.91 |          0.95 |           0.94 |
| OBJECTIVE   |       0.63 |          0.71 |           0.70 |
| RESULTS     |       0.90 |          0.93 |           0.93 |
| ACCURACY    |       0.87 |          0.90 |           0.90 |
