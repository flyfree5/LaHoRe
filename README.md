# Large Language Model Can Be a Foundation for Hidden Rationale-Based Retrieval

Official repository for the paper [Large Language Model Can Be a Foundation for Hidden Rationale-Based Retrieval](https://arxiv.org/abs/2412.16615).

**Accepted by ECIR 2025** [pp 284-293](https://link.springer.com/chapter/10.1007/978-3-031-88714-7_27)

## Training
To train a LaHoRe model, you can use the scripts in `examples/train_full/*`:

```bash
bash examples/train_full/train_qwen_7B_sft.sh
```

## Testing
To test a LaHoRe model, you can use the script:

```bash
bash examples/inference/eval_my_model.sh
```

## Some results
ESconv dataset

| Model        | Recall@1 | Recall@3 | p-MRR    |
|--------------|----------|----------|----------|
| RepLLaMA     | 17.4     | 49.1     | 39.7     |
| Promptriever | 17.6     | 48.9     | 39.5     |
| E5-Mistral   | 15.5     | 50.2     | 39.3     |
| LlaMA2Vec    | 23.9     | 55.6     | 45.5     |
| GritLM       | 19.2     | 50.2     | 41.7     |
| LaHoRe-SFT   | 34.4     | 67.1     | 54.8     |
| LaHoRe-DPO   | **38.0** | **69.5** | **57.6** |

