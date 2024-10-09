# Simultaneous Masking, Not Prompting Optimization: A Paradigm Shift in Fine-tuning LLMs for Simultaneous Translation

Large language models (LLMs) have achieved state-of-the-art performance in various language processing tasks, motivating their adoption in simultaneous translation. Current finetuning methods to adapt LLMs for simultaneous translation focus on prompting optimization strategies using either data augmentation or prompt structure modifications. However, these methods suffer from several issues, such as unnecessarily expanded training sets, computational inefficiency from dumping the key and value cache, increased prompt sizes, or restriction to a single decision policy. To eliminate these issues, in this work, we propose SimulMask, a new paradigm for fine-tuning LLMs for simultaneous translation. It utilizes a novel attention mask approach that models simultaneous translation during fine-tuning by masking attention for a desired decision policy. Applying the proposed SimulMask on a Falcon LLM for the IWSLT 2017 dataset, we have observed a significant translation quality improvement compared to state-of-the-art prompting optimization strategies on five language pairs while reducing the computational cost.

---

## Fine-tuning

The script used to fine-tune a falcon-1.3B model using SimulMask is provided below.  
```
python cli/finetune.py \
    --model tiiuae/falcon-rw-1b \
    --training-set iwslt2017 --training-subset iwslt2017-en-fr \
    --source-lang en --target-lang fr \
    --lora-alpha 16 --lora-dropout 0.1 --lora-r 64 \
    --use-4bit --bnb-4bit-compute-dtype bfloat16 --bnb-4bit-quant-type nf4 \
    --bsz 16 --update-freq 4 \
    --optim paged_adamw_32bit --lr 2e-4 --lr-scheduler inverse_sqrt --weight-decay 0.1 \
    --warmup-ratio 0.03 --max-grad-norm 1 \
    --save-strategy epoch --eval-interval 10000 --log-interval 1000 --num-train-epochs 2 \
    --max-seq-length 512 --waitk 9 --attmask-type simul \
    --user-dir examples/simulmask \
    --output-dir ${checkpoint_save_path} \
```

The model checkpoints fine-tuned at wait-9 and evaluated at wait-5 are provided below.
| en-fr | en-nl | en-it |
| ----- | ----- | ----- |
| [falcon-chkpt](https://huggingface.co/raffelm/falcon-simulmask-en-fr) | [falcon-chkpt](https://huggingface.co/raffelm/falcon-simulmask-en-nl) | [falcon-chkpt](https://huggingface.co/raffelm/falcon-simulmask-en-it) |

---

## Evaluation
The script used to evaluate a fine-tuned falcon-1.3B model is provided below.

```
python3 cli/simuleval_wrapper.py \
    --agent examples/simulmask/falcon_simulmask_agent.py \
    --source ${source_data_path} \
    --target ${target_data_path} \
    --source-lang en --target-lang fr \
    --model ${checkpoint_save_path} \
    --output ${eval_save_path} \
    --waitk 5 --device cuda --attmask-type causal \
    --compute-dtype bfloat16 --quality-metric BLEU CHRF \
```
---

## Results

### Translation Quality and Latency Results

The LLM Fine-tuned with SimulMask outperforms or matches alternative approaches in terms of translation quality.

| English-French | English-Dutch  | English-Italian  |
|---------|---------|---------|
| ![English-French](figures/en-fr.png) | ![English-Dutch](figures/en-nl.png) | ![English-Italian](figures/en-it.png) |

---
### Computational Saving Results

Fine-tuning an LLM with SimulMask reduces training time compared to alternative approaches.

<img src="figures/training_computation.png" alt="training" width="350"> 

The LLM fine-tuned with SimulMask performs inference at a reduced computational cost compared to alternative approaches.

<img src="figures/inference_computation.png" alt="inference" width="350"> 

---

## Citation

When employing or extending SimulMask, please consider citing us as:

```
@article{raffel2024simultaneous,
  title={Simultaneous Masking, Not Prompting Optimization: A Paradigm Shift in Fine-tuning LLMs for Simultaneous Translation},
  author={Raffel, Matthew and Agostinelli, Victor and Chen, Lizhong},
  journal={arXiv preprint arXiv:2405.10443},
  year={2024}
}
```
