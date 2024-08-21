# Simul-LLM

Simul-LLM is the first open-source fine-tuning to evaluation pipeline development framework for applying LLMs to text-to-text simultaneous machine translation (SimulMT). It currently supports Falcon and Mistral-based models with some support for Llama-based models. You can read our introductory preprint [here](https://arxiv.org/abs/2312.04691).

To work with framework and replicate the environment it was initially constructed in, clone recursively and do the following. 

```
pip install -r requirements.txt
cd SimulEval
pip install -e .
```

Some example fine-tuning datasets are provided on the HuggingFace Hub under `maxolotl/.`

Example scripts are provided via `scripts/` with default arguments being set-up for a fully sharded Falcon 7B model on relatively low-end hardware. To extend this framework to other LLMs, it's recommended that the setup within `llmsimul/falcon/` be copied. 

A few PEFT LoRA checkpoints are provided here for Falcon-based models. 

| en-es | en-de |
| ----- | ----- |
| [NMT-chkpt](https://huggingface.co/agostinvic/nmt-en-es) | [NMT-chkpt](https://huggingface.co/agostinvic/nmt-en-de) |
| [Exp-Wait7-chkpt](https://huggingface.co/agostinvic/simulmt-2M-en-de) | [Exp-Wait7-chkpt](https://huggingface.co/agostinvic/simulmt-2M-en-de) | 

---

### To-Do/Wish List

1. Add support for more custom tailoring of "computationally aware" MT latency. 
2. ~~Add support for more evaluation metrics (e.g. COMET).~~
3. ~~Add support for easy to swap out translation scheduling. While not difficult in the project's current form, the process of swapping schedules could be rendered essentially seamless with a bit of additional work.~~
4. Finish validating Llama examples.
5. Finish adding some extra ``accelerate`` support to ensure access to more interesting DP paradigms.
6. Add multi-modal options/support for SimulST possibilities.

---

### Extending to Other LLMs

The structure of this repository is meant to be friendly to attempts to extend its functionality to new LLMs. The following general steps can be executed to enable fine-tuning and evaluation via SimulEval:

1. Add some parsing infrastructure to `cli/finetune.py`.

2. Create a SFTTrainer wrapper for a new LLM by inheriting from the general wrapper found in `llmsimul/trainer_wrapper.py` and implementing the remaining functions.

3. Create an evaluation agent for SimulEval, copying typical text-to-text agent structure.

When in doubt, those attempting to extend this repo's functionality to new LLMs are encouraged to refer to `llmsimul/falcon/` as a reference point.

---

### Citation

When employing or extending this framework, please consider citing us as:

```
@inproceedings{agostinelli-etal-2024-simul,
    title = "Simul-{LLM}: A Framework for Exploring High-Quality Simultaneous Translation with Large Language Models",
    author = "Agostinelli, Victor  and Wild, Max  and Raffel, Matthew  and Fuad, Kazi  and Chen, Lizhong",
    editor = "Ku, Lun-Wei  and Martins, Andre  and Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.567",
    pages = "10530--10541",
}
```
