# Simul-LLM

Simul-LLM is the first open-source fine-tuning to evaluation pipeline development framework for applying LLMs to simultaneous machine translation (SimulMT). It currently supports Falcon and Mistral-based models with some support for Llama-based models. You can read our introductory preprint [here](https://aclanthology.org/2024.acl-long.567/).

To work with framework and replicate the environment it was initially constructed in, clone recursively and do the following. 

```
pip install -r requirements.txt
cd SimulEval
pip install -e .
```

Some example fine-tuning datasets are provided on the HuggingFace Hub under `maxolotl/.`

Example scripts are provided via `scripts/` with default arguments being set-up for a fully sharded Falcon 7B model on relatively low-end hardware. To extend this framework to other LLMs, it's recommended that the setup within [`llmsimul/falcon/`](llmsimul/falcon) be copied. 

A few PEFT LoRA checkpoints are provided here for Falcon-based models. 

| en-es | en-de |
| ----- | ----- |
| [NMT-chkpt](https://huggingface.co/agostinvic/nmt-en-es) | [NMT-chkpt](https://huggingface.co/agostinvic/nmt-en-de) |
| [Exp-Wait7-chkpt](https://huggingface.co/agostinvic/simulmt-2M-en-de) | [Exp-Wait7-chkpt](https://huggingface.co/agostinvic/simulmt-2M-en-de) | 

---

### What's New?

- September 2024: SimulMask is accepted to EMNLP '24, hosted in Miami, Florida!
- September 2024: Example simultaneous speech-to-text translation (SimulST) pipeline available [here](examples/basic_speech_to_text).
- August 2024: Official implementation of SimulMask available [here](examples/simulmask)! Preprint for SimulMask available [here](https://arxiv.org/abs/2405.10443).
- May 2024: Simul-LLM is accepted to ACL '24, hosted in Bangkok, Thailand!

---

### Extending to Other LLMs

The structure of this repository is meant to be friendly to attempts to extend its functionality to new LLMs. The following general steps can be executed to enable fine-tuning and evaluation via SimulEval:

1. Add some parsing infrastructure to [`cli/finetune.py`](cli/finetune.py).

2. Create a SFTTrainer wrapper for a new LLM by inheriting from the general wrapper found in [`llmsimul/trainer_wrapper.py`](llmsimul/trainer_wrapper.py) and implementing the remaining functions.

3. Create an evaluation agent for SimulEval, copying typical text-to-text agent structure.

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
    doi = "10.18653/v1/2024.acl-long.567",
    pages = "10530--10541",
}
```
