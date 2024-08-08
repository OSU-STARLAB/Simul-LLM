## Simul-LLM Source Code Overview/Layout

Below, a brief guide to the Simul-LLM source code is offered. Only high-level descriptions are provided here, with some additional, detailed information present in the codebase on a file-by-file basis. Adding extra documentation that is a bit more organized is an ongoing effort.

### Fundamental Classes

| Files | Description |
| --- | --- |
| `basic_eval_agent.py` | Structure of basic Simul-LLM evaluation agents with behavior inherited by LLM-specific agents. General behavior should be defined here versus implemented on a model-by-model basis. |
| `trainer_wrapper.py` | Structure of basic Simul-LLM training/fine-tuning wrapper. Interfaces with a `SFTTrainer` object from the [transformers](https://huggingface.co/docs/transformers/en/index) library in addition to a few similarly sourced configs (e.g. PEFT, BitsAndBytes, etc.). |


### Specific Sub-Directories
| Sub-Directory | Description |
| --- | --- |
| `falcon`, `mistral`, `llama` | LLM-specific files and implementations are found in these sub-directories. `falcon` was the basis for most other LLMs in this framework. |
| `schedulers` | Translation schedulers are defined in this sub-directory, with the expectation being that behavior related to them is added to `basic_eval_agent.py`. |
| `simuleval_extensions` | A critical approach taken to constructing this framework was the avoidance of modifying the core functionality of [SimulEval](https://github.com/facebookresearch/SimulEval) at all costs. Given that, this sub-directory contains a number of extensions that are supported innately by SimulEval. |
| `utils_and_misc` | Some utility-focused functionality resides in this sub-directory. Currently hosting the beam rescoring implementation supported out of the box by Simul-LLM. |
| `waitk_transformer` | Some SimulEval-specific functionality required for validating a few classical baselines in our published work, based on [fairseq](https://github.com/facebookresearch/fairseq). |
