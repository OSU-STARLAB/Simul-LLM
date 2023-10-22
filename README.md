# llm-simultaneous-translation

To work with this and replicate the environment it was initially constructed in, clone recursively and do the following. 

```
pip install -r requirements.txt
cd SimulEval
pip install -e .
```

Currently, it is assumed that the training set is available on HuggingFace via 'maxolotl/.' Test and validation splits must be generated separately, at the moment. 

Example scripts are provided via scripts/ with default arguments being set-up for a fully sharded Falcon 7B model. To extend this framework to other LLMs, it's recommended that the setup within llmsimul/falcon/ be copied. 

---

### Extending to Other LLMs

The structure of this repository is meant to be friendly to attempts to extend its functionality to new LLMs. The following general steps can be executed to enable fine-tuning and evaluation via SimulEval:

1. Add some parsing infrastructure to cli/finetune.py.

2. Create a SFTTrainer wrapper for a new LLM by inheriting from the general wrapper found in llmsimul/trainer_wrapper.py and implementing the remaining functions.

3. Create an evaluation agent for SimulEval, copying typical text-to-text agent structure.

When in doubt, those attempting to extend this repo's functionality to new LLMs are encouraged to refer to llmsimul/falcon/ as a reference point.

---

### TODO: 

1. Finish validating DDP capabilities for non-sharded LLMs via Accelerate.

2. Add easily replicable preprocessing pipeline.

3. Explore customized attention masks per sample for better simultaneous training flow.

4. Extend to LLaMa, Mistral, etc.
