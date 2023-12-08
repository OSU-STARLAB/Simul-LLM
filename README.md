# Simul-LLM

Simul-LLM is the first open-source fine-tuning to evaluation pipeline development framework for applying LLMs to text-to-text simultaneous machine translation (SimulMT). It currently supports Falcon-based models and will imminently support both LLaMa and Mistral-based models. 

To work with framework and replicate the environment it was initially constructed in, clone recursively and do the following. 

```
pip install -r requirements.txt
cd SimulEval
pip install -e .
```

Some example fine-tuning datasets are provided on the HuggingFace Hub under 'maxolotl/.'

Example scripts are provided via 'scripts/' with default arguments being set-up for a fully sharded Falcon 7B model on relatively low-end hardware. To extend this framework to other LLMs, it's recommended that the setup within 'llmsimul/falcon/' be copied. 

---

### Extending to Other LLMs

The structure of this repository is meant to be friendly to attempts to extend its functionality to new LLMs. The following general steps can be executed to enable fine-tuning and evaluation via SimulEval:

1. Add some parsing infrastructure to 'cli/finetune.py'.

2. Create a SFTTrainer wrapper for a new LLM by inheriting from the general wrapper found in 'llmsimul/trainer_wrapper.py' and implementing the remaining functions.

3. Create an evaluation agent for SimulEval, copying typical text-to-text agent structure.

When in doubt, those attempting to extend this repo's functionality to new LLMs are encouraged to refer to 'llmsimul/falcon/' as a reference point.
