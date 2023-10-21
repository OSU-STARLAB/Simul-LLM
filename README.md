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

TODO: 

1. Finish validating DDP capabilities for non-sharded LLMs via Accelerate.

2. Finish data collation setup to speed up fine-tuning. Includes custom formatting functions for each LLM for increased extendability.

3. Add easily replicable preprocessing pipeline.

4. Explore customized attention masks per sample for better simultaneous training flow.

5. Extend to LLaMa, Mistral, etc.
