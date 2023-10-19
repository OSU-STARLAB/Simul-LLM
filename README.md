# llm-simultaneous-translation

To work with this and replicate the environment it was initially constructed in, clone recursively and do the following. 

```
pip install -r requirements.txt
cd SimulEval
pip install -e .
```

Currently, it is assumed that the training set is available on HuggingFace via 'maxolotl/.' Test and validation splits must be generated separately, at the moment. 

Example scripts are provided for Falcon-based models via falcon/scripts.
