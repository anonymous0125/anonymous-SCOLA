# Anonymous Code Repository-SCOLA

This repository provides an **anonymized implementation** for the paper:

> **SCOLA: Human-Inspired Schematic Cognition for Contrastive 
  Reinforcement Learning in Non-Stationary Environments**  
> (Anonymous submission)

---

## Overview

This repository implements a contextual reinforcement learning framework that
learns belief embeddings from interaction trajectories and conditions control
policies on the inferred context. The implementation supports training,
inference, and evaluation in parameterized control environments.

---

## Repository Structure

```text
.
├── agent_class.py      # Core ContextualNet agent
├── FactualEncoder.py     # Contextual encoder
├── train.py              # Training entry point
├── inference.py                      # Inference / evaluation script
├── requirements.txt                     # Python dependencies
└── README.md
```
## Training

To train the agent with contextual belief embeddings, run:

```bash
python train.py
```
The training script initializes the environment, agent, and contextual encoder,
and performs episodic reinforcement learning with belief updates.

By default:
	•	Environment parameters are sampled programmatically at runtime.
	•	Contextual belief embeddings are learned online from interaction trajectories.
	•	Model checkpoints and logs are saved locally according to the script settings.

## Inference / Evaluation

To run inference or evaluate a trained model, execute:
```bash
python inference.py
```
Please ensure that the correct checkpoint path is specified in the script before
running evaluation.

The inference process conditions action selection on the learned contextual
belief representation without access to ground-truth environment parameters.




