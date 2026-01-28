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
├── agent_class_Context_PureC_EN.py      # Core agent and DQN variants
├── transformer_module_Context_EN.py     # Transformer-based context encoder
├── run_Context_PureC_EN.py              # Training entry point
├── inference_EN.py                      # Inference / evaluation script
├── requirements.txt                     # Python dependencies
└── README.md
