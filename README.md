# Gemma-2B LoRA Fine-Tuning for Question Answering

## Overview
This project explores parameter-efficient fine-tuning of Google's Gemma-2B language model using LoRA (Low-Rank Adaptation) and the SQuAD question-answering dataset.

## Tech Stack
- Python
- Google Colab
- Hugging Face Transformers
- PEFT (LoRA)
- PyTorch
- SQuAD Dataset

## Methodology
1. Load the pretrained Gemma-2B model.
2. Prepare question-answering examples from SQuAD.
3. Apply LoRA adapters to selected attention projection layers.
4. Fine-tune on 500 training examples for one epoch.
5. Evaluate generated answers on 100 validation examples.

## Initial Evaluation Results
- Exact Match (EM): 91.00%
- Token-level F1: 93.33%
- Training Loss: 1.1531

Note: These are initial results from the current evaluation setup. A controlled baseline comparison and further validation are needed before claiming a performance improvement.

## Project Files
- `gemma-2b-lora-finetuning.ipynb`: Google Colab notebook containing the training workflow.
- `gemma2b-lora-adapter.zip`: Saved LoRA adapter files, if included.

## How to Run
1. Open the notebook in Google Colab.
2. Enable a GPU runtime.
3. Request access to the Gemma model on Hugging Face if required.
4. Configure your Hugging Face token using Colab Secrets.
5. Run the notebook cells in order.

## Future Improvements
- Compare the fine-tuned model with the original Gemma-2B baseline.
- Evaluate on a larger held-out dataset.
- Test more training examples and epochs.
- Analyze incorrect answers and improve answer generation.

## Author
Kalash Dwivedi
