#!/bin/bash
# run_nlp_pipeline.sh: Automate NER data pipeline and spaCy training
set -e

# Install Python dependencies
pip install -r requirements.txt

# Download spaCy model if needed
python -m spacy download en_core_web_md

# Run the NER data pipeline (data cleaning, augmentation, export)
python -m nlp_utils.main

# Train spaCy models
python -m spacy train models/resume_model/config_tok2vec.cfg --output models/tok2vec --gpu-id 0
python -m spacy train models/resume_model/config_transformer.cfg --output models/transformer --gpu-id 0

echo "Pipeline and training complete."
