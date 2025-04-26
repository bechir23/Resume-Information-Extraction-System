# nlp_utils/main.py
# This script runs the NER data pipeline: load, augment, analyze, and export.
import logging
from pathlib import Path
from nlp_utils.data_loader import load_and_export_ner_data
from nlp_utils.augmentation import convert_and_save_augmented_spacy_files
from nlp_utils.visualization import visualize_entity_analysis
from nlp_utils.entity_analysis import analyze_all_entities
from .augmentation import create_balanced_augmentation

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    input_json = "resume_system_data/ner_data/Entity Recognition in Resumes.json"  # Update path as needed
    train_path = "models/resume_model/train.spacy"
    dev_path = "models/resume_model/dev.spacy"

    # Load and export data
    train_data, val_data = load_and_export_ner_data(
        input_json_path=input_json,
        train_output_path=train_path,
        dev_output_path=dev_path
    )
    original_data= {
        "train_data": train_data,
        "val_data": val_data
    }
    # Analyze original data
    analysis_results = analyze_all_entities(train_data)

    # Visualize results for a specific entity type
    entity_types = [
        "Email Address", "Name", "Degree", "Graduation Year", 
        "Years of Experience", "Skills", "Designation", 
        "Companies worked at", "College Name", "Location"
    ]

    for ent in entity_types:
        visualize_entity_analysis(analysis_results, ent)

    # Augment data (optional)
    augmented_data = create_balanced_augmentation(original_data)
        
    # Print final statistics
    print("\n=== FINAL AUGMENTATION STATISTICS ===")
    print(f"Original training examples: {len(original_data['train_data'])}")
    print(f"Augmented training examples: {augmented_data['train_count']}")

    # Save augmented data (optional)
    convert_and_save_augmented_spacy_files(
    train_data=augmented_data['train_data'],
    val_data=augmented_data['val_data'],
    train_output_path="augmented_train.spacy",
    dev_output_path="augmented_dev.spacy",
    debug_mode=True,
    run_validation=True
)
