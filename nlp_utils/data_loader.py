# nlp_utils/data_loader.py
# This module contains functions for loading, cleaning, and exporting NER data for spaCy training.
import subprocess
import spacy
import random
import logging
from pathlib import Path
from spacy.tokens import DocBin
from tqdm import tqdm
import json

from .visualization import (
    visualize_entity_lengths,
    create_pattern_visualizations,
    create_entity_relationship_visualizations
)

from .entity_analysis import analyze_entity_statistics



from .normalization import clean_entity_by_type, entity_priority, handle_token_overlaps, split_skills

def load_and_export_ner_data(
    train_ratio=0.8,
    train_output_path="./train.spacy", 
    dev_output_path="./dev.spacy",
    debug_mode=False,
    run_validation=False,
    visualize_patterns=False, 
    exclude_entities=["UNKNOWN","Graduation Year","Years of Experience"],
    analyze_before_processing=False,
    split_skill_entities=True,
    max_entity_length=150  # Exclude extremely long entities
):
    """
    Enhanced resume NER data loader with advanced cleaning and analysis.
    """
    
    ner_dataset_path = Path('/kaggle/input/resume-entities-for-ner/Entity Recognition in Resumes.json')
    
    if not ner_dataset_path.exists():
        logging.error("Dataset not found. Please download from Kaggle.")
        return None, None
    
    # Create blank model for processing
    nlp = spacy.blank("en")
    
    # Initialize counters and tracking lists
    entity_stats = {
        "total": 0,
        "accepted": 0,
        "rejected": 0,
        "fixed": 0,
        "filtered_by_length": 0,
        "split_skills": 0,
        "by_type": {}
    }
    
    # Initialize data containers
    all_data = []
    raw_data = []
    
    # First pass: Extract all entities without processing
    logging.info("Extracting raw entities for analysis...")
    try:
        with open(ner_dataset_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Loading JSON data"):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    
                    if 'content' not in item or 'annotation' not in item:
                        continue
                    
                    text = item['content']
                    entities = []
                    
                    for annotation in item['annotation']:
                        if not isinstance(annotation, dict) or 'label' not in annotation or 'points' not in annotation:
                            continue
                        
                        label = annotation['label'][0] if isinstance(annotation['label'], list) else annotation['label']
                        
                        # Skip excluded entity types
                        if label in exclude_entities:
                            continue
                        
                        for point in annotation['points']:
                            if not isinstance(point, dict):
                                continue
                            
                            start = point.get('start')
                            end = point.get('end')
                            
                            if start is not None and end is not None:
                                try:
                                    start, end = int(start), int(end)
                                    
                                    # Skip invalid ranges
                                    if not (0 <= start < end <= len(text)):
                                        continue
                                    
                                    entities.append((start, end, label))
                                except (ValueError, TypeError):
                                    continue
                    
                    if entities:
                        raw_data.append((text, {"entities": entities}))
                
                except Exception as e:
                    continue
    
    except Exception as e:
        logging.error(f"Error loading raw data: {e}")
        return None, None
    
    # Analyze raw entities if requested
    if analyze_before_processing and raw_data:
        logging.info(f"Analyzing {len(raw_data)} documents with raw entities...")
        raw_stats = analyze_entity_statistics(raw_data)
        
        # Display statistics
        print("\n=== RAW ENTITY STATISTICS ===")
        for entity_type, stats in raw_stats.items():
            print(f"\n{entity_type}:")
            print(f"  Count: {stats['count']}")
            print(f"  Avg Length: {stats['avg_length']:.1f} chars")
            print(f"  Length Range: {stats['min_length']} - {stats['max_length']} chars")
            print(f"  Examples: {', '.join(stats['examples'][:3])}")
            
            # Visualize length distribution
            visualize_entity_lengths(raw_stats, entity_type)

    if visualize_patterns and raw_data:
        logging.info("Generating pattern visualizations...")
        entity_types = [
            "Email Address", "Name", "Degree", "Graduation Year", 
            "Years of Experience", "Skills", "Designation", 
            "Companies worked at", "College Name", "Location"
        ]
        
        for entity_type in entity_types:
            print(f"Creating visualizations for {entity_type}...")
            create_pattern_visualizations(raw_data, entity_type)
        
        # Create relationship visualizations
        print("Creating entity relationship visualizations...")
        create_entity_relationship_visualizations(raw_data)
    
    # Create a log file for detailed diagnostics
    with open("entity_processing.log", "w", encoding="utf-8") as log_file:
        log_file.write("=== ENTITY PROCESSING LOG ===\n\n")
        
        try:
            # Second pass: Process and clean the data
            for doc_idx, (text, annotations) in enumerate(tqdm(raw_data, desc="Processing entities")):
                doc = nlp.make_doc(text)
                processed_entities = []
                
                if debug_mode:
                    log_file.write(f"\n--- Document {doc_idx + 1} ---\n")
                    log_file.write(f"Text (first 100 chars): {text[:100]}...\n")
                
                # Process entities
                for start, end, label in annotations["entities"]:
                    # Initialize stats for this entity type if not exists
                    if label not in entity_stats["by_type"]:
                        entity_stats["by_type"][label] = {
                            "total": 0, 
                            "accepted": 0, 
                            "rejected": 0,
                            "fixed": 0,
                            "filtered_by_length": 0
                        }
                    
                    entity_stats["total"] += 1
                    entity_stats["by_type"][label]["total"] += 1
                    
                    # Extract the span text
                    span_text = text[start:end]
                    
                    # Skip if completely empty
                    if not span_text.strip():
                        entity_stats["rejected"] += 1
                        entity_stats["by_type"][label]["rejected"] += 1
                        continue
                    
                    # Filter extremely long entities
                    if len(span_text) > max_entity_length and label != "Email Address":
                        entity_stats["filtered_by_length"] += 1
                        entity_stats["by_type"][label]["filtered_by_length"] += 1
                        if debug_mode:
                            log_file.write(f"  LENGTH FILTERED: '{span_text[:30]}...' ({len(span_text)} chars) - {label}\n")
                        continue

                    if label =="Graduation Year":
                        print(span_text)
                        
                    # Apply entity-specific cleaning
                    cleaned_text, new_start, new_end = clean_entity_by_type(
                        text, start, end, label, log_file, debug_mode
                    )
                    
                    # Skip if normalization emptied the entity
                    if not cleaned_text or not cleaned_text.strip():
                        entity_stats["rejected"] += 1
                        entity_stats["by_type"][label]["rejected"] += 1
                        continue
                    
                    # If boundaries changed, mark as fixed
                    if new_start != start or new_end != end:
                        entity_stats["fixed"] += 1
                        entity_stats["by_type"][label]["fixed"] += 1
                    
                    # Add to processed entities with new boundaries
                    processed_entities.append((new_start, new_end, label))
                
                # Skip if no entities in document
                if not processed_entities:
                    continue
                
                # Align with token boundaries - with progressive fallbacks
                aligned_entities = []
                
                for start, end, label in processed_entities:
                    entity_text = text[start:end]
                    
                    # Try alignment modes in this order: strict -> expand -> contract
                    span = None
                    for alignment_mode in ["strict", "expand", "contract"]:
                        span = doc.char_span(start, end, label=label, alignment_mode=alignment_mode)
                        if span is not None:
                            if debug_mode:
                                log_file.write(f"  {alignment_mode.upper()} ALIGNMENT: '{span.text}' ({span.start_char}:{span.end_char}) Label: {label}\n")
                            break
                    
                    # Skip if all alignment modes failed
                    if span is None:
                        entity_stats["rejected"] += 1
                        entity_stats["by_type"][label]["rejected"] += 1
                        if debug_mode:
                            log_file.write(f"  REJECTED: '{entity_text}' - Failed all alignment modes\n")
                        continue
                        
                    # Check if span is empty or only whitespace
                    if not span.text.strip():
                        entity_stats["rejected"] += 1
                        entity_stats["by_type"][label]["rejected"] += 1
                        if debug_mode:
                            log_file.write(f"  REJECTED: '{entity_text}' -> '{span.text}' - Empty after alignment\n")
                        continue
                        
                    # Check if span has tokens
                    if len(span) == 0:
                        entity_stats["rejected"] += 1
                        entity_stats["by_type"][label]["rejected"] += 1
                        if debug_mode:
                            log_file.write(f"  REJECTED: '{entity_text}' -> '{span.text}' - No tokens\n")
                        continue
                        
                    # Store both character offsets and token indices
                    aligned_entities.append((span.start_char, span.end_char, label, span.start, span.end))
                    entity_stats["accepted"] += 1
                    entity_stats["by_type"][label]["accepted"] += 1
                    
                    if debug_mode:
                        log_file.write(f"  ACCEPTED: '{span.text}' ({span.start_char}:{span.end_char}) Label: {label}\n")
                
                # Skip if no aligned entities
                if not aligned_entities:
                    continue
                
                # Handle token-level overlaps
                final_entities = handle_token_overlaps(aligned_entities, text, debug_mode, log_file)
                
                # Process if we have valid entities after handling overlaps
                if final_entities:
                    # Convert to character offsets for storage
                    char_entities = [(start, end, label) for start, end, label, _, _ in final_entities]
                    
                    # Handle skill splitting if enabled
                    if split_skill_entities:
                        # Find all skills entities
                        has_skills = any(label == "Skills" for _, _, label, _, _ in final_entities)
                        
                        if has_skills:
                            # Get original annotations with character offsets
                            original_annotations = {"entities": char_entities}
                            
                            # Split skills into individual entities
                            split_annotations = split_skills(text, original_annotations, log_file, debug_mode)
                            
                            # Count skill splits
                            orig_skill_count = len([e for e in char_entities if e[2] == "Skills"])
                            new_skill_count = len([e for e in split_annotations["entities"] if e[2] == "Skills"])
                            entity_stats["split_skills"] += (new_skill_count - orig_skill_count)
                            
                            if debug_mode and new_skill_count > orig_skill_count:
                                log_file.write(f"  SKILLS SPLIT: {orig_skill_count} -> {new_skill_count}\n")
                                
                            # Use the split annotations
                            all_data.append((text, split_annotations))
                        else:
                            # No skills to split, just add the data
                            all_data.append((text, {"entities": char_entities}))
                    else:
                        # No skill splitting, add the data as is
                        all_data.append((text, {"entities": char_entities}))
            
            # Split into training and validation sets
            random.seed(42)
            random.shuffle(all_data)
            
            split_idx = int(len(all_data) * train_ratio)
            train_data = all_data[:split_idx]
            val_data = all_data[split_idx:]
            
            # Log statistics
            log_file.write("\n=== ENTITY PROCESSING STATISTICS ===\n")
            log_file.write(f"Total entities processed: {entity_stats['total']}\n")
            log_file.write(f"Entities accepted: {entity_stats['accepted']} ({entity_stats['accepted']/entity_stats['total']:.1%})\n")
            log_file.write(f"Entities rejected: {entity_stats['rejected']} ({entity_stats['rejected']/entity_stats['total']:.1%})\n")
            log_file.write(f"Entities filtered by length: {entity_stats['filtered_by_length']} ({entity_stats['filtered_by_length']/entity_stats['total']:.1%})\n")
            log_file.write(f"Entities fixed: {entity_stats['fixed']} ({entity_stats['fixed']/entity_stats['total']:.1%})\n")
            log_file.write(f"Skills split: {entity_stats['split_skills']}\n\n")
            
            log_file.write("Entity statistics by type:\n")
            for entity_type, stats in entity_stats["by_type"].items():
                acceptance_rate = stats["accepted"] / stats["total"] if stats["total"] > 0 else 0
                log_file.write(f"  {entity_type}: {stats['total']} total, {stats['accepted']} accepted ({acceptance_rate:.1%})\n")
            
            logging.info(f"Processed {len(raw_data)} documents, extracted {entity_stats['total']} raw entities")
            logging.info(f"Accepted {entity_stats['accepted']} entities ({entity_stats['accepted']/entity_stats['total']:.1%})")
            logging.info(f"Created {len(all_data)} valid examples with clean entities")
            
            # Log entity statistics
            for entity_type, stats in entity_stats["by_type"].items():
                acceptance_rate = stats["accepted"] / stats["total"] if stats["total"] > 0 else 0
                logging.info(f"  {entity_type}: {stats['accepted']}/{stats['total']} ({acceptance_rate:.1%})")
            
            logging.info(f"Split into {len(train_data)} training and {len(val_data)} validation examples")
            
            # Analyze processed data if requested
            if analyze_before_processing and all_data:
                logging.info("Analyzing processed entities...")
                processed_stats = analyze_entity_statistics(all_data)
                
                # Display statistics
                print("\n=== PROCESSED ENTITY STATISTICS ===")
                for entity_type, stats in processed_stats.items():
                    print(f"\n{entity_type}:")
                    print(f"  Count: {stats['count']} (Raw: {entity_stats['by_type'].get(entity_type, {}).get('total', 0)})")
                    print(f"  Avg Length: {stats['avg_length']:.1f} chars")
                    print(f"  Length Range: {stats['min_length']} - {stats['max_length']} chars")
                    print(f"  Examples: {', '.join(stats['examples'][:3])}")
            
            # Convert to spaCy binary format
            train_doc_bin = DocBin()
            train_entities = 0
            
            for idx, (text, annotations) in enumerate(tqdm(train_data, desc="Creating training data")):
                doc = nlp.make_doc(text)
                ents = []
                
                # Token-level overlap prevention during binary conversion
                taken_tokens = set()
                
                for start, end, label in sorted(annotations["entities"], 
                                               key=lambda x: (entity_priority(x[2]), x[1] - x[0]), 
                                               reverse=True):
                    # Convert to span object (using strict mode for final dataset)
                    span = doc.char_span(start, end, label=label, alignment_mode="strict")
                    
                    # Skip invalid spans
                    if span is None or not span.text.strip() or len(span) == 0:
                        continue
                        
                    # Check for token overlap
                    span_tokens = set(range(span.start, span.end))
                    if span_tokens.intersection(taken_tokens):
                        if debug_mode:
                            log_file.write(f"  OVERLAP SKIPPED IN BINARY: '{span.text}' Label: {label}\n")
                        continue
                        
                    # Add span and mark tokens as taken
                    ents.append(span)
                    taken_tokens.update(span_tokens)
                    train_entities += 1
                
                # Only add documents with entities
                if ents:
                    doc.ents = ents
                    train_doc_bin.add(doc)
            
            # Similar process for validation data
            dev_doc_bin = DocBin()
            dev_entities = 0
            
            for text, annotations in tqdm(val_data, desc="Creating validation data"):
                doc = nlp.make_doc(text)
                ents = []
                
                # Token-level overlap prevention during binary conversion
                taken_tokens = set()
                
                for start, end, label in sorted(annotations["entities"], 
                                              key=lambda x: (entity_priority(x[2]), x[1] - x[0]), 
                                              reverse=True):
                    # Convert to span object
                    span = doc.char_span(start, end, label=label, alignment_mode="strict")
                    
                    # Skip invalid spans
                    if span is None or not span.text.strip() or len(span) == 0:
                        continue
                        
                    # Check for token overlap
                    span_tokens = set(range(span.start, span.end))
                    if span_tokens.intersection(taken_tokens):
                        continue
                        
                    # Add span and mark tokens as taken
                    ents.append(span)
                    taken_tokens.update(span_tokens)
                    dev_entities += 1
                
                # Only add documents with entities
                if ents:
                    doc.ents = ents
                    dev_doc_bin.add(doc)
            
            # Save to disk
            train_doc_bin.to_disk(train_output_path)
            dev_doc_bin.to_disk(dev_output_path)
            
            logging.info(f"Saved {train_entities} entities in training data")
            logging.info(f"Saved {dev_entities} entities in validation data")
            
            # Run spaCy validation if requested
            if run_validation:
                print("\n=== RUNNING SPACY VALIDATION ===")
                try:
                    # Create a basic config file if it doesn't exist
                    config_path = Path("./config.cfg")
                    if not config_path.exists():
                        subprocess.run(
                            ["python", "-m", "spacy", "init", "config", 
                             "config.cfg", "--lang", "en", "--pipeline", "ner"],
                            check=True
                        )
                    
                    # Run the validation command
                    print(f"Running: python -m spacy debug data config.cfg " + 
                          f"--paths.train {train_output_path} --paths.dev {dev_output_path}")
                    
                    result = subprocess.run(
                        ["python", "-m", "spacy", "debug", "data", 
                         "config.cfg", 
                         "--paths.train", train_output_path,
                         "--paths.dev", dev_output_path],
                        capture_output=True, 
                        text=True,
                        check=False
                    )
                    
                    print("\n=== VALIDATION RESULTS ===")
                    print(result.stdout)
                
                except Exception as e:
                    print(f"Error running validation: {e}")
            
            return {
                'train_data': train_data, 
                'val_data': val_data,
                'stats': entity_stats
            }
            
        except Exception as e:
            logging.error(f"Error processing data: {e}")
            import traceback
            logging.error(traceback.format_exc())
            return None, None