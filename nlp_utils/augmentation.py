# nlp_utils/augmentation.py
# This module contains augmentation functions for NER data.
import copy
import json
import logging
import random
import re

import numpy as np
from .normalization import entity_priority
from pathlib import Path
from matplotlib import pyplot as plt
import spacy
from tqdm import tqdm
from faker import Faker
from spacy.tokens import DocBin
faker = Faker()


def create_balanced_augmentation(result_dict):
    """
    Create a balanced augmentation approach based on model performance.
    
    Args:
        result_dict: Dictionary from load_and_export_ner_data function
    """
    train_data = result_dict['train_data']
    val_data = result_dict['val_data']
    
    logging.info("Starting targeted, balanced augmentation based on model performance...")
    
    # Apply our targeted augmentation
    augmented_train_data = targeted_entity_augmentation(train_data, val_data)
    
    # Convert to spaCy binary format
    train_output_path = "./augmented_train.spacy"
    val_output_path = "./dev.spacy"
    
    # Process and convert augmented data to spaCy format
    nlp = spacy.blank("en")
    
    # Convert augmented training data
    train_doc_bin = DocBin()
    train_entities = 0
    
    for text, annotations in tqdm(augmented_train_data, desc="Converting augmented training data"):
        doc = nlp.make_doc(text)
        ents = []
        
        # Token-level overlap prevention
        taken_tokens = set()
        
        for start, end, label in sorted(annotations["entities"], 
                                       key=lambda x: (entity_priority(x[2]), x[1] - x[0]), 
                                       reverse=True):
            # Create span with strict alignment
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
            train_entities += 1
        
        # Only add documents with entities
        if ents:
            doc.ents = ents
            train_doc_bin.add(doc)
    
    # Convert validation data (keep original)
    val_doc_bin = DocBin()
    val_entities = 0
    
    for text, annotations in tqdm(val_data, desc="Converting validation data"):
        doc = nlp.make_doc(text)
        ents = []
        
        # Token-level overlap prevention
        taken_tokens = set()
        
        for start, end, label in sorted(annotations["entities"], 
                                      key=lambda x: (entity_priority(x[2]), x[1] - x[0]), 
                                      reverse=True):
            # Create span with strict alignment
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
            val_entities += 1
        
        # Only add documents with entities
        if ents:
            doc.ents = ents
            val_doc_bin.add(doc)
    
    # Save to disk
    train_doc_bin.to_disk(train_output_path)
    val_doc_bin.to_disk(val_output_path)
    
    logging.info(f"Saved {train_entities} entities in augmented training data")
    logging.info(f"Saved {val_entities} entities in validation data")
    
    # Also save as JSON for inspection
    with open("augmented_train_data.json", "w", encoding="utf-8") as f:
        json.dump([{"text": text, "entities": [[s, e, l] for s, e, l in anns["entities"]]} 
                  for text, anns in augmented_train_data], f, indent=2)
    
    # Show summary statistics
    logging.info("\n=== AUGMENTATION SUMMARY ===")
    logging.info(f"Original training examples: {len(train_data)}")
    logging.info(f"Augmented training examples: {len(augmented_train_data)}")
    logging.info(f"Augmentation factor: {len(augmented_train_data) / len(train_data):.1f}x")
    
    # Visualize entity distribution
    entity_counts_before = count_entities_by_type(train_data)
    entity_counts_after = count_entities_by_type(augmented_train_data)
    
    plot_entity_distribution_comparison(entity_counts_before, entity_counts_after)
    
    return {
        "train_data": augmented_train_data,
        "val_data": val_data,
        "train_count": len(augmented_train_data),
        "val_count": len(val_data),
        "entity_counts_before": entity_counts_before,
        "entity_counts_after": entity_counts_after
    }


def convert_and_save_augmented_spacy_files(
    train_data, val_data,
    train_output_path="augmented_train.spacy",
    dev_output_path="augmented_dev.spacy",
    debug_mode=False,
    run_validation=False
):
    nlp = spacy.blank("en")
    train_doc_bin = DocBin(store_user_data=True)
    dev_doc_bin = DocBin(store_user_data=True)

    train_entities = 0
    dev_entities = 0

    # Optional log file
    log_file = open("ner_overlap_debug.log", "w") if debug_mode else None

    for idx, (text, annotations) in enumerate(tqdm(train_data, desc="Creating training data")):
        doc = nlp.make_doc(text)
        ents = []
        taken_tokens = set()

        for start, end, label in sorted(annotations["entities"], key=lambda x: (entity_priority(x[2]), x[1] - x[0]), reverse=True):
            span = doc.char_span(start, end, label=label, alignment_mode="strict")
            if span is None or not span.text.strip():
                continue
            span_tokens = set(range(span.start, span.end))
            if span_tokens.intersection(taken_tokens):
                if debug_mode:
                    log_file.write(f"OVERLAP SKIPPED: '{span.text}' Label: {label}\n")
                continue
            ents.append(span)
            taken_tokens.update(span_tokens)
            train_entities += 1

        if ents:
            doc.ents = ents
            train_doc_bin.add(doc)

    for text, annotations in tqdm(val_data, desc="Creating validation data"):
        doc = nlp.make_doc(text)
        ents = []
        taken_tokens = set()

        for start, end, label in sorted(annotations["entities"], key=lambda x: (entity_priority(x[2]), x[1] - x[0]), reverse=True):
            span = doc.char_span(start, end, label=label, alignment_mode="strict")
            if span is None or not span.text.strip():
                continue
            span_tokens = set(range(span.start, span.end))
            if span_tokens.intersection(taken_tokens):
                continue
            ents.append(span)
            taken_tokens.update(span_tokens)
            dev_entities += 1

        if ents:
            doc.ents = ents
            dev_doc_bin.add(doc)

    train_doc_bin.to_disk(train_output_path)
    dev_doc_bin.to_disk(dev_output_path)

    logging.info(f"Saved {train_entities} entities in training data")
    logging.info(f"Saved {dev_entities} entities in validation data")

    if debug_mode:
        log_file.close()

    if run_validation:
        import subprocess
        config_path = Path("./config.cfg")
        if not config_path.exists():
            subprocess.run(
                ["python", "-m", "spacy", "init", "config", "config.cfg", "--lang", "en", "--pipeline", "ner"],
                check=True
            )

        result = subprocess.run(
            ["python", "-m", "spacy", "debug", "data", 
             "config.cfg", 
             "--paths.train", train_output_path,
             "--paths.dev", dev_output_path],
            capture_output=True, text=True
        )
        print("\n=== SPACY VALIDATION ===")
        print(result.stdout)

def targeted_entity_augmentation(train_data, val_data=None):
    """
    Applies targeted augmentation strategies based on current model performance.
    """
    logging.info("Starting targeted NER data augmentation...")
    logging.info(f"Original training examples: {len(train_data)}")
    
    # Create a copy of the original data
    augmented_data = copy.deepcopy(train_data)
    
    # Track entity counts for reporting
    entity_counts_before = count_entities_by_type(train_data)
    
    # Apply targeted strategies based on current model performance
    augmented_data.extend(augment_low_recall_entities(train_data))
    augmented_data.extend(augment_low_precision_entities(train_data))
    augmented_data.extend(augment_skills_extensively(train_data))
    augmented_data.extend(generate_boundary_edge_cases(train_data))
    
    # Deduplicate examples
    unique_data = deduplicate_data(augmented_data)
    
    # Track entity counts after augmentation
    entity_counts_after = count_entities_by_type(unique_data)
    
    # Log augmentation results
    logging.info(f"Augmented training examples: {len(unique_data)}")
    log_augmentation_statistics(entity_counts_before, entity_counts_after)
    
    return unique_data

def plot_entity_distribution_comparison(before, after):
    """
    Plot comparison of entity distributions before and after augmentation.
    """
    plt.figure(figsize=(14, 8))
    
    # Combine all entity types from both dictionaries
    all_types = sorted(set(list(before.keys()) + list(after.keys())))
    
    # Create data for plotting
    x = np.arange(len(all_types))
    width = 0.35
    
    # Extract counts, ensuring 0 for missing keys
    before_counts = [before.get(t, 0) for t in all_types]
    after_counts = [after.get(t, 0) for t in all_types]
    
    # Plot bars
    plt.bar(x - width/2, before_counts, width, label='Before Augmentation')
    plt.bar(x + width/2, after_counts, width, label='After Augmentation')
    
    # Customize plot
    plt.xlabel('Entity Type')
    plt.ylabel('Count')
    plt.title('Entity Distribution Before and After Augmentation')
    plt.xticks(x, all_types, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    
    # Add value labels on bars
    for i, v in enumerate(before_counts):
        plt.text(i - width/2, v + 10, str(v), ha='center')
        
    for i, v in enumerate(after_counts):
        plt.text(i + width/2, v + 10, str(v), ha='center')
    
    plt.savefig('entity_augmentation_comparison.png')
    plt.show()


def augment_low_recall_entities(data, factor=3):
    """
    Focus on improving recall for entities like Designation, Companies worked at, and Degree
    by generating more diverse examples and contextual variations.
    """
    logging.info("Augmenting entities with low recall (Designation, Companies worked at, Degree)")
    augmented = []
    
    # Extract all examples of these entity types
    designations = extract_entities_of_type(data, "Designation")
    companies = extract_entities_of_type(data, "Companies worked at")
    degrees = extract_entities_of_type(data, "Degree")
    
    # Generate position title variations
    designation_variations = generate_designation_variations(designations)
    
    # Generate company name variations
    company_variations = generate_company_variations(companies)
    
    # Generate degree variations
    degree_variations = generate_degree_variations(degrees)
    
    # Apply entity swapping and contextual enrichment
    for text, annotations in tqdm(data, desc="Augmenting low-recall entities"):
        entities = annotations["entities"]
        has_target_entity = any(label in ["Designation", "Companies worked at", "Degree"] 
                               for _, _, label in entities)
        
        if not has_target_entity:
            continue
            
        # Create multiple augmented versions
        for _ in range(factor):
            aug_text = text
            aug_entities = []
            offset = 0
            
            # Sort entities by position
            sorted_entities = sorted(entities, key=lambda x: x[0])
            
            for start, end, label in sorted_entities:
                entity_text = text[start:end]
                
                # Generate variations for target entity types
                if label == "Designation":
                    if random.random() < 0.8:  # 80% chance to substitute
                        new_designation = random.choice(designation_variations)
                        before = aug_text[:start + offset]
                        after = aug_text[end + offset:]
                        aug_text = before + new_designation + after
                        
                        new_offset = offset + len(new_designation) - len(entity_text)
                        aug_entities.append((start + offset, start + offset + len(new_designation), label))
                        offset = new_offset
                        continue
                
                elif label == "Companies worked at":
                    if random.random() < 0.7:  # 70% chance to substitute
                        new_company = random.choice(company_variations)
                        before = aug_text[:start + offset]
                        after = aug_text[end + offset:]
                        aug_text = before + new_company + after
                        
                        new_offset = offset + len(new_company) - len(entity_text)
                        aug_entities.append((start + offset, start + offset + len(new_company), label))
                        offset = new_offset
                        continue
                        
                elif label == "Degree":
                    if random.random() < 0.8:  # 80% chance to substitute
                        new_degree = random.choice(degree_variations)
                        before = aug_text[:start + offset]
                        after = aug_text[end + offset:]
                        aug_text = before + new_degree + after
                        
                        new_offset = offset + len(new_degree) - len(entity_text)
                        aug_entities.append((start + offset, start + offset + len(new_degree), label))
                        offset = new_offset
                        continue
                
                # For other entities, keep them as is
                aug_entities.append((start + offset, end + offset, label))
            
            augmented.append((aug_text, {"entities": aug_entities}))
            
    # Generate synthetic examples with rich context for these entities
    synthetic_examples = generate_synthetic_context_examples(factor * 2)
    augmented.extend(synthetic_examples)
    
    logging.info(f"Created {len(augmented)} examples for low-recall entities")
    return augmented

def generate_designation_variations(designations):
    """
    Generate variations of job titles and designations to improve recall.
    """
    base_titles = set()
    for designation in designations:
        # Extract root title
        base = re.sub(r'^(Senior|Junior|Lead|Principal|Chief|Associate|Assistant)\s+', '', designation)
        base = re.sub(r'\s+(I|II|III|IV|V)$', '', base)
        if len(base) > 3:  # Avoid too short titles
            base_titles.add(base)
    
    variations = []
    prefixes = ['Senior', 'Junior', 'Lead', 'Principal', 'Chief', 'Associate', 'Assistant', '']
    suffixes = [' I', ' II', ' III', '', ' Manager', ' Lead']
    
    for base in base_titles:
        for prefix in prefixes:
            for suffix in suffixes:
                variation = f"{prefix} {base}{suffix}".strip()
                if variation and variation != base:
                    variations.append(variation)
    
    # Add completely new designations
    additional_titles = [
        "Machine Learning Engineer", "Cloud Architect", "DevOps Specialist",
        "AI Researcher", "Data Engineer", "Blockchain Developer",
        "Frontend Engineer", "Backend Developer", "Full Stack Engineer",
        "Site Reliability Engineer", "UX Designer", "UI Developer",
        "Product Owner", "Scrum Master", "Technical Program Manager",
        "Solutions Architect", "Systems Analyst", "Network Administrator",
        "Information Security Analyst", "Database Administrator"
    ]
    
    variations.extend(additional_titles)
    
    # Add fake but realistic titles from Faker
    for _ in range(50):
        variations.append(faker.job())
    
    return list(set(variations))

def generate_company_variations(companies):
    """
    Generate company name variations to improve recall.
    """
    variations = []
    suffixes = [' Inc.', ' LLC', ' Ltd.', ' Corporation', ' Corp.', ' Company', 
                ' Technologies', ' Group', ' Solutions', ' International', '']
    
    for company in companies:
        # Remove existing suffix if any
        base = re.sub(r'\s+(Inc|LLC|Ltd|Corporation|Corp|Company|Technologies|Group|Solutions|International)\.?$', '', company)
        
        # Add different suffixes
        for suffix in suffixes:
            if not company.endswith(suffix):
                variation = f"{base}{suffix}".strip()
                if variation and variation != company:
                    variations.append(variation)
    
    # Add completely new company names
    tech_companies = [
        "Quantum Computing", "Neural Dynamics", "Cloud Solutions",
        "Data Insights", "Blockchain Innovations", "Tech Frontiers",
        "Digital Transformation", "AI Systems", "Smart Analytics",
        "Future Technologies", "Cyber Security Solutions", "Virtual Systems",
        "Global Software", "Mobile Innovations", "Enterprise Solutions"
    ]
    
    for company in tech_companies:
        for suffix in suffixes:
            variations.append(f"{company}{suffix}".strip())
    
    # Add fake but realistic company names from Faker
    for _ in range(50):
        variations.append(faker.company())
    
    return list(set(variations))

def generate_degree_variations(degrees):
    """
    Generate degree variations to improve recall.
    """
    variations = []
    
    # Common degree types and their variations
    degree_types = {
        "Bachelor": ["Bachelor of", "Bachelor's in", "Bachelor's degree in", "B.S. in", "B.A. in", "BS in", "BA in"],
        "Master": ["Master of", "Master's in", "Master's degree in", "M.S. in", "M.A. in", "MS in", "MA in"],
        "PhD": ["PhD in", "Ph.D. in", "Doctorate in", "Doctoral degree in"],
        "Associate": ["Associate of", "Associate's in", "A.S. in", "A.A. in"]
    }
    
    # Common fields of study
    fields = [
        "Computer Science", "Information Technology", "Software Engineering", 
        "Data Science", "Artificial Intelligence", "Business Administration",
        "Information Systems", "Electrical Engineering", "Computer Engineering",
        "Mathematics", "Statistics", "Economics", "Finance", "Marketing",
        "Management", "Human Resources", "Psychology", "Communications"
    ]
    
    # Generate variations
    for degree_type, variations_list in degree_types.items():
        for variation in variations_list:
            for field in fields:
                variations.append(f"{variation} {field}")
    
    # Add existing degrees with slight modifications
    for degree in degrees:
        # Try different abbreviations and formatting
        degree = degree.replace("Bachelor of", "B.S. in")
        degree = degree.replace("Master of", "M.S. in")
        variations.append(degree)
        
        # Add "honors" or other qualifiers
        if "Bachelor" in degree or "B.S." in degree or "B.A." in degree:
            variations.append(f"{degree} with Honors")
            variations.append(f"{degree} (Honours)")
    
    return list(set(variations))

def generate_synthetic_context_examples(count=100):
    """
    Generate synthetic resume examples with rich context around entities.
    """
    synthetic_examples = []
    
    # Templates with rich context
    templates = [
        "Currently working as {DESIGNATION} at {COMPANIES_WORKED_AT} where I lead projects and initiatives.",
        "Previously served as {DESIGNATION} at {COMPANIES_WORKED_AT} with responsibilities in team management.",
        "Graduated with {DEGREE} and immediately joined {COMPANIES_WORKED_AT} as a {DESIGNATION}.",
        "My role as {DESIGNATION} at {COMPANIES_WORKED_AT} involves strategic planning and execution.",
        "{DEGREE} graduate with experience as {DESIGNATION} at multiple companies including {COMPANIES_WORKED_AT}.",
        "After completing my {DEGREE}, I worked at {COMPANIES_WORKED_AT} as their {DESIGNATION}.",
        "Managed teams as a {DESIGNATION} at {COMPANIES_WORKED_AT} after earning my {DEGREE}."
    ]
    
    # Generate synthetic examples
    for _ in range(count):
        template = random.choice(templates)
        
        # Generate entities
        designation = random.choice([
            "Software Engineer", "Data Scientist", "Project Manager", "Product Manager",
            "Senior Developer", "Technical Lead", "Engineering Manager", "CTO",
            "System Architect", "DevOps Engineer", "Frontend Developer", "Backend Engineer",
            "Full Stack Developer", "Machine Learning Engineer", "UI/UX Designer"
        ])
        
        company = random.choice([
            "Google", "Microsoft", "Amazon", "Apple", "Facebook", "Netflix", 
            "IBM", "Oracle", "Intel", "Salesforce", "Adobe", "Twitter",
            "LinkedIn", "Uber", "Airbnb", "Slack", "Spotify", "PayPal",
            "Dropbox", "Square", "Stripe", "Twilio", "Atlassian"
        ])
        
        degree = random.choice([
            "Bachelor of Science in Computer Science",
            "Master of Science in Data Science",
            "MBA in Technology Management",
            "Bachelor of Engineering in Software Development",
            "Master's in Information Technology",
            "PhD in Computer Engineering",
            "Bachelor's degree in Information Systems"
        ])
        
        # Replace placeholders
        text = template.replace("{DESIGNATION}", designation)
        text = text.replace("{COMPANIES_WORKED_AT}", company)
        text = text.replace("{DEGREE}", degree)
        
        # Create entities list
        entities = []
        
        # Find positions of each entity in text
        designation_pos = text.find(designation)
        if designation_pos >= 0:
            entities.append((designation_pos, designation_pos + len(designation), "Designation"))
            
        company_pos = text.find(company)
        if company_pos >= 0:
            entities.append((company_pos, company_pos + len(company), "Companies worked at"))
            
        degree_pos = text.find(degree)
        if degree_pos >= 0:
            entities.append((degree_pos, degree_pos + len(degree), "Degree"))
        
        if entities:
            synthetic_examples.append((text, {"entities": entities}))
    
    return synthetic_examples
def augment_low_precision_entities(data, factor=2):
    """
    Focus on improving precision for entities like Email Address and College Name
    by adding negative examples and boundary cases.
    """
    logging.info("Augmenting entities with low precision (Email Address, College Name)")
    augmented = []
    
    # Generate hard negative examples
    for _ in range(factor):
        for text, annotations in tqdm(data, desc="Creating precision-focused examples"):
            entities = annotations["entities"]
            
            # Only process documents with target entities
            has_target = any(label in ["Email Address", "College Name"] for _, _, label in entities)
            if not has_target:
                continue
            
            # Create a new example with more challenging boundaries
            aug_text = text
            aug_entities = []
            offset = 0
            
            # Sort entities by position
            sorted_entities = sorted(entities, key=lambda x: x[0])
            
            for start, end, label in sorted_entities:
                entity_text = text[start:end]
                
                if label == "Email Address":
                    # Add challenging email variations
                    if random.random() < 0.6:
                        # Make emails more complex with dots, dashes, numbers
                        parts = entity_text.split('@')
                        if len(parts) == 2:
                            username = parts[0]
                            domain = parts[1]
                            
                            # Add complexity to username
                            chars = list(username)
                            if len(chars) > 3:
                                insert_pos = random.randint(1, len(chars) - 1)
                                chars.insert(insert_pos, random.choice(['_', '.', '-', '123', '2023']))
                            
                            new_email = ''.join(chars) + '@' + domain
                            
                            # Replace in text
                            before = aug_text[:start + offset]
                            after = aug_text[end + offset:]
                            aug_text = before + new_email + after
                            
                            # Update offset
                            new_offset = offset + len(new_email) - len(entity_text)
                            aug_entities.append((start + offset, start + offset + len(new_email), label))
                            offset = new_offset
                            continue
                
                elif label == "College Name":
                    # Add challenging college name variations
                    if random.random() < 0.7:
                        # Add department or location information to college names
                        variations = [
                            f"{entity_text}, Department of Computer Science",
                            f"{entity_text} - School of Engineering",
                            f"{entity_text} (Main Campus)",
                            f"The {entity_text}",
                            f"{entity_text} Online"
                        ]
                        
                        new_college = random.choice(variations)
                        
                        # Replace in text
                        before = aug_text[:start + offset]
                        after = aug_text[end + offset:]
                        aug_text = before + new_college + after
                        
                        # Update offset
                        new_offset = offset + len(new_college) - len(entity_text)
                        aug_entities.append((start + offset, start + offset + len(new_college), label))
                        offset = new_offset
                        continue
                
                # For other entities, keep as is
                aug_entities.append((start + offset, end + offset, label))
            
            # Add non-email text that looks like email to challenge the model
            if random.random() < 0.3:
                almost_email = generate_email_lookalike()
                challenge_pos = find_safe_insertion_point(aug_text, aug_entities)
                if challenge_pos:
                    before = aug_text[:challenge_pos]
                    after = aug_text[challenge_pos:]
                    aug_text = before + " " + almost_email + " " + after
                    
                    # Adjust entity positions after insertion
                    insertion_len = len(almost_email) + 2  # +2 for spaces
                    aug_entities = [
                        (s, e, l) if s < challenge_pos else (s + insertion_len, e + insertion_len, l) 
                        for s, e, l in aug_entities
                    ]
            
            # Add almost-college text
            if random.random() < 0.3:
                almost_college = generate_college_lookalike()
                challenge_pos = find_safe_insertion_point(aug_text, aug_entities)
                if challenge_pos:
                    before = aug_text[:challenge_pos]
                    after = aug_text[challenge_pos:]
                    aug_text = before + " " + almost_college + " " + after
                    
                    # Adjust entity positions after insertion
                    insertion_len = len(almost_college) + 2  # +2 for spaces
                    aug_entities = [
                        (s, e, l) if s < challenge_pos else (s + insertion_len, e + insertion_len, l) 
                        for s, e, l in aug_entities
                    ]
            
            augmented.append((aug_text, {"entities": aug_entities}))
    
    logging.info(f"Created {len(augmented)} precision-focused examples")
    return augmented

def generate_email_lookalike():
    """
    Generate text that looks like email but isn't valid.
    """
    almost_emails = [
        "john.doe[at]gmail.com",
        "contact-us(example.com)",
        "email: info@company",
        "www.example.com/contact",
        "user@localhost",
        "firstname_lastname@",
        "info(at)company.com",
        "email-address.com"
    ]
    return random.choice(almost_emails)

def generate_college_lookalike():
    """
    Generate text that looks like college name but shouldn't be labeled as one.
    """
    almost_colleges = [
        "Company University Program",
        "Educational Resources Inc",
        "The Learning Center",
        "Professional Academy",
        "Training Institute",
        "Certification Program",
        "Corporate University",
        "The Knowledge Hub"
    ]
    return random.choice(almost_colleges)

def find_safe_insertion_point(text, entities):
    """
    Find a position in text that doesn't overlap with entities.
    """
    if not entities:
        return len(text) // 2
    
    # Find gaps between entities
    sorted_entities = sorted(entities, key=lambda x: x[0])
    gaps = []
    
    # Gap at the beginning
    if sorted_entities[0][0] > 20:
        gaps.append(10)
    
    # Gaps between entities
    for i in range(len(sorted_entities) - 1):
        current_end = sorted_entities[i][1]
        next_start = sorted_entities[i+1][0]
        
        if next_start - current_end > 20:
            gaps.append(current_end + (next_start - current_end) // 2)
    
    # Gap at the end
    if len(text) - sorted_entities[-1][1] > 20:
        gaps.append(sorted_entities[-1][1] + 10)
    
    if gaps:
        return random.choice(gaps)
    else:
        return None
def augment_skills_extensively(data, factor=3):
    """
    Extensively augment Skills entities which have the lowest performance.
    """
    logging.info("Extensively augmenting Skills entities")
    augmented = []
    
    # Extract existing skills
    all_skills = extract_entities_of_type(data, "Skills")
    
    # Add more skills variations and combinations
    expanded_skills = expand_skills_vocabulary(all_skills)
    
    # Create skill combination patterns
    skill_patterns = [
        "Skills: {skills_list}",
        "Technical Skills: {skills_list}",
        "Proficient in: {skills_list}",
        "Core Competencies: {skills_list}",
        "Expert in {skills_list}",
        "Technologies: {skills_list}",
        "Programming Languages: {skills_list}",
        "Technical Expertise: {skills_list}"
    ]
    
    # Separators for skills lists
    separators = [", ", " | ", "; ", " • ", "\n- ", ", and "]
    
    # Generate skills-focused examples
    for _ in range(factor):
        for text, annotations in tqdm(data, desc="Augmenting skills"):
            entities = annotations["entities"]
            
            # Check if document has skills
            has_skills = any(label == "Skills" for _, _, label in entities)
            
            if has_skills:
                # Get existing skills in the document
                doc_skills = []
                for start, end, label in entities:
                    if label == "Skills":
                        doc_skills.append(text[start:end])
                
                # Create new skills list with variations
                new_skills = doc_skills.copy()
                
                # Add some more skills
                additional_skills = random.sample(expanded_skills, k=min(5, len(expanded_skills)))
                new_skills.extend(additional_skills)
                
                # Deduplicate
                new_skills = list(set(new_skills))
                
                # Create a skills section
                separator = random.choice(separators)
                skills_list = separator.join(new_skills)
                pattern = random.choice(skill_patterns)
                skills_section = pattern.replace("{skills_list}", skills_list)
                
                # Create a new document with these skills
                new_text = skills_section
                new_entities = []
                
                # Mark each skill as an entity
                current_pos = 0
                for skill_pattern in [f"{s}" for s in new_skills]:
                    skill_pos = new_text.find(skill_pattern, current_pos)
                    if skill_pos >= 0:
                        new_entities.append((skill_pos, skill_pos + len(skill_pattern), "Skills"))
                        current_pos = skill_pos + len(skill_pattern)
                
                # Add other essential entities to make the example more realistic
                if random.random() < 0.7:
                    name = faker.name()
                    name_pos = 0
                    new_text = name + "\n" + new_text
                    new_entities.append((0, len(name), "Name"))
                    
                    # Adjust other entity positions
                    new_entities = [(s + len(name) + 1, e + len(name) + 1, l) for s, e, l in new_entities]
                
                if new_entities:
                    augmented.append((new_text, {"entities": new_entities}))
            
            # Also create variations of existing documents with skills
            if has_skills:
                aug_text = text
                aug_entities = []
                offset = 0
                
                # Sort entities by position
                sorted_entities = sorted(entities, key=lambda x: x[0])
                
                for start, end, label in sorted_entities:
                    entity_text = text[start:end]
                    
                    if label == "Skills" and random.random() < 0.8:
                        # Replace with a more specific skill variation
                        new_skill = get_specific_skill_variation(entity_text, expanded_skills)
                        
                        before = aug_text[:start + offset]
                        after = aug_text[end + offset:]
                        aug_text = before + new_skill + after
                        
                        new_offset = offset + len(new_skill) - len(entity_text)
                        aug_entities.append((start + offset, start + offset + len(new_skill), label))
                        offset = new_offset
                    else:
                        # Keep other entities as is
                        aug_entities.append((start + offset, end + offset, label))
                
                augmented.append((aug_text, {"entities": aug_entities}))
    
    # Add synthetic skills-focused examples
    synthetic_skills_examples = generate_synthetic_skills_examples(expanded_skills, factor * 10)
    augmented.extend(synthetic_skills_examples)
    
    logging.info(f"Created {len(augmented)} skills-focused examples")
    return augmented

def expand_skills_vocabulary(existing_skills):
    """
    Expand skills vocabulary with variations and additional skills.
    """
    expanded = set(existing_skills)
    
    # Technical skills
    programming_languages = [
        "Python", "Java", "JavaScript", "TypeScript", "C++", "C#", "Go", "Rust",
        "Swift", "Kotlin", "PHP", "Ruby", "Scala", "R", "MATLAB", "Perl", "Shell"
    ]
    
    web_technologies = [
        "HTML", "CSS", "React", "Angular", "Vue.js", "Node.js", "Express.js", "Django",
        "Flask", "Spring Boot", "ASP.NET", "jQuery", "Bootstrap", "Tailwind CSS", 
        "GraphQL", "REST API", "SOAP", "WebSockets"
    ]
    
    databases = [
        "SQL", "MySQL", "PostgreSQL", "MongoDB", "SQLite", "Oracle", "SQL Server",
        "Redis", "Cassandra", "DynamoDB", "Firebase", "Neo4j", "Elasticsearch"
    ]
    
    cloud_technologies = [
        "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "Terraform",
        "Jenkins", "GitLab CI/CD", "GitHub Actions", "Ansible", "Puppet", "Chef",
        "Serverless", "Lambda", "S3", "EC2", "Azure Functions", "Cloud Run"
    ]
    
    data_science = [
        "Machine Learning", "Deep Learning", "NLP", "Computer Vision", "Data Analysis",
        "Statistical Modeling", "TensorFlow", "PyTorch", "scikit-learn", "pandas",
        "NumPy", "SciPy", "Matplotlib", "Tableau", "Power BI", "Big Data", "Hadoop",
        "Spark", "Airflow", "Jupyter", "Neural Networks"
    ]
    
    other_tech = [
        "Git", "Agile", "Scrum", "Kanban", "Jira", "Confluence", "DevOps", "CI/CD",
        "Test-Driven Development", "Microservices", "RESTful API", "OOP", "Linux",
        "Unix", "Windows Server", "Networking", "Security", "A/B Testing"
    ]
    
    # Add all these skills
    expanded.update(programming_languages)
    expanded.update(web_technologies)
    expanded.update(databases)
    expanded.update(cloud_technologies)
    expanded.update(data_science)
    expanded.update(other_tech)
    
    # Create variations with frameworks and tools
    variations = []
    for skill in ["Python", "JavaScript", "Java", "C#"]:
        if skill == "Python":
            variations.extend([
                "Python Django", "Python Flask", "Python FastAPI", "Python Pandas",
                "Python scikit-learn", "Python Data Analysis", "Python Automation"
            ])
        elif skill == "JavaScript":
            variations.extend([
                "JavaScript React", "JavaScript Node.js", "JavaScript Angular",
                "JavaScript Vue", "JavaScript Express", "JavaScript Front-end"
            ])
        elif skill == "Java":
            variations.extend([
                "Java Spring", "Java Hibernate", "Java J2EE", "Java Android",
                "Java Microservices", "Java Backend Development"
            ])
        elif skill == "C#":
            variations.extend([
                "C# .NET", "C# ASP.NET", "C# Unity", "C# WPF", "C# Xamarin",
                "C# Entity Framework"
            ])
    
    expanded.update(variations)
    
    # Add specific versions
    version_variations = []
    for skill in ["Python", "Java", "JavaScript", "React", "Angular"]:
        if skill == "Python":
            version_variations.extend(["Python 3.8", "Python 3.9", "Python 3.10"])
        elif skill == "Java":
            version_variations.extend(["Java 11", "Java 17", "Java 8"])
        elif skill == "JavaScript":
            version_variations.extend(["JavaScript ES6", "JavaScript ES2022"])
        elif skill == "React":
            version_variations.extend(["React 16", "React 17", "React 18"])
        elif skill == "Angular":
            version_variations.extend(["Angular 12", "Angular 13", "Angular 14"])
    
    expanded.update(version_variations)
    
    return list(expanded)

def get_specific_skill_variation(skill, expanded_skills):
    """
    Get a more specific variation of a skill.
    """
    # Find more specific variations
    base_skill = skill.lower()
    specific_variations = []
    
    for exp_skill in expanded_skills:
        if base_skill in exp_skill.lower() and len(exp_skill) > len(skill):
            specific_variations.append(exp_skill)
    
    # If found variations, return one
    if specific_variations:
        return random.choice(specific_variations)
    
    # For skills without variations, pick another skill
    return random.choice(expanded_skills)

def generate_synthetic_skills_examples(skills, count=50):
    """
    Generate synthetic examples focused on skills.
    """
    examples = []
    
    templates = [
        "Technical Skills:\n{skills}",
        "Proficient in the following technologies: {skills}",
        "Key skills include: {skills}",
        "Technologies: {skills}",
        "Programming Languages & Frameworks: {skills}",
        "Technical Expertise: {skills}",
        "Skills & Competencies: {skills}"
    ]
    
    for _ in range(count):
        # Select random skills
        selected_skills = random.sample(skills, k=random.randint(3, 10))
        
        # Choose a template
        template = random.choice(templates)
        
        # Choose a separator
        separator = random.choice([", ", " | ", "; ", "\n- ", ", and "])
        
        # Create the skills list
        skills_list = separator.join(selected_skills)
        
        # Generate text
        text = template.replace("{skills}", skills_list)
        
        # Create entities list
        entities = []
        for skill in selected_skills:
            skill_pos = text.find(skill)
            if skill_pos >= 0:
                entities.append((skill_pos, skill_pos + len(skill), "Skills"))
        
        if entities:
            examples.append((text, {"entities": entities}))
    
    return examples
def generate_boundary_edge_cases(data, factor=1):
    """
    Generate boundary edge cases for all entity types to improve model robustness.
    """
    logging.info("Generating boundary edge cases")
    augmented = []
    
    # Entity boundary patterns
    boundary_patterns = {
        "Email Address": [
            ["Contact: ", ""],
            ["Email: ", ""],
            ["", " (preferred contact)"],
            ["", " | Phone:"]
        ],
        "Phone": [
            ["Phone: ", ""],
            ["Call: ", ""],
            ["Tel: ", ""],
            ["", " (mobile)"],
            ["", " (cell)"]
        ],
        "Name": [
            ["Candidate: ", ""],
            ["Applicant: ", ""],
            ["", ", Applicant"],
            ["", " - Resume"]
        ],
        "Designation": [
            ["Role: ", ""],
            ["Position: ", ""],
            ["", " role"],
            ["Current: ", ""]
        ],
        "Companies worked at": [
            ["Company: ", ""],
            ["Employer: ", ""],
            ["", " (employer)"],
            ["", ", Inc."]
        ],
        "Degree": [
            ["Qualification: ", ""],
            ["Education: ", ""],
            ["", " (completed)"],
            ["", " with honors"]
        ],
        "College Name": [
            ["School: ", ""],
            ["University: ", ""],
            ["", " (Graduated)"],
            ["", ", accredited"]
        ],
        "Location": [
            ["Based in: ", ""],
            ["Located at: ", ""],
            ["", " area"],
            ["", " region"]
        ]
    }
    
    for _ in range(factor):
        for text, annotations in tqdm(data, desc="Creating boundary edge cases"):
            entities = annotations["entities"]
            
            # Create a new example with boundary challenges
            aug_text = text
            aug_entities = []
            offset = 0
            
            # Sort entities by position
            sorted_entities = sorted(entities, key=lambda x: x[0])
            
            for start, end, label in sorted_entities:
                entity_text = text[start:end]
                
                # Apply boundary modification 50% of the time
                if label in boundary_patterns and random.random() < 0.5:
                    # Choose a boundary pattern
                    prefix, suffix = random.choice(boundary_patterns[label])
                    
                    # Apply the pattern
                    modified_text = prefix + entity_text + suffix
                    
                    # Replace in text
                    before = aug_text[:start + offset]
                    after = aug_text[end + offset:]
                    aug_text = before + modified_text + after
                    
                    # Update entity position
                    new_start = start + offset + len(prefix)
                    new_end = new_start + len(entity_text)
                    aug_entities.append((new_start, new_end, label))
                    
                    # Update offset
                    offset += len(modified_text) - len(entity_text)
                else:
                    # Keep entity as is
                    aug_entities.append((start + offset, end + offset, label))
            
            augmented.append((aug_text, {"entities": aug_entities}))
    
    logging.info(f"Created {len(augmented)} boundary edge cases")
    return augmented
def extract_entities_of_type(data, entity_type):
    """
    Extract all instances of a specific entity type from the data.
    """
    entities = []
    for text, annotations in data:
        for start, end, label in annotations["entities"]:
            if label == entity_type:
                entities.append(text[start:end])
    return entities

def deduplicate_data(data):
    """
    Remove duplicate examples based on text content.
    """
    seen_texts = set()
    unique_data = []
    
    for text, annotations in data:
        text_hash = hash(text)
        if text_hash not in seen_texts:
            seen_texts.add(text_hash)
            unique_data.append((text, annotations))
    
    return unique_data

def count_entities_by_type(data):
    """
    Count entities by type in the data.
    """
    counts = {}
    for text, annotations in data:
        for start, end, label in annotations["entities"]:
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
    return counts

def log_augmentation_statistics(before, after):
    """
    Log statistics about the augmentation process.
    """
    all_types = set(list(before.keys()) + list(after.keys()))
    
    logging.info("Entity augmentation statistics:")
    for entity_type in sorted(all_types):
        before_count = before.get(entity_type, 0)
        after_count = after.get(entity_type, 0)
        if before_count > 0:
            increase = (after_count - before_count) / before_count * 100
            logging.info(f"  {entity_type}: {before_count} → {after_count} (+{increase:.1f}%)")
        else:
            logging.info(f"  {entity_type}: 0 → {after_count}")