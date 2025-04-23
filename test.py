import requests
import json
import os
import spacy

def test_spacy_model():
    """Test the spaCy model's entity labels directly"""
    print("\n===== TESTING SPACY MODEL DIRECTLY =====\n")
    
    model_paths = [
        "./models/resume_model"
    ]
    
    model_loaded = False
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                print(f"Loading model from {path}")
                nlp = spacy.load(path)
                model_loaded = True
                
                test_text = """
                Vikram Mehta
                Technical Lead
                vikram.mehta@accenture.com
                Hyderabad, Telangana
                
                PROFESSIONAL EXPERIENCE
                Technical Lead, Accenture (2017-Present)
                """
                
                print(f"\nProcessing text with spaCy model...")
                doc = nlp(test_text)
                
                print("\nEntities found by spaCy:")
                for ent in doc.ents:
                    print(f"  Label: '{ent.label_}' = '{ent.text}'")
                
                print("\nIMPORTANT: Make sure these labels match your entity_mapping dictionary!")
                break
            except Exception as e:
                print(f"Error loading model from {path}: {e}")
    
    if not model_loaded:
        print("⚠️ Could not load spaCy model from any path!")

def test_rasa_parse():
    """Test Rasa's entity extraction"""
    print("\n===== TESTING RASA ENTITY EXTRACTION =====\n")
    
    test_text = """
    Vikram Mehta
    Technical Lead
    vikram.mehta@accenture.com
    Hyderabad, Telangana
    
    PROFESSIONAL EXPERIENCE
    Technical Lead, Accenture (2017-Present)
    """
    
    url = "http://localhost:5005/model/parse"
    payload = {"text": test_text}
    
    try:
        response = requests.post(url, json=payload)
        result = response.json()
        
        print("Entities found by Rasa:")
        for entity in result.get("entities", []):
            print(f"  {entity.get('entity')}: '{entity.get('value')}' (extractor: {entity.get('extractor', 'unknown')})")
        
        print("\nDomain entities defined in domain.yml:")
        print("  - Location")
        print("  - Email Address")
        print("  - Designation")
        print("  - Companies worked at")
        print("  - Degree")
        print("  - Skills")
        print("  - College Name")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_spacy_model()
    test_rasa_parse()