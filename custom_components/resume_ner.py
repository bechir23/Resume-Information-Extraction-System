import os
import spacy
import re
from typing import Any, Dict, List, Text

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.nlu.extractors.extractor import EntityExtractorMixin
from rasa.shared.nlu.constants import ENTITIES
from rasa.shared.nlu.training_data.message import Message

@DefaultV1Recipe.register(
    [DefaultV1Recipe.ComponentType.ENTITY_EXTRACTOR], is_trainable=False
)
class ResumeNER(GraphComponent, EntityExtractorMixin):
    @classmethod
    def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource, execution_context: ExecutionContext) -> "ResumeNER":
        model_path = config.get("model_path", "./models/resume_model")
        print(f"[ResumeNER] Loading model from: {os.path.abspath(model_path)}")
        try:
            nlp = spacy.load(model_path)
            print(f"[ResumeNER] MODEL LOADED: {nlp.pipe_names}")
            # Test the model with sample text
            test_text = "John Smith works at Google and has a degree from MIT."
            test_doc = nlp(test_text)
            print(f"[ResumeNER] TEST: {[(ent.text, ent.label_) for ent in test_doc.ents]}")
        except Exception as e:
            print(f"[ResumeNER] ERROR LOADING MODEL: {str(e)}")
            nlp = None
        return cls(config, nlp)

    def __init__(self, config: Dict[Text, Any], nlp=None) -> None:
        self.component_config = config
        self.nlp = nlp
        
        # Map spaCy entity labels to your domain entity names EXACTLY as defined
        self.entity_mapping = {
            "PERSON": "Name",               # Maps to name slot
            "EMAIL": "Email Address",       # Maps to email slot  
            "GPE": "Location",              # Maps to location slot
            "ORG": "Companies worked at",   # Maps to companies slot
            "JOB_TITLE": "Designation",     # Maps to designation slot
            "SKILL": "Skills",              # Maps to skills slot
            "DEGREE": "Degree",             # Maps to degree slot
            "UNIVERSITY": "College Name",   # Maps to college slot
            "LOC": "Location"               # Alternative for location slot
        }
        
        # For direct mapping to slot names (lowercase)
        self.slot_mapping = {
            "name": "Name",
            "email": "Email Address",
            "location": "Location",
            "companies": "Companies worked at",
            "designation": "Designation", 
            "skills": "Skills",
            "degree": "Degree",
            "college": "College Name"
        }
        
        # Pre-compile patterns for efficiency
        self.email_pattern = re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
        self.url_email_pattern = re.compile(r'(?:https?://)?(?:www\.)?indeed\.com/r/[A-Za-z0-9-]+/[a-zA-Z0-9]+')
        
        # Expanded skills list
        self.skills_list = [
            "Python", "Java", "JavaScript", "C++", "C#", "Ruby", "PHP", "Swift", "TypeScript",
            "Kotlin", "Go", "Rust", "COBOL", "JCL", "Mainframe", "SQL", "NoSQL", "MongoDB",
            "MySQL", "PostgreSQL", "Oracle", "HTML", "CSS", "React", "Angular", "Vue", "Node.js",
            "Express", "Django", "Flask", "Spring", "Hibernate", "AWS", "Azure", "GCP", "Docker",
            "Kubernetes", "Jenkins", "Git", "SVN", "Jira", "Confluence", "Agile", "Scrum", "DevOps",
            "CI/CD", "TDD", "REST", "API", "GraphQL", "Machine Learning", "AI", "Data Science",
            "Data Analysis", "Big Data", "Hadoop", "Spark", "ETL", "Power BI", "Tableau"
        ]
        # Case-insensitive pattern
        self.skills_pattern = re.compile(r'\b(' + '|'.join(re.escape(skill) for skill in self.skills_list) + r')\b', re.IGNORECASE)
        
        # Common degrees
        self.degrees = [
            "B.Tech", "B-TECH", "B.E", "Bachelor of Engineering", "Bachelor of Technology",
            "M.Tech", "M-TECH", "M.E", "Master of Engineering", "Master of Technology",
            "PhD", "Ph.D", "MBA", "BBA", "BCA", "MCA", "B.Sc", "M.Sc", "B.Com", "M.Com"
        ]
        
    def process(self, messages: List[Message]) -> List[Message]:
        for message in messages:
            self.process_message(message)
        return messages

    def process_message(self, message: Message) -> Message:
        text = message.get("text")
        print(f"[ResumeNER] Processing: {text[:50]}...")
        
        # Get existing entities to replace
        diet_entities = message.get(ENTITIES, [])
        print(f"[ResumeNER] Found {len(diet_entities)} entities from DIET")
        
        our_entities = []
        
        # Process with spaCy if available
        if self.nlp:
            doc = self.nlp(text)
            
            # Add entities from spaCy
            for ent in doc.ents:
                if ent.label_ in self.entity_mapping:
                    entity_name = self.entity_mapping[ent.label_]
                    print(f"[ResumeNER] Found: {entity_name} = '{ent.text}'")
                    our_entities.append({
                        "entity": entity_name,  # Exact entity name from domain
                        "value": ent.text,
                        "start": ent.start_char,
                        "end": ent.end_char,
                        "confidence": 0.99,
                        "extractor": "ResumeNER"
                    })
                else:
                    print(f"[ResumeNER] Unknown entity type: {ent.label_}")
        
        # Add regex-based extraction for common patterns
        
        # 1. Name extraction - look at the beginning of resume or after "Name:" patterns
        name_patterns = [
            r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})\s',  # First line name pattern
            r'(?:Name|NAME):\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2})',  # Name: John Doe
            r'^([A-Z][A-Z\s]+)$',  # ALL CAPS name on its own line
        ]
        
        if not any(e["entity"] == "Name" for e in our_entities):
            for pattern in name_patterns:
                matches = re.search(pattern, text, re.MULTILINE)
                if matches:
                    name = matches.group(1).strip()
                    print(f"[ResumeNER] Regex found: Name = '{name}'")
                    our_entities.append({
                        "entity": "Name",
                        "value": name,
                        "start": matches.start(1),
                        "end": matches.end(1),
                        "confidence": 0.95,
                        "extractor": "ResumeNER"
                    })
                    break  # Once we have a name, stop looking
        
        # 2. Email Address - standard email and indeed URLs
        if not any(e["entity"] == "Email Address" for e in our_entities):
            # Standard email pattern
            for match in self.email_pattern.finditer(text):
                print(f"[ResumeNER] Regex found: Email Address = '{match.group()}'")
                our_entities.append({
                    "entity": "Email Address", 
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "confidence": 0.98,
                    "extractor": "ResumeNER"
                })
            
            # Indeed URL pattern
            for match in self.url_email_pattern.finditer(text):
                print(f"[ResumeNER] Regex found: Email Address (URL) = '{match.group()}'")
                our_entities.append({
                    "entity": "Email Address",
                    "value": match.group(),
                    "start": match.start(),
                    "end": match.end(), 
                    "confidence": 0.97,
                    "extractor": "ResumeNER"
                })
        
        # 3. Location - improved patterns
        location_patterns = [
            r'(?:Location|Address|City|Town|Place):\s*([A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+)*)',  # Location: Hyderabad
            r'(?:in|at|from)\s+([A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+)*)',  # in Hyderabad, Telangana
            r'([A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+)*),\s*(?:India|USA|UK|Canada)',  # Hyderabad, India
            r'(?<=\n)([A-Z][a-z]+(?:[\s,]+[A-Z][a-z]+)*?)(?=\s*\n)',  # Location on separate line
        ]
        
        if not any(e["entity"] == "Location" for e in our_entities):
            for pattern in location_patterns:
                for match in re.finditer(pattern, text, re.MULTILINE):
                    # Extract captured group if it exists, otherwise use the whole match
                    location = match.group(1) if len(match.groups()) > 0 else match.group()
                    print(f"[ResumeNER] Regex found: Location = '{location}'")
                    our_entities.append({
                        "entity": "Location",  # Exact domain entity name
                        "value": location.strip(),
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.92,
                        "extractor": "ResumeNER"
                    })
        
        # 4. Companies worked at - improved patterns
        company_patterns = [
            r'(?:Company|Employer|Workplace|Work):\s*([A-Z][a-zA-Z\s]+)',  # Company: SAP
            r'(?:at|with|for)\s+([A-Z][a-zA-Z\s]+)(?:\s+(?:as|for|from))',  # at Accenture as
            r'(?<=\n)([A-Z][a-zA-Z\s]+)(?:\s+\(\d{4})',  # Accenture (2017
            r'(?<=\n)([A-Z][A-Za-z\s\.]+)(?:\s*(?:-|–|,))',  # Company at the start of a line
            r'(?:experience at|worked at|work at|employed at|with)\s+([A-Z][a-zA-Z\s]+)', # experience at Google
        ]
        
        for pattern in company_patterns:
            for match in re.finditer(pattern, text):
                # Extract captured group if it exists
                company = match.group(1) if len(match.groups()) > 0 else match.group()
                company = company.strip()
                
                if company.lower() not in ["in", "at", "with", "company", "employer", "workplace"]:
                    print(f"[ResumeNER] Regex found: Companies worked at = '{company}'")
                    our_entities.append({
                        "entity": "Companies worked at",  # Exact domain entity name
                        "value": company,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.93,
                        "extractor": "ResumeNER"
                    })
        
        # 5. Skills - using expanded skills list
        for match in self.skills_pattern.finditer(text):
            skill = match.group()
            # Find the correctly capitalized version from our list
            for original_skill in self.skills_list:
                if original_skill.lower() == skill.lower():
                    skill = original_skill
                    break
                
            print(f"[ResumeNER] Regex found: Skills = '{skill}'")
            our_entities.append({
                "entity": "Skills",  # Exact domain entity name
                "value": skill,
                "start": match.start(),
                "end": match.end(),
                "confidence": 0.95,
                "extractor": "ResumeNER"
            })
            
        # 6. Designation / Job Title
        designation_patterns = [
            r'(?:Designation|Position|Title|Role|Job):\s*([A-Za-z\s]+)',  # Designation: Software Engineer
            r'(?<=\n)([A-Z][a-z]+\s+(?:Engineer|Developer|Consultant|Architect|Manager|Director|Analyst|Designer|Administrator))(?=\s|\n)',  # Job title on its own line
            r'(?:as a|as an|as|worked as)\s+([A-Z][a-z]+\s+(?:Engineer|Developer|Consultant|Architect|Manager|Director|Analyst|Designer|Administrator))', # Worked as Senior Engineer
            r'(?<=\n)([A-Z][A-Z\s]+)(?=\n)',  # ALL CAPS job title on its own line
        ]
        
        if not any(e["entity"] == "Designation" for e in our_entities):
            for pattern in designation_patterns:
                for match in re.finditer(pattern, text, re.MULTILINE):
                    designation = match.group(1) if len(match.groups()) > 0 else match.group()
                    designation = designation.strip()
                    
                    if designation.lower() not in ["designation", "position", "title", "role"]:
                        print(f"[ResumeNER] Regex found: Designation = '{designation}'")
                        our_entities.append({
                            "entity": "Designation",
                            "value": designation,
                            "start": match.start(),
                            "end": match.end(),
                            "confidence": 0.94,
                            "extractor": "ResumeNER"
                        })
                        
        # 7. Degree
        degree_patterns = [
            r'(?:Degree|Education|Qualification):\s*([A-Za-z\.\-\s]+)',  # Degree: B.Tech
            r'\b(' + '|'.join(re.escape(degree) for degree in self.degrees) + r')\b',  # Match from degree list
            r'(?:Bachelor|Master|Bachelors|Masters|PhD|Ph\.D)\s+(?:of|in)\s+([A-Za-z\s]+)', # Bachelor of Engineering
        ]
        
        if not any(e["entity"] == "Degree" for e in our_entities):
            for pattern in degree_patterns:
                for match in re.finditer(pattern, text):
                    degree = match.group(1) if len(match.groups()) > 0 else match.group()
                    degree = degree.strip()
                    
                    print(f"[ResumeNER] Regex found: Degree = '{degree}'")
                    our_entities.append({
                        "entity": "Degree",
                        "value": degree,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.94,
                        "extractor": "ResumeNER"
                    })
                    
        # 8. College Name
        college_patterns = [
            r'(?:College|University|Institute|School):\s*([A-Za-z\s]+)',  # College: MIT
            r'(?:from|at)\s+([A-Z][a-zA-Z\s]+(?:University|College|Institute|School))', # from Stanford University
            r'(?<=\n)([A-Z][a-zA-Z\s]+(?:University|College|Institute|School))(?=\s|\n)', # College on its own line
            r'(?<=\n)([A-Z][A-Z\s]+(?:UNIVERSITY|COLLEGE|INSTITUTE|SCHOOL))(?=\s|\n)', # ALL CAPS college
        ]
        
        if not any(e["entity"] == "College Name" for e in our_entities):
            for pattern in college_patterns:
                for match in re.finditer(pattern, text, re.MULTILINE):
                    college = match.group(1) if len(match.groups()) > 0 else match.group()
                    college = college.strip()
                    
                    if college.lower() not in ["college", "university", "institute", "school"]:
                        print(f"[ResumeNER] Regex found: College Name = '{college}'")
                        our_entities.append({
                            "entity": "College Name",
                            "value": college,
                            "start": match.start(),
                            "end": match.end(),
                            "confidence": 0.93,
                            "extractor": "ResumeNER"
                        })
                        
        # 9. Direct entity mentions from domain slots (e.g., "My name is...")
        for slot_name, entity_name in self.slot_mapping.items():
            patterns = [
                rf'(?:my|the)\s+{slot_name}\s+(?:is|are|:)\s+([\w\s,@\.-]+)',  # my location is Chennai
                rf'{slot_name}[:\s]\s*([\w\s,@\.-]+)',  # location: Chennai
            ]
            
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    value = match.group(1).strip()
                    if value and len(value) > 1:  # Avoid single chars
                        print(f"[ResumeNER] Direct mention found: {entity_name} = '{value}'")
                        our_entities.append({
                            "entity": entity_name,
                            "value": value,
                            "start": match.start(1),
                            "end": match.end(1),
                            "confidence": 0.97,
                            "extractor": "ResumeNER"
                        })
        
        # Custom extraction for the format: "Designation: value"
        labeled_entity_pattern = r'(\w+):\s*([\w\s,@\.-]+)'
        for match in re.finditer(labeled_entity_pattern, text):
            label = match.group(1).strip().lower()
            value = match.group(2).strip()
            
            # Map common labels to entity names
            label_map = {
                "name": "Name",
                "email": "Email Address",
                "location": "Location",
                "company": "Companies worked at",
                "companies": "Companies worked at",
                "designation": "Designation",
                "position": "Designation", 
                "title": "Designation",
                "skills": "Skills",
                "skill": "Skills",
                "degree": "Degree",
                "education": "Degree",
                "college": "College Name",
                "university": "College Name"
            }
            
            if label in label_map and value:
                entity_name = label_map[label]
                print(f"[ResumeNER] Labeled entity found: {entity_name} = '{value}'")
                our_entities.append({
                    "entity": entity_name,
                    "value": value,
                    "start": match.start(2),
                    "end": match.end(2),
                    "confidence": 0.96,
                    "extractor": "ResumeNER"
                })
        
        # Remove duplicates based on entity type and value
        unique_entities = {}
        for entity in our_entities:
            key = (entity["entity"], entity["value"])
            if key not in unique_entities or entity["confidence"] > unique_entities[key]["confidence"]:
                unique_entities[key] = entity
        
        our_entities = list(unique_entities.values())
        
        # Override all entities with ours
        if our_entities or diet_entities:
            print(f"[ResumeNER] Replacing {len(diet_entities)} DIET entities with {len(our_entities)} ResumeNER entities")
            message.set(ENTITIES, our_entities)  # Replace with our entities
        else:
            print("[ResumeNER] No entities found")
        
        return message