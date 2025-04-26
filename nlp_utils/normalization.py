# nlp_utils/normalization.py
# This module contains normalization and cleaning functions for NER entities.
import re

def normalize_email_address(text):
    """
    Normalize email addresses and Indeed URLs with comprehensive pattern handling.
    """
    if not text:
        return ""
        
    # Remove line breaks and extra spacing
    text = text.replace('\n', ' ').strip()
    
    # Check for actual email addresses first
    email_match = re.search(r'[\w.+-]+@[\w-]+\.[a-zA-Z0-9.-]{2,}', text)
    if email_match:
        return email_match.group(0)
    
    # Strip any leading "Indeed:" prefix
    text = re.sub(r'^Indeed:\s*', '', text)
    
    # Handle complete Indeed URLs with parameters
    full_url_match = re.search(r'https?://(?:www\.)?indeed\.com/r/[\w-]+/[a-zA-Z0-9]+(?:\?[^"\s]+)?', text)
    if full_url_match:
        # Extract just the core path without parameters
        url_core = re.search(r'(indeed\.com/r/[\w-]+/[a-zA-Z0-9]+)', full_url_match.group(0))
        if url_core:
            return f"https://www.{url_core.group(1)}"
    
    # Handle shortened Indeed URLs, preserving hyphens in names
    indeed_match = re.search(r'(?:^|[^\w])(?:https?://(?:www\.)?)?indeed\.com/r/[\w-]+/[a-zA-Z0-9]+', text)
    if indeed_match:
        url = indeed_match.group(0).strip()
        # Remove any leading non-URL character
        url = re.sub(r'^[^\w]+', '', url)
        # Add prefix if missing
        if not url.startswith('http'):
            url = f"https://www.{url}"
        return url
    
    # Check for LinkedIn URLs
    linkedin_match = re.search(r'https?://(?:www\.)?linkedin\.com/in/[\w-]+(?:/[a-zA-Z0-9-]+)?', text)
    if linkedin_match:
        return linkedin_match.group(0)
    
    # Filter out non-URLs and irrelevant text
    if text.strip() in ["months."]:
        return ""
        
    return text.strip()

def normalize_graduation_year(text):
    """
    Extract graduation year from text with comprehensive pattern recognition.
    """
    if not text or text.strip().upper() in ["EDUCATION"]:
        return ""
    
    # Handle complete date patterns first
    month_year = re.search(r'(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+(19|20\d{2})', text, re.I)
    if month_year:
        return month_year.group(1)
    
    # Extract standalone years 
    year_match = re.search(r'(19|20)\d{2}', text)
    
    # Handle special formats
    class_match = re.search(r'class\s+of\s+((?:19|20)\d{2})', text.lower())
    batch_match = re.search(r'batch\s+(?:of\s+)?((?:19|20)\d{2})', text.lower())
    grad_match = re.search(r'graduat(?:ed|ion)(?:\s+in|\s+on)?\s+(?:19|20\d{2})', text.lower())
    date_match = re.search(r'\d{1,2}[/-]((?:19|20)\d{2})', text)
    
    # Return the first valid match
    if class_match:
        return class_match.group(1)
    elif batch_match:
        return batch_match.group(1)
    elif grad_match:
        return re.search(r'(19|20\d{2})', grad_match.group(0)).group(0)
    elif date_match:
        return date_match.group(1)
    elif year_match and 1980 <= int(year_match.group(0)) <= 2025:
        return year_match.group(0)
    
    return ""



def normalize_location(text):
    """
    Normalize location entities with comprehensive pattern recognition.
    """
    # Skip URLs, Indeed references, and empty strings
    if not text or re.search(r'(?:https?://|www\.|@|indeed\.com)', text):
        return ""
    
    # Remove prefixes and non-location elements
    text = re.sub(r'^(?:Indeed:|in|at|located\s+in|from)\s*', '', text)
    text = re.sub(r'(?:Office:|IT\s+Pvt|Institute$|Education$|British\s+Telecom|Technology\s+Lead)', '', text)
    
    # Clean up special characters and standardize spacing
    text = re.sub(r'[\n-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # City name standardization with Indian locations
    city_mapping = {
        'bangalore': 'Bengaluru',
        'bombay': 'Mumbai',
        'calcutta': 'Kolkata',
        'madras': 'Chennai',
        'poona': 'Pune',
        'delhi': 'New Delhi',
        'secunderabad': 'Hyderabad',  
        'chnadigarh': 'Chandigarh',   
        'hyderbad': 'Hyderabad',      
        'bhubaneshwar': 'Bhubaneswar',
        'orrisha': 'Odisha',          
        'trichur': 'Thrissur',        
        'gurgaon': 'Gurugram',
        'mangalore': 'Mangaluru',
        'mysore': 'Mysuru',
        'tiruchchirappalli': 'Tiruchirappalli',
        'bangalore urban': 'Bengaluru'
    }
    
    # Standardize the location if it matches a key in city_mapping
    text_lower = text.lower().strip()
    if text_lower in city_mapping:
        return city_mapping[text_lower]
    
    # Extract city name if followed by state or country
    if ',' in text:
        city = text.split(',')[0].strip()
        if city.lower() in city_mapping:
            return city_mapping[city.lower()]
        return city
    
    # Skip very short or likely non-location entries
    if len(text.strip()) < 3 or text.strip() in ["ru", "-", "Gokul"]:
        return ""
        
    return text.strip()

def normalize_company(text):
    """
    Normalize company names with comprehensive pattern handling.
    """
    # Skip non-company entries and URLs
    if not text or text.lower() in ['awards', 'architecture'] or re.search(r'(?:https?://|www\.linkedin\.com)', text):
        return ""
    
    # Remove role indicators and prefixes
    text = re.sub(r'(?:Client:|Executive\s+at|successfully|Azure\s+Client|-)', '', text)
    
    # Remove parenthetical information and role-related keywords
    text = re.sub(r'\([^)]*\)', '', text)
    text = re.sub(r'\b(?:Lead,|Lead|Manager|Engineer|Consultant|Architect|HR|Intern|Analyst|Developer|Executive|Sr|Jr)\b', '', text, flags=re.I)
    
    # Normalize whitespace
    text = re.sub(r'[-–|@,:;•]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # SAP product standardization
    if re.search(r'\bSAP\s+(?:ABAP|EWM|Basis)\b', text, re.I):
        return "SAP"
    
    # Company name standardization map
    company_mapping = {
        'infosys': 'Infosys',
        'infosys limited': 'Infosys',
        'infosys ltd': 'Infosys',
        'INFOSYS': 'Infosys',
        'INFOSYS LIMITED': 'Infosys',
        'oracle': 'Oracle',
        'microsoft': 'Microsoft',
        'microsoft india': 'Microsoft India',
        'teleperformance for microsoft': 'Teleperformance',
        'tcs': 'TCS',
        'ibm': 'IBM',
        'wipro': 'Wipro',
        'accenture': 'Accenture',
        'amazon': 'Amazon',
        'google': 'Google',
        'cisco': 'Cisco',
        'sap': 'SAP',
        'sap labs': 'SAP Labs',
        'convergys': 'Convergys'
    }
    
    # Try case-insensitive match
    text_lower = text.lower()
    if text_lower in company_mapping:
        return company_mapping[text_lower]
    
    # Try partial match with word boundaries
    for key, value in company_mapping.items():
        if re.search(r'\b' + re.escape(key) + r'\b', text_lower):
            return value
    
    # Return empty for very short entries
    if len(text.strip()) <= 2:
        return ""
    
    return text.strip()

def normalize_skills(text):
    """
    Normalize skill entries with comprehensive pattern recognition.
    """
    # Skip section headers and non-skill text
    if text.strip().upper() in ["SKILLS", "SKILLS:", "SKILL SETS", "COMPUTER LANGUAGES KNOWN:"]:
        return ""
    
    # Extract experience info in parentheses (preserve it for output)
    exp_pattern = r'\s*\(([^)]*(?:year|month|yr)[^)]*)\)'
    exp_match = re.search(exp_pattern, text, re.I)
    
    if exp_match:
        experience_info = exp_match.group(1)
        text = re.sub(exp_pattern, '', text)
    
    # Remove formatting elements and clean up text
    text = re.sub(r'^[•\-\*\d]+\.?\s*', '', text)
    text = re.sub(r'^\s*[–•:]\s*', '', text)
    text = re.sub(r'[.;:]$', '', text.strip())
    
    # Skip non-skill entries
    if text.lower().strip() in ["teaching", "polysaccarides'"]:
        return ""
    
    # Technology capitalization map
    skill_mapping = {
        'javascript': 'JavaScript',
        'java': 'Java',
        'python': 'Python',
        'c++': 'C++',
        'angular js': 'AngularJS',
        'html': 'HTML',
        'css': 'CSS',
        'aws': 'AWS',
        'sql': 'SQL',
        'docker': 'Docker',
        'git': 'Git',
        'ms office': 'Microsoft Office',
        'microsoft office': 'Microsoft Office',
        'excel': 'Excel',
        'end user computing': 'End User Computing',
        'active directory': 'Active Directory',
        'tally': 'Tally',
        'velocity': 'Velocity'
    }
    
    # Apply skill-specific capitalization
    for skill, proper_form in skill_mapping.items():
        text = re.sub(r'\b' + re.escape(skill) + r'\b', proper_form, text, flags=re.I)
    
    # Reattach experience info if present
    if exp_match:
        text = f"{text} ({experience_info})"
    
    return text.strip()

def normalize_degree(text):
    """
    Normalize degree information to standard forms.
    """
    # Skip section headers and invalid entries
    if re.search(r'^(?:EDUCATION|PROFFESIONAL CERTIFICATION)$', text, re.I) or re.search(r'^(?:19|20)\d{2}$', text):
        return ""
    
    # Skip non-degree entries
    if text.strip() in ["Engineering", "December 2014"]:
        return ""
    
    # Degree abbreviation normalization mapping
    degree_mapping = {
        r'^b[\.\s-]*tech': "Bachelor of Technology",
        r'^b[\.\s-]*e[\.\s]*': "Bachelor of Engineering",
        r'^b[\.\s-]*sc': "Bachelor of Science", 
        r'c\.b\.s\.e': "CBSE"
    }
    
    # Try each pattern match
    for pattern, replacement in degree_mapping.items():
        if re.search(pattern, text.lower(), re.I):
            # Check for specialization
            spec_match = re.search(r'in\s+([\w\s&]+)', text)
            if spec_match and replacement != "CBSE":
                specialization = spec_match.group(1).strip()
                if specialization == "CSE":
                    return f"{replacement} in Computer Science Engineering"
                elif specialization == "ECE":
                    return f"{replacement} in Electronics and Communication Engineering"
                else:
                    return f"{replacement} in {specialization.title()}"
            return replacement
    
    # Keep the original text for complex degrees
    if "Bachelor" in text or "Master" in text:
        return text
        
    return text.strip()

def normalize_college(text):
    """
    Normalize college/university names with comprehensive pattern handling.
    """
    # Skip invalid entries
    if not text or text.strip() in ["2011", "Engineering", "Anna", "B.SC."]:
        return ""
    
    # Remove extraneous information and clean up
    text = re.sub(r'\s*\([^)]*\)\s*', ' ', text)
    text = re.sub(r'\s*,[^,]*$', '', text)
    text = re.sub(r'\b(?:19|20)\d{2}\b', '', text)
    
    # Institution mapping for abbreviations and standardization
    college_mapping = {
        'bput': 'Biju Patnaik University of Technology',
        'psg college of technology': 'PSG College of Technology',
        'magadh university': 'Magadh University',
        'vssut': 'Veer Surendra Sai University of Technology',
        'uptu': 'Uttar Pradesh Technical University',
        'dr mgr university': 'Dr. M.G.R. Educational and Research Institute',
        'manipal university': 'Manipal University'
    }
    
    # Apply mappings for known institutions
    text_lower = text.lower().strip()
    for abbr_lower, full_name in college_mapping.items():
        if text_lower == abbr_lower or re.search(r'\b' + re.escape(abbr_lower) + r'\b', text_lower):
            return full_name
    
    return text.strip()

def normalize_designation(text):
    """
    Normalize job designations with comprehensive pattern handling.
    """
    # Skip section headers and non-designations
    if text.strip().upper() in ["EXPERIENCE"] or text.strip() in ["◦ Technical", ""]:
        return ""
    
    # Clean up designations
    text = re.sub(r'^[◦\-:•]+\s*', '', text)
    text = re.sub(r'^(?:Technical|Offshore Technical|:)', '', text)
    text = re.sub(r'[\"\']', '', text)
    
    # Normalize job titles
    title_mapping = {
        'sr': 'Senior',
        'technical architect': 'Technical Architect', 
        'technical architecture': 'Architecture',
        'store executive': 'Store Executive',
        'security analyst': 'Security Analyst',
        'volunteer contestant': 'Volunteer Contestant'
    }
    
    # Apply title mappings
    for title, normalized in title_mapping.items():
        text = re.sub(r'\b' + re.escape(title) + r'\b', normalized, text, flags=re.I)
    
    # Remove company information after comma
    if ',' in text:
        text = text.split(',')[0].strip()
    
    return text.strip()

def normalize_name(text):
    """Normalize person names."""
    # Remove salutations
    text = re.sub(r'^(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?)\s+', '', text, flags=re.I)
    
    # Remove credentials
    text = re.sub(r',\s*(?:PhD|MD|MBA|CPA|PE|PMP|CISSP|CFA).*$', '', text, flags=re.I)
    
    # Apply title case (capitalize first letter of each word)
    text = " ".join(word.capitalize() for word in text.split())
    
    return text.strip()

def normalize_experience_years(text):
    """
    Normalize years of experience with comprehensive pattern recognition.
    """
    # Extract numeric years with "years" keyword
    year_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:year|yr)s?', text.lower())
    if year_match:
        years = float(year_match.group(1))
        return f"{years:.1f} years".replace(".0 ", " ")
    
    # Extract months as fraction of years
    month_match = re.search(r'(\d+)\s*month', text.lower())
    if month_match:
        months = int(month_match.group(1))
        years = round(months / 12, 1)
        return f"{years:.1f} years".replace(".0 ", " ")
    
    # Extract from "Less than X year" patterns
    less_than_match = re.search(r'less\s+than\s+(\d+)(?:\.\d+)?\s*(?:year|yr)', text.lower())
    if less_than_match:
        return f"<{less_than_match.group(1)} years"
    
    # Handle specific cases like "7 years" at the start of text
    exact_years_match = re.match(r'\(?\s*(\d+)\s*\)?$', text.strip())
    if exact_years_match:
        return f"{exact_years_match.group(1)} years"
    
    return text.strip()

def test_normalizers():
    """
    Test all normalization functions with representative data.
    """
    # Set up test data
    tests = {
        "Email Address": [
            "user@example.com",
            "indeed.com/r/Shaheen-Unissa/c54e7a04da30c354",
            "https://www.indeed.com/r/Dilliraja-Baskaran/4a3bc8a35879ce5c?isid=rex-download&ikw=download-top&co=IN",
            "indeed.com/r/Raja-Chandra-\nMouli/445cbf3eb0a361cd",
            "Indeed: indeed.com/r/Jay-Madhavi/1e7d0305af766bf6",
            "months.",
            "ndeed.com/r/Mohamed-Ameen/ba052bfa70e4c0b7"
        ],
        "Graduation Year": [
            "2011",
            "Class of 2015",
            "Batch of 2018",
            "Graduated in May 2017",
            "12/2014",
            "December 2014",
            "EDUCATION"
        ],
        "Location": [
            "Bengaluru", 
            "indeed.com/r/Darshan-G/025a61a82c6a8c5a", 
            "hyderbad", 
            "Bangalore", 
            "ru",
            "Chennai",
            "Gurgaon"
        ],
        "Company": [
            "Microsoft", 
            "Client: Microsoft", 
            "INFOSYS LIMITED", 
            "SAP ABAP", 
            "Lead, - Infosys",
            "Executive at",
            "Client -Microsoft",
            "SAP EWM",
            "SAP Basis",
            "Convergys"
        ],
        "Skills": [
            "java (Less than 1 year", 
            "End user computing (3 years", 
            "JAVASCRIPT", 
            "Angular JS", 
            "Velocity",
            "9 Computer Languages Known:", 
            "• C programming basics",
            "Teaching",
            "SKILLS:"
        ],
        "Degree": [
            "B-TECH", 
            "B.E in CSE", 
            "EDUCATION", 
            "Bachelor of Technology in ECE", 
            "December 2014", 
            "B.SC.",
            "C.B.S.E."
        ],
        "College": [
            "Manipal University", 
            "2011", 
            "BPUT", 
            "Anna", 
            "REiume Institute",
            "PSG COLLEGE OF TECHNOLOGY",
            "VSSUT"
        ],
        "Designation": [
            "DevOps Consultant", 
            "Technical Architect & Sr. Software Engineer", 
            "◦ Technical", 
            "Store Executive", 
            "EXPERIENCE", 
            "Technical Architecture",
            "Security Analyst"
        ]
    }

    # Process and print results
    for entity_type, examples in tests.items():
        print(f"\n=== {entity_type.upper()} NORMALIZATION TESTS ===")
        for example in examples:
            if entity_type == "Email Address":
                result = normalize_email_address(example)
            elif entity_type == "Graduation Year":
                result = normalize_graduation_year(example)
            elif entity_type == "Location":
                result = normalize_location(example)
            elif entity_type == "Company":
                result = normalize_company(example)
            elif entity_type == "Skills":
                result = normalize_skills(example)
            elif entity_type == "Degree":
                result = normalize_degree(example)
            elif entity_type == "College":
                result = normalize_college(example)
            elif entity_type == "Designation":
                result = normalize_designation(example)
            print(f"{example} -> {result}")



def clean_entity_by_type(text, start, end, label, log_file=None, debug_mode=False):
    """
    Apply entity-specific cleaning and normalization rules.
    Returns cleaned text and adjusted start/end positions.
    """
    original_text = text[start:end]
    cleaned_text = original_text.strip()
    
    # Track position adjustments from stripping
    start_offset = original_text.find(cleaned_text)
    end_offset = start_offset + len(cleaned_text)
    
    new_start = start + start_offset
    new_end = start + end_offset
    
    # Skip extremely long entities (likely parsing errors)
    if len(cleaned_text) > 150 and label != "Email Address":
        if debug_mode and log_file:
            log_file.write(f"  REJECTED: '{cleaned_text[:30]}...' - Entity too long ({len(cleaned_text)} chars)\n")
        return "", 0, 0
    
    # Apply entity-specific normalization
    if label == "Email Address":
        cleaned_text = normalize_email_address(cleaned_text)
        
    elif label == "Name":
        cleaned_text = normalize_name(cleaned_text)
        
    elif label == "Degree":
        cleaned_text = normalize_degree(cleaned_text)
        
    elif label == "Graduation Year":
        cleaned_text = normalize_graduation_year(cleaned_text)
        
    elif label == "Years of Experience":
        cleaned_text = normalize_experience_years(cleaned_text)
        
    elif label == "Skills":
        cleaned_text = normalize_skills(cleaned_text)
        
    elif label == "Designation":
        cleaned_text = normalize_designation(cleaned_text)
        
    elif label == "Companies worked at":
        cleaned_text = normalize_company(cleaned_text)
        
    elif label == "College Name":
        cleaned_text = normalize_college(cleaned_text)
        
    elif label == "Location":
        cleaned_text = normalize_location(cleaned_text)
    
    # If normalization emptied the entity or it's only whitespace
    if not cleaned_text or not cleaned_text.strip():
        if debug_mode and log_file:
            log_file.write(f"  REJECTED: '{original_text}' - Empty after cleaning\n")
        return "", 0, 0
        
    # Log changes if debugging
    if debug_mode and log_file and cleaned_text != original_text.strip():
        log_file.write(f"  NORMALIZED {label}: '{original_text}' -> '{cleaned_text}'\n")
    
    # Calculate new boundaries based on original text
    if cleaned_text != original_text and cleaned_text in text[start:end]:
        try:
            # Try to find the normalized text within the original span
            match_pos = text[start:end].find(cleaned_text)
            if match_pos >= 0:
                new_start = start + match_pos
                new_end = new_start + len(cleaned_text)
        except:
            # If there's an error (e.g., with regex), keep the trimmed boundaries
            pass
    
    return cleaned_text, new_start, new_end
def entity_priority(entity_type):
    """
    Define priority order for entities to handle overlaps.
    Higher number = higher priority
    """
    priority_map = {
        "Name": 10,
        "Email Address": 9,
        "Designation": 8,
        "Companies worked at": 7,
        "Skills": 6,
        "College Name": 5,
        "Degree": 4,
        "Graduation Year": 3,
        "Years of Experience": 2,
        "Location": 1
    }
    return priority_map.get(entity_type, 0)

def handle_token_overlaps(entities, text, debug_mode=False, log_file=None):
    """
    Handle token-level overlaps by keeping higher priority entities.
    Entities should be tuples of (start_char, end_char, label, token_start, token_end)
    """
    if not entities:
        return []
    
    # Sort by priority (higher priority first) then by length (longer first)
    entities = sorted(entities, 
                     key=lambda x: (entity_priority(x[2]), x[1] - x[0]), 
                     reverse=True)
    
    # Token-based overlap resolution
    final_entities = []
    taken_tokens = set()
    
    for start_char, end_char, label, token_start, token_end in entities:
        # Get the span's tokens
        span_tokens = set(range(token_start, token_end))
        
        # Check if this span overlaps with any already accepted span
        if span_tokens.intersection(taken_tokens):
            if debug_mode and log_file:
                log_file.write(f"  OVERLAP REMOVED: '{text[start_char:end_char]}' Label: {label}\n")
            continue
        
        # No overlap, so keep this entity
        final_entities.append((start_char, end_char, label, token_start, token_end))
        taken_tokens.update(span_tokens)
    
    return final_entities


def split_skills(text, annotations, log_file=None, debug_mode=False):
    """
    Split 'Skills' entities into individual skill entities.
    This creates multiple separate "Skills" entities from comma/semicolon separated lists.
    """
    new_annotations = {"entities": []}
    
    for start, end, label in annotations["entities"]:
        # Only process Skills entities
        if label != "Skills":
            new_annotations["entities"].append((start, end, label))
            continue
        
        skill_text = text[start:end]
        
        # Skip very short skills (likely not a list)
        if len(skill_text) < 10:
            new_annotations["entities"].append((start, end, label))
            continue
            
        # Check for common list separators
        if "," in skill_text or ";" in skill_text:
            # First normalize the separators
            normalized = skill_text.replace(";", ",")
            
            # Split by commas, but handle special cases
            # We don't want to split "C++", "C#", etc.
            skill_parts = []
            current_part = ""
            
            # Basic splitting - we'll refine this next
            raw_parts = normalized.split(",")
            
            # Process each part, checking for programming languages that
            # shouldn't be split (like C++)
            for i, part in enumerate(raw_parts):
                part = part.strip()
                if not part:
                    continue
                    
                # Check if this part looks like it belongs to previous part
                if i > 0 and re.search(r'^(?:\+\+|\#|\.NET)', part):
                    # This is likely part of a language name, append to previous
                    skill_parts[-1] = f"{skill_parts[-1]},{part}"
                else:
                    skill_parts.append(part)
            
            # Now process individual skills
            current_pos = start
            for skill in skill_parts:
                skill = skill.strip()
                if not skill:
                    continue
                    
                # Find this skill within the original text
                skill_pos = text[current_pos:end].find(skill)
                if skill_pos >= 0:
                    skill_start = current_pos + skill_pos
                    skill_end = skill_start + len(skill)
                    
                    # Normalize the individual skill
                    normalized_skill = normalize_skills(skill)
                    
                    if normalized_skill:
                        new_annotations["entities"].append((skill_start, skill_end, "Skills"))
                    
                    # Update position for next search
                    current_pos = skill_end
                    
            if debug_mode and log_file and len(skill_parts) > 1:
                log_file.write(f"  SKILL LIST SPLIT: '{skill_text}' -> {len(skill_parts)} skills\n")
        else:
            # Not a list, keep as a single entity
            new_annotations["entities"].append((start, end, label))
    
    return new_annotations