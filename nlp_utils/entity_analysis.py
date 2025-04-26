# nlp_utils/entity_analysis.py
# This module contains functions for analyzing and visualizing NER entity patterns.
import collections
import re
import json
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.util import ngrams
import re
import json
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.util import ngrams

def analyze_entities(data, entity_type):
    """Extract and analyze entities of a specific type from data"""
    entities = []
    contexts = []
    
    for text, annotations in data:
        for start, end, label in annotations['entities']:
            if label == entity_type:
                entity_text = text[start:end]
                
                # Get context (words before and after)
                pre_context = text[max(0, start-50):start].strip()
                post_context = text[end:min(end+50, len(text))].strip()
                
                entities.append(entity_text)
                contexts.append({
                    "entity": entity_text,
                    "pre": pre_context,
                    "post": post_context
                })
    
    return {
        "entities": entities,
        "contexts": contexts
    }

def entity_pattern_analysis(entity_data):
    """Generate comprehensive pattern analysis for entity texts"""
    entities = entity_data["entities"]
    
    if not entities:
        return {"count": 0, "message": "No entities found"}
    
    # Basic statistics
    counts = Counter(entities)
    
    # Length analysis
    lengths = [len(e) for e in entities]
    
    # Character distribution
    all_chars = "".join(entities)
    char_counts = Counter(all_chars)
    
    # Word analysis for multi-word entities
    words = []
    for entity in entities:
        words.extend(entity.split())
    word_counts = Counter(words)
    
    # N-gram analysis
    bigrams = list(ngrams(" ".join(entities).split(), 2))
    bigram_counts = Counter(bigrams)
    
    # Pattern analysis
    patterns = []
    for entity in entities:
        # Create a simplified pattern representation
        pattern = ""
        for char in entity:
            if char.isupper():
                pattern += "A"
            elif char.islower():
                pattern += "a"
            elif char.isdigit():
                pattern += "0"
            else:
                pattern += char
        patterns.append(pattern)
    pattern_counts = Counter(patterns)
    
    return {
        "count": len(entities),
        "unique_count": len(counts),
        "top_examples": counts.most_common(10),
        "random_examples": list(set(entities))[:10],
        "length_stats": {
            "min": min(lengths),
            "max": max(lengths),
            "avg": sum(lengths)/len(lengths),
            "distribution": Counter(lengths)
        },
        "character_distribution": dict(char_counts.most_common(20)),
        "common_words": dict(word_counts.most_common(20)),
         "common_bigrams": { " ".join(bigram): count for bigram, count in bigram_counts.most_common(10) },

        "pattern_distribution": dict(pattern_counts.most_common(10))
    }

def analyze_email_addresses(entity_data):
    """Analyze email address patterns"""
    entities = entity_data["entities"]
    results = {}
    
    # Domain analysis
    domains = []
    domain_tlds = []
    
    # URL pattern analysis
    url_patterns = {
        "indeed_urls": 0,
        "linkedin_urls": 0,
        "other_urls": 0
    }
    
    # Format patterns
    has_at = 0
    has_dots = 0
    malformed = 0
    
    for entity in entities:
        # Check for URLs
        if "indeed.com" in entity:
            url_patterns["indeed_urls"] += 1
        elif "linkedin.com" in entity:
            url_patterns["linkedin_urls"] += 1
        elif any(domain in entity for domain in [".com", ".org", ".net", ".edu"]):
            url_patterns["other_urls"] += 1
        
        # Analyze email components
        if "@" in entity:
            has_at += 1
            parts = entity.split("@")
            if len(parts) == 2 and "." in parts[1]:
                domains.append(parts[1])
                domain_parts = parts[1].split(".")
                if len(domain_parts) > 1:
                    domain_tlds.append(domain_parts[-1])
            
        if "." in entity:
            has_dots += 1
        
        # Check for potentially malformed emails
        if "@" not in entity and not any(domain in entity for domain in ["indeed.com", "linkedin.com"]):
            malformed += 1
    
    # Compile results
    results["format_stats"] = {
        "has_at_symbol": has_at,
        "has_dots": has_dots,
        "potentially_malformed": malformed
    }
    
    results["url_stats"] = url_patterns
    
    if domains:
        results["domain_stats"] = {
            "top_domains": Counter(domains).most_common(10),
            "top_tlds": Counter(domain_tlds).most_common(5)
        }
    
    return results

def analyze_names(entity_data):
    """Analyze name patterns"""
    entities = entity_data["entities"]
    results = {}
    
    # Name part analysis
    word_counts = []
    first_words = []
    last_words = []
    
    # Special patterns
    has_prefix = 0
    has_suffix = 0
    has_initials = 0
    
    prefixes = ["Mr", "Mr.", "Mrs", "Mrs.", "Ms", "Ms.", "Dr", "Dr.", "Prof", "Prof."]
    suffixes = ["Jr", "Jr.", "Sr", "Sr.", "III", "II", "IV", "PhD", "MD", "Esq", "Esq."]
    
    for entity in entities:
        parts = entity.split()
        word_counts.append(len(parts))
        
        if parts:
            first_words.append(parts[0])
            last_words.append(parts[-1])
        
        # Check for prefixes/titles
        if any(entity.startswith(prefix) for prefix in prefixes):
            has_prefix += 1
            
        # Check for suffixes
        if any(entity.endswith(suffix) for suffix in suffixes):
            has_suffix += 1
            
        # Check for initials (single letters with periods)
        if re.search(r'\b[A-Z]\.\s', entity) or re.search(r'\s[A-Z]\.\b', entity):
            has_initials += 1
    
    results["word_stats"] = {
        "avg_words": sum(word_counts)/len(word_counts) if word_counts else 0,
        "word_count_distribution": Counter(word_counts)
    }
    
    results["format_stats"] = {
        "has_prefix_or_title": has_prefix,
        "has_suffix": has_suffix,
        "has_initials": has_initials
    }
    
    if first_words:
        results["common_first_words"] = Counter(first_words).most_common(10)
    if last_words:
        results["common_last_words"] = Counter(last_words).most_common(10)
    
    return results

def analyze_degrees(entity_data):
    """Analyze degree patterns"""
    entities = entity_data["entities"]
    results = {}
    
    # Degree categories
    bachelors = 0
    masters = 0
    phd = 0
    diploma = 0
    other = 0
    
    # Format patterns
    has_subject = 0  # "in Computer Science"
    has_abbreviation = 0  # B.Tech, M.Sc
    has_full_form = 0  # Bachelor of Technology
    
    bachelor_patterns = [r'B\.?\s*Tech', r'B\.?\s*E', r'B\.?\s*Sc', r'B\.?\s*A', r'Bachelor']
    master_patterns = [r'M\.?\s*Tech', r'M\.?\s*E', r'M\.?\s*Sc', r'M\.?\s*A', r'Master', r'MBA']
    phd_patterns = [r'Ph\.?\s*D', r'Doctor', r'Doctorate']
    diploma_patterns = [r'Diploma', r'Certificate', r'PG\s*Dip']
    
    for entity in entities:
        # Categorize degree level
        if any(re.search(pattern, entity, re.I) for pattern in bachelor_patterns):
            bachelors += 1
        elif any(re.search(pattern, entity, re.I) for pattern in master_patterns):
            masters += 1
        elif any(re.search(pattern, entity, re.I) for pattern in phd_patterns):
            phd += 1
        elif any(re.search(pattern, entity, re.I) for pattern in diploma_patterns):
            diploma += 1
        else:
            other += 1
        
        # Check format patterns
        if re.search(r'\bin\b', entity):
            has_subject += 1
        
        if re.search(r'\b[A-Z]\.?[A-Z]?\.?[A-Z]?\.?\b', entity):
            has_abbreviation += 1
        
        if any(word in entity.lower() for word in ['bachelor', 'master', 'doctor']):
            has_full_form += 1
            
    # Extract subjects when present
    subjects = []
    for entity in entities:
        subject_match = re.search(r'\bin\b\s+(.+)', entity)
        if subject_match:
            subjects.append(subject_match.group(1).strip())
    
    results["degree_level"] = {
        "bachelors": bachelors,
        "masters": masters,
        "phd": phd,
        "diploma": diploma,
        "other": other
    }
    
    results["format_stats"] = {
        "with_subject": has_subject,
        "with_abbreviation": has_abbreviation,
        "with_full_form": has_full_form
    }
    
    if subjects:
        results["common_subjects"] = Counter(subjects).most_common(10)
    
    return results

def analyze_graduation_years(entity_data):
    """Analyze graduation year patterns"""
    entities = entity_data["entities"]
    results = {}
    
    # Year patterns
    years = []
    has_year_only = 0
    has_prefix = 0  # "Class of", "Batch of"
    has_month = 0  # "May 2015"
    
    for entity in entities:
        # Extract years
        year_match = re.search(r'(19|20)\d{2}', entity)
        if year_match:
            years.append(int(year_match.group(0)))
            
            # Check patterns
            if entity.strip() == year_match.group(0):
                has_year_only += 1
                
            if re.search(r'class\s+of|batch\s+of', entity.lower()):
                has_prefix += 1
                
            if re.search(r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}', 
                         entity.lower()):
                has_month += 1
    
    results["year_stats"] = {
        "min_year": min(years) if years else None,
        "max_year": max(years) if years else None,
        "most_common": Counter(years).most_common(5) if years else [],
        "year_distribution": Counter(years) if years else {}
    }
    
    results["format_stats"] = {
        "year_only": has_year_only,
        "with_prefix": has_prefix,
        "with_month": has_month
    }
    
    return results

def analyze_experience(entity_data):
    """Analyze years of experience patterns"""
    entities = entity_data["entities"]
    results = {}
    
    # Experience values
    years = []
    months = []
    has_range = 0  # "3-5 years"
    has_plus = 0   # "5+ years"
    has_text = 0   # "five years"
    
    for entity in entities:
        # Look for numeric years
        year_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:year|yr)s?', entity.lower())
        if year_match:
            years.append(float(year_match.group(1)))
        
        # Look for months
        month_match = re.search(r'(\d+)\s*month', entity.lower())
        if month_match:
            months.append(int(month_match.group(1)))
        
        # Check for ranges
        if re.search(r'\d+\s*-\s*\d+', entity):
            has_range += 1
            
        # Check for + notation
        if re.search(r'\d+\+', entity):
            has_plus += 1
            
        # Check for text numbers
        text_nums = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 
                     'eight', 'nine', 'ten']
        if any(num in entity.lower() for num in text_nums):
            has_text += 1
    
    results["year_stats"] = {
        "min": min(years) if years else None,
        "max": max(years) if years else None,
        "avg": sum(years)/len(years) if years else None,
        "distribution": Counter([int(y) for y in years]) if years else {}
    }
    
    results["format_stats"] = {
        "with_range": has_range,
        "with_plus": has_plus,
        "with_text_numbers": has_text
    }
    
    if months:
        results["month_stats"] = {
            "min": min(months),
            "max": max(months)
        }
    
    return results

def analyze_skills(entity_data):
    """Analyze skill patterns"""
    entities = entity_data["entities"]
    results = {}
    
    # Skill types
    categories = {
        "programming": ["java", "python", "javascript", "c++", "php", "ruby", "typescript", "scala"],
        "frameworks": ["react", "angular", "vue", "django", "flask", "spring", "express", ".net"],
        "databases": ["sql", "mysql", "postgresql", "mongodb", "oracle", "nosql", "redis"],
        "cloud": ["aws", "azure", "gcp", "cloud", "docker", "kubernetes", "terraform"],
        "tools": ["git", "jira", "jenkins", "maven", "gradle", "npm", "yarn"]
    }
    
    # Format patterns
    skill_counts = {cat: 0 for cat in categories}
    has_version = 0  # "Java 8", "Python 3.7"
    has_exp = 0      # "Java (5 years)"
    has_level = 0    # "Expert in Python"
    multiple_skills = 0  # "Java, Python, SQL"
    
    for entity in entities:
        entity_lower = entity.lower()
        
        # Categorize skills
        for category, keywords in categories.items():
            if any(re.search(r'\b' + re.escape(kw) + r'\b', entity_lower) for kw in keywords):
                skill_counts[category] += 1
        
        # Check for versions
        if re.search(r'\b\w+\s+\d+(?:\.\d+)*\b', entity):
            has_version += 1
            
        # Check for experience parenthetical
        if re.search(r'\(.*(?:year|month|yr).*\)', entity):
            has_exp += 1
            
        # Check for skill levels
        levels = ["basic", "beginner", "intermediate", "advanced", "expert", "proficient"]
        if any(level in entity_lower for level in levels):
            has_level += 1
            
        # Check for multiple skills
        if re.search(r',|\band\b', entity):
            multiple_skills += 1
    
    results["category_stats"] = skill_counts
    
    results["format_stats"] = {
        "with_version": has_version,
        "with_experience": has_exp,
        "with_level": has_level,
        "multiple_skills": multiple_skills
    }
    
    return results

def analyze_designations(entity_data):
    """Analyze job designation patterns"""
    entities = entity_data["entities"]
    results = {}
    
    # Job categories
    seniority = {
        "junior": ["junior", "jr", "associate", "trainee", "intern"],
        "mid_level": ["developer", "engineer", "analyst", "consultant"],
        "senior": ["senior", "sr", "lead", "manager", "head", "chief", "director", "principal"]
    }
    
    job_areas = {
        "engineering": ["developer", "engineer", "programmer", "architect", "devops"],
        "data": ["data", "analyst", "scientist", "ml", "ai"],
        "management": ["manager", "lead", "director", "head", "chief"],
        "design": ["designer", "ux", "ui"],
        "support": ["support", "help", "admin", "administrator"]
    }
    
    # Parse patterns
    seniority_counts = {level: 0 for level in seniority}
    area_counts = {area: 0 for area in job_areas}
    
    has_location = 0  # "Developer at Google"
    has_dept = 0      # "Software Engineer, AI Team"
    
    for entity in entities:
        entity_lower = entity.lower()
        
        # Check seniority indicators
        for level, keywords in seniority.items():
            if any(re.search(r'\b' + re.escape(kw) + r'\b', entity_lower) for kw in keywords):
                seniority_counts[level] += 1
        
        # Check job area
        for area, keywords in job_areas.items():
            if any(re.search(r'\b' + re.escape(kw) + r'\b', entity_lower) for kw in keywords):
                area_counts[area] += 1
        
        # Check for location
        if re.search(r'\bat\b|\bin\b|\bfor\b', entity):
            has_location += 1
            
        # Check for department
        if "," in entity:
            has_dept += 1
    
    results["seniority_stats"] = seniority_counts
    results["area_stats"] = area_counts
    
    results["format_stats"] = {
        "with_location_or_company": has_location,
        "with_department": has_dept
    }
    
    # Most common words
    words = []
    for entity in entities:
        words.extend(entity.lower().split())
    
    results["common_words"] = Counter(words).most_common(15)
    
    return results

def analyze_companies(entity_data):
    """Analyze company name patterns"""
    entities = entity_data["entities"]
    results = {}
    
    # Common types
    company_types = {
        "tech": ["technologies", "software", "systems", "digital", "tech"],
        "consulting": ["consulting", "consultancy", "consultants", "advisor"],
        "service": ["service", "services", "solutions"],
        "product": ["product", "products", "labs"]
    }
    
    # Legal forms
    legal_forms = ["ltd", "inc", "llc", "limited", "corporation", "corp", "gmbh", "pvt"]
    
    # Format analysis
    type_counts = {ctype: 0 for ctype in company_types}
    has_legal_form = 0
    has_location = 0  # "Microsoft India"
    has_prefix = 0    # "at Google", "for Apple"
    
    for entity in entities:
        entity_lower = entity.lower()
        
        # Check company type
        for ctype, keywords in company_types.items():
            if any(keyword in entity_lower for keyword in keywords):
                type_counts[ctype] += 1
        
        # Check for legal form
        if any(form in entity_lower.split() for form in legal_forms):
            has_legal_form += 1
            
        # Check for location
        location_patterns = [r'\b(india|us|usa|uk|australia|canada|singapore|germany|france)\b', 
                            r',\s*(inc|ltd|llc)']
        if any(re.search(pattern, entity_lower) for pattern in location_patterns):
            has_location += 1
            
        # Check for prefixes
        if re.match(r'^(?:at|with|for)\s+', entity_lower):
            has_prefix += 1
    
    results["type_stats"] = type_counts
    
    results["format_stats"] = {
        "with_legal_form": has_legal_form,
        "with_location": has_location,
        "with_prefix": has_prefix
    }
    
    # Word frequency (for identifying common company names)
    words = []
    for entity in entities:
        parts = re.split(r'\W+', entity.lower())
        words.extend([p for p in parts if p and p not in legal_forms])
    
    results["common_words"] = Counter(words).most_common(15)
    
    return results

def analyze_college_names(entity_data):
    """Analyze college name patterns"""
    entities = entity_data["entities"]
    results = {}
    
    # Institution types
    institution_types = {
        "university": ["university", "universities", "univ"],
        "institute": ["institute", "institution", "college"],
        "school": ["school", "academy"],
        "technical": ["technical", "technology", "engineering", "polytechnic"]
    }
    
    # Format analysis
    type_counts = {itype: 0 for itype in institution_types}
    has_location = 0      # "University of California, Berkeley"
    has_abbreviation = 0  # "MIT", "UCLA", "IIT"
    
    for entity in entities:
        entity_lower = entity.lower()
        
        # Check institution type
        for itype, keywords in institution_types.items():
            if any(keyword in entity_lower for keyword in keywords):
                type_counts[itype] += 1
        
        # Check for location indicators
        if "," in entity or " of " in entity_lower:
            has_location += 1
            
        # Check for abbreviations (all caps words)
        if re.search(r'\b[A-Z]{2,}\b', entity):
            has_abbreviation += 1
    
    results["type_stats"] = type_counts
    
    results["format_stats"] = {
        "with_location": has_location,
        "with_abbreviation": has_abbreviation
    }
    
    return results

def analyze_locations(entity_data):
    """Analyze location patterns"""
    entities = entity_data["entities"]
    results = {}
    
    # Format patterns
    has_comma = 0  # "New York, NY"
    has_state_code = 0  # 2-letter state/province codes
    has_multiple_parts = 0  # Location with multiple parts separated by spaces
    
    # Location types (keywords often seen in location entities)
    location_types = {
        "city": ["city", "town", "village"],
        "state": ["state", "province", "county"],
        "area": ["area", "district", "region", "zone"]
    }
    
    type_counts = {ltype: 0 for ltype in location_types}
    
    for entity in entities:
        entity_lower = entity.lower()
        
        # Check format
        if "," in entity:
            has_comma += 1
            
        # Check for state/province codes
        if re.search(r',\s*[A-Z]{2}$', entity):
            has_state_code += 1
            
        # Check word count
        if len(entity.split()) > 1:
            has_multiple_parts += 1
            
        # Check location type indicators
        for ltype, keywords in location_types.items():
            if any(keyword in entity_lower for keyword in keywords):
                type_counts[ltype] += 1
    
    results["format_stats"] = {
        "with_comma": has_comma,
        "with_state_code": has_state_code,
        "multi_part": has_multiple_parts
    }
    
    results["type_stats"] = type_counts
    
    return results

def analyze_resume_entities(data):
    """Run comprehensive analysis on all entity types in resume data"""
    entity_types = [
        "Email Address", "Name", "Degree", "Graduation Year", 
        "Years of Experience", "Skills", "Designation", 
        "Companies worked at", "College Name", "Location"
    ]
    
    results = {}
    
    for entity_type in entity_types:
        print(f"Analyzing {entity_type}...")
        entity_data = analyze_entities(data, entity_type)
        
        # Run basic pattern analysis for all types
        basic_analysis = entity_pattern_analysis(entity_data)
        
        # Run specialized analysis for this entity type
        specialized_analysis = None
        
        if entity_type == "Email Address":
            specialized_analysis = analyze_email_addresses(entity_data)
        elif entity_type == "Name":
            specialized_analysis = analyze_names(entity_data)
        elif entity_type == "Degree":
            specialized_analysis = analyze_degrees(entity_data)
        elif entity_type == "Graduation Year":
            specialized_analysis = analyze_graduation_years(entity_data)
        elif entity_type == "Years of Experience":
            specialized_analysis = analyze_experience(entity_data)
        elif entity_type == "Skills":
            specialized_analysis = analyze_skills(entity_data)
        elif entity_type == "Designation":
            specialized_analysis = analyze_designations(entity_data)
        elif entity_type == "Companies worked at":
            specialized_analysis = analyze_companies(entity_data)
        elif entity_type == "College Name":
            specialized_analysis = analyze_college_names(entity_data)
        elif entity_type == "Location":
            specialized_analysis = analyze_locations(entity_data)
        
        # Combine analyses
        combined_analysis = {
            "basic": basic_analysis
        }
        
        if specialized_analysis:
            combined_analysis["specialized"] = specialized_analysis
            
        results[entity_type] = combined_analysis
    
    return results


def visualize_entity_analysis(analysis_results, entity_type):
    """Generate visualizations for entity analysis results"""
    if entity_type not in analysis_results:
        print(f"No analysis results for {entity_type}")
        return
    
    print(f"\n--- Visualization for entity: {entity_type} ---")
    results = analysis_results[entity_type]
    basic = results.get("basic", {})
    
    fig = plt.figure(figsize=(15, 12))
    fig.suptitle(f"Analysis of '{entity_type}' Entities", fontsize=16)
    
    # 1. Length distribution
    if "length_stats" in basic and "distribution" in basic["length_stats"]:
        ax1 = fig.add_subplot(2, 2, 1)
        lengths = basic["length_stats"]["distribution"]
        if lengths:
            x_vals = list(lengths.keys())
            y_vals = list(lengths.values())
            print(f"[Length Distribution] X (Length): {x_vals}")
            print(f"[Length Distribution] Y (Count): {y_vals}")
            sns.barplot(x=x_vals, y=y_vals, ax=ax1)
            ax1.set_title("Entity Length Distribution")
            ax1.set_xlabel("Length")
            ax1.set_ylabel("Count")
    
    # 2. Top examples
    if "top_examples" in basic:
        ax2 = fig.add_subplot(2, 2, 2)
        top_examples = basic["top_examples"]
        if top_examples:
            examples = [ex[0] for ex in top_examples]
            counts = [ex[1] for ex in top_examples]
            print(f"[Top Examples] Y (Entity): {examples}")
            print(f"[Top Examples] X (Count): {counts}")
            sns.barplot(y=examples, x=counts, ax=ax2)
            ax2.set_title("Most Common Entities")
            ax2.set_xlabel("Count")
    
    # 3. Pattern distribution
    if "pattern_distribution" in basic:
        ax3 = fig.add_subplot(2, 2, 3)
        patterns = basic["pattern_distribution"]
        if patterns:
            pattern_labels = list(patterns.keys())[:8]
            pattern_counts = [patterns[p] for p in pattern_labels]
            print(f"[Pattern Distribution] Y (Patterns): {pattern_labels}")
            print(f"[Pattern Distribution] X (Count): {pattern_counts}")
            sns.barplot(y=pattern_labels, x=pattern_counts, ax=ax3)
            ax3.set_title("Common Patterns")
            ax3.set_xlabel("Count")
    
    # 4. Specialized visualizations
    if "specialized" in results:
        specialized = results["specialized"]
        ax4 = fig.add_subplot(2, 2, 4)
        
        if entity_type == "Skills" and "category_stats" in specialized:
            cats = specialized["category_stats"]
            y_vals = list(cats.keys())
            x_vals = list(cats.values())
            print(f"[Skill Categories] Y (Category): {y_vals}")
            print(f"[Skill Categories] X (Count): {x_vals}")
            sns.barplot(y=y_vals, x=x_vals, ax=ax4)
            ax4.set_title("Skill Categories")
        
        elif entity_type == "Degree" and "degree_level" in specialized:
            levels = specialized["degree_level"]
            y_vals = list(levels.keys())
            x_vals = list(levels.values())
            print(f"[Degree Levels] Y (Level): {y_vals}")
            print(f"[Degree Levels] X (Count): {x_vals}")
            sns.barplot(y=y_vals, x=x_vals, ax=ax4)
            ax4.set_title("Degree Levels")
        
        elif entity_type == "Years of Experience" and "year_stats" in specialized:
            if "distribution" in specialized["year_stats"]:
                years_dist = specialized["year_stats"]["distribution"]
                if years_dist:
                    x_vals = list(years_dist.keys())
                    y_vals = list(years_dist.values())
                    print(f"[Years of Experience] X (Years): {x_vals}")
                    print(f"[Years of Experience] Y (Count): {y_vals}")
                    sns.barplot(x=x_vals, y=y_vals, ax=ax4)
                    ax4.set_title("Years of Experience Distribution")
        
        elif entity_type == "Graduation Year" and "year_stats" in specialized:
            if "year_distribution" in specialized["year_stats"]:
                year_dist = specialized["year_stats"]["year_distribution"]
                if year_dist:
                    years = sorted(year_dist.keys())
                    counts = [year_dist[y] for y in years]
                    print(f"[Graduation Year] X (Years): {years}")
                    print(f"[Graduation Year] Y (Count): {counts}")
                    sns.barplot(x=years, y=counts, ax=ax4)
                    ax4.set_title("Graduation Year Distribution")
                    ax4.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


# Run analysis on all entity types
def analyze_all_entities(data):
    results = analyze_resume_entities(data)
    
    # Print summary of findings
    print("\n=== ENTITY ANALYSIS SUMMARY ===\n")
    for entity_type, analysis in results.items():
        print(f"*** {entity_type} ***")
        basic = analysis.get("basic", {})
        print(f"Total: {basic.get('count', 0)} entities, {basic.get('unique_count', 0)} unique")
        
        if "specialized" in analysis:
            spec = analysis["specialized"]
            print("Key insights:")
            
            if entity_type == "Email Address" and "format_stats" in spec:
                fmt = spec["format_stats"]
                print(f"- {fmt.get('has_at_symbol', 0)} have @ symbol")
                print(f"- {fmt.get('potentially_malformed', 0)} potentially malformed")
                
            elif entity_type == "Skills" and "category_stats" in spec:
                cats = spec["category_stats"]
                top_cat = max(cats.items(), key=lambda x: x[1])[0]
                print(f"- Most common category: {top_cat} ({cats[top_cat]} occurrences)")
                fmt = spec.get("format_stats", {})
                print(f"- {fmt.get('multiple_skills', 0)} entries have multiple skills")
                
        
        print()
    
    return results

