from typing import Any, Text, Dict, List
from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.events import SlotSet

class ActionExtractResumeInfo(Action):
    def name(self) -> Text:
        return "action_extract_resume_info"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Get entities from latest message
        entities = tracker.latest_message.get("entities", [])
        
        # Log the entities received
        print(f"[ActionExtractResumeInfo] Received {len(entities)} entities")
        for ent in entities:
            print(f"  - {ent.get('entity')}: '{ent.get('value')}' from {ent.get('extractor')}")
        
        # Map from domain entity names to slot names
        entity_to_slot = {
            "Email Address": "email",
            "Location": "location",
            "Designation": "designation", 
            "Companies worked at": "companies",
            "Degree": "degree",
            "Skills": "skills",
            "College Name": "college",
            "Name": "name"  
        }
        
        # Track all slots we'll set
        slot_events = []
        extracted_info = {}
        
        # For each entity, map to appropriate slot
        for entity in entities:
            entity_type = entity.get("entity")
            entity_value = entity.get("value")
            
            if entity_type in entity_to_slot:
                slot_name = entity_to_slot[entity_type]
                
                # Handle list slots differently
                if slot_name in ["skills", "companies", "experience", "education"]:
                    # For lists, get current value or initialize empty list
                    current_value = tracker.get_slot(slot_name) or []
                    if isinstance(current_value, list):
                        if entity_value not in current_value:
                            current_value.append(entity_value)
                    else:
                        current_value = [entity_value]
                    
                    slot_events.append(SlotSet(slot_name, current_value))
                    extracted_info[slot_name] = current_value
                else:
                    # For regular text slots
                    slot_events.append(SlotSet(slot_name, entity_value))
                    extracted_info[slot_name] = entity_value
        
        # Create response message
        if extracted_info:
            message = "I've extracted the following information from your resume:\n\n"
            
            if "name" in extracted_info:
                message += f"Name: {extracted_info['name']}\n"
            if "email" in extracted_info:
                message += f"Email: {extracted_info['email']}\n"
            if "location" in extracted_info:
                message += f"Location: {extracted_info['location']}\n"
            if "companies" in extracted_info:
                message += f"Companies: {', '.join(extracted_info['companies'])}\n"
            if "skills" in extracted_info:
                message += f"Skills: {', '.join(extracted_info['skills'])}\n"
            if "degree" in extracted_info:
                message += f"Degree: {extracted_info['degree']}\n"
            if "college" in extracted_info:
                message += f"College: {extracted_info['college']}\n"
            if "designation" in extracted_info:
                message += f"Designation: {extracted_info['designation']}\n"
            
            dispatcher.utter_message(text=message)
        else:
            dispatcher.utter_message(text="I couldn't extract any information from your resume. Could you provide more details?")
        
        return slot_events

class ActionDebugInfo(Action):
    def name(self) -> Text:
        return "action_debug_info"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Get all slots and their values
        slots = tracker.slots
        
        message = "Here's the information I have:\n\n"
        
        for slot_name, slot_value in slots.items():
            if slot_value is not None and slot_name != "resume_text":
                message += f"{slot_name}: {slot_value}\n"
        
        dispatcher.utter_message(text=message)
        
        return []

class ActionUpdateEntity(Action):
    def name(self) -> Text:
        return "action_update_entity"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Check for direct entity updates
        entities = {
            "name": next(tracker.get_latest_entity_values("name"), None),
            "email": next(tracker.get_latest_entity_values("email"), None),
            "location": next(tracker.get_latest_entity_values("location"), None),
            "skills": list(tracker.get_latest_entity_values("skills")),
            "companies": list(tracker.get_latest_entity_values("companies")),
            "degree": next(tracker.get_latest_entity_values("degree"), None),
            "college": next(tracker.get_latest_entity_values("college"), None),
            "designation": next(tracker.get_latest_entity_values("designation"), None),
        }
        
        # For entity_type + value pairs
        entity_type = next(tracker.get_latest_entity_values("entity_type"), None)
        
        slot_events = []
        updated_fields = []
        
        # Handle direct entity updates
        for slot_name, entity_value in entities.items():
            if entity_value:
                if isinstance(entity_value, list) and entity_value:  # For list slots like skills, companies
                    current_value = tracker.get_slot(slot_name) or []
                    if not isinstance(current_value, list):
                        current_value = []
                    
                    # Add new values that aren't already in the list
                    for value in entity_value:
                        if value not in current_value:
                            current_value.append(value)
                    
                    slot_events.append(SlotSet(slot_name, current_value))
                    updated_fields.append(f"{slot_name}: {', '.join(current_value)}")
                else:  # For text slots
                    slot_events.append(SlotSet(slot_name, entity_value))
                    updated_fields.append(f"{slot_name}: {entity_value}")
        
        # If nothing was updated but we have entity_type, try to use full message
        if not updated_fields and entity_type:
            # Get full message as a fallback value
            message_text = tracker.latest_message.get("text", "")
            
            # Try to extract value after "is" or "are" or "to"
            import re
            match = re.search(f"(?:{entity_type} is |{entity_type} are |{entity_type} to )(.+)", message_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                
                # Handle lists for skills and companies
                if entity_type in ["skills", "companies"]:
                    value_list = [v.strip() for v in value.split(',')]
                    current_value = tracker.get_slot(entity_type) or []
                    if not isinstance(current_value, list):
                        current_value = []
                    
                    for v in value_list:
                        if v and v not in current_value:
                            current_value.append(v)
                    
                    slot_events.append(SlotSet(entity_type, current_value))
                    updated_fields.append(f"{entity_type}: {', '.join(current_value)}")
                else:
                    slot_events.append(SlotSet(entity_type, value))
                    updated_fields.append(f"{entity_type}: {value}")
        
        if updated_fields:
            dispatcher.utter_message(text=f"I've updated the following information:\n" + "\n".join(updated_fields))
        else:
            # Guide the user if nothing was updated
            dispatcher.utter_message(
                text="I couldn't understand which information you want to update. "
                "Please specify what you'd like to update, for example:\n"
                "- 'My name is Aymen'\n"
                "- 'My location is Tamil Nadu'\n"
                "- 'My skills are Python and Java'"
            )
        
        return slot_events

class ActionSummarizeInfo(Action):
    def name(self) -> Text:
        return "action_summarize_info"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Get all slots and their values
        name = tracker.get_slot("name")
        email = tracker.get_slot("email")
        phone = tracker.get_slot("phone")
        skills = tracker.get_slot("skills")
        education = tracker.get_slot("education")
        experience = tracker.get_slot("experience")
        location = tracker.get_slot("location")
        degree = tracker.get_slot("degree")
        college = tracker.get_slot("college")
        companies = tracker.get_slot("companies")
        designation = tracker.get_slot("designation")
        
        # Create a summary message
        summary = "Here's a summary of your resume:\n\n"
        
        if name:
            summary += f"Name: {name}\n"
        if email:
            summary += f"Email: {email}\n"
        if phone:
            summary += f"Phone: {phone}\n"
        if location:
            summary += f"Location: {location}\n"
        
        if experience or companies or designation:
            summary += "\nExperience:\n"
            if designation:
                summary += f"Position: {designation}\n"
            if experience and isinstance(experience, list):
                for exp in experience:
                    summary += f"- {exp}\n"
            elif experience:
                summary += f"- {experience}\n"
                
            if companies and isinstance(companies, list):
                summary += "Companies: " + ", ".join(companies) + "\n"
            elif companies:
                summary += f"Company: {companies}\n"
        
        if education or degree or college:
            summary += "\nEducation:\n"
            if education and isinstance(education, list):
                for edu in education:
                    summary += f"- {edu}\n"
            elif education:
                summary += f"- {education}\n"
                
            if degree:
                summary += f"Degree: {degree}\n"
            if college:
                summary += f"College: {college}\n"
        
        if skills:
            summary += "\nSkills:\n"
            if isinstance(skills, list):
                for skill in skills:
                    summary += f"- {skill}\n"
            else:
                summary += f"- {skills}\n"
        
        dispatcher.utter_message(text=summary)
        
        return []

class ActionAskCorrection(Action):
    def name(self) -> Text:
        return "action_ask_correction"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Ask the user if they want to make any corrections
        dispatcher.utter_message(text="Is this information correct?")
        
        return []

class ActionShowUpdatedInfo(Action):
    def name(self) -> Text:
        return "action_show_updated_info"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        # Create a message with all current information
        message = "Here's the current information I have:\n\n"
        
        name = tracker.get_slot("name")
        email = tracker.get_slot("email")
        skills = tracker.get_slot("skills")
        location = tracker.get_slot("location")
        degree = tracker.get_slot("degree")
        college = tracker.get_slot("college")
        companies = tracker.get_slot("companies")
        designation = tracker.get_slot("designation")
        
        if name:
            message += f"Name: {name}\n"
        if email:
            message += f"Email: {email}\n"
        if location:
            message += f"Location: {location}\n"
        if companies:
            if isinstance(companies, list):
                message += f"Companies: {', '.join(companies)}\n"
            else:
                message += f"Company: {companies}\n"
        if skills:
            if isinstance(skills, list):
                message += f"Skills: {', '.join(skills)}\n"
            else:
                message += f"Skill: {skills}\n"
        if degree:
            message += f"Degree: {degree}\n"
        if college:
            message += f"College: {college}\n"
        if designation:
            message += f"Designation: {designation}\n"
        
        dispatcher.utter_message(text=message)
        
        return []