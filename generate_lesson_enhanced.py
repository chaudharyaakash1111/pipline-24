"""
generate_lesson_enhanced.py - Enhanced lesson generator that combines data from Wikipedia and Ollama
"""

import os
import json
import logging
import subprocess
import requests
import time
from typing import Dict, Any, Optional, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("enhanced_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import local modules
try:
    from wikipedia_utils import get_relevant_wikipedia_info
    from knowledge_store import save_lesson, get_lesson
except ImportError:
    logger.warning("Could not import one or more local modules. Some functionality may be limited.")

# Define constants
OLLAMA_API_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "llama3"
FALLBACK_OLLAMA_MODELS = ["mistral", "phi", "gemma", "llama2"]

def check_ollama_service() -> Tuple[bool, str]:
    """
    Check if Ollama service is running and which models are available
    
    Returns:
        Tuple[bool, str]: (is_running, available_model)
    """
    try:
        # Check if Ollama API is responding
        response = requests.get("http://localhost:11434", timeout=5)
        if response.status_code != 200:
            return False, ""
        
        # Check available models
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return False, ""
        
        # Parse output to find available models
        models = result.stdout.strip().split("\n")[1:]  # Skip header
        available_models = []
        
        for model_line in models:
            if not model_line:
                continue
            parts = model_line.split()
            if parts:
                model_name = parts[0]
                available_models.append(model_name)
        
        # Choose the best available model
        for model in [DEFAULT_OLLAMA_MODEL] + FALLBACK_OLLAMA_MODELS:
            if model in available_models:
                return True, model
        
        # If no preferred models are available but there are other models
        if available_models:
            return True, available_models[0]
        
        return True, ""  # Ollama is running but no models available
    
    except Exception as e:
        logger.error(f"Error checking Ollama service: {str(e)}")
        return False, ""

def generate_with_ollama(subject: str, topic: str, model: str) -> Optional[Dict[str, Any]]:
    """
    Generate a lesson using Ollama
    
    Args:
        subject: The subject of the lesson
        topic: The topic of the lesson
        model: The Ollama model to use
        
    Returns:
        Optional[Dict[str, Any]]: The generated lesson or None if generation failed
    """
    try:
        # Create a comprehensive prompt for Ollama
        prompt = f"""
        Create a comprehensive, educational lesson on {topic} within the subject of {subject} based on ancient Indian wisdom traditions.

        The lesson should be structured as follows:
        1. A meaningful title that captures the essence of the lesson
        2. An authentic Sanskrit shloka or verse related to the topic
        3. An accurate English translation of the shloka
        4. A detailed explanation (at least 300 words) that:
           - Explains the core concepts in depth
           - Connects the topic to ancient Indian knowledge systems
           - Provides historical context and significance
           - Relates the wisdom to modern understanding
        5. A practical activity for students to engage with the concept
        6. A thought-provoking reflective question

        If the subject is Veda, focus on Vedic knowledge, mantras, and philosophical concepts.
        If the subject is Ayurveda, focus on holistic health principles, treatments, and wellness practices.
        If the subject is Ganita (mathematics), focus on ancient Indian mathematical concepts, techniques, and their applications.
        If the subject is Yoga, focus on yogic practices, philosophy, and their benefits.
        If the subject is Darshana, focus on the philosophical schools and their core tenets.

        Format your response as a valid JSON object with the following structure:
        {{
            "title": "The title of the lesson",
            "shloka": "The Sanskrit shloka or verse",
            "translation": "English translation of the shloka",
            "explanation": "Detailed explanation of the concept",
            "activity": "A practical activity for students",
            "question": "A reflective question for students",
            "subject": "{subject}",
            "topic": "{topic}"
        }}

        Ensure the content is authentic, respectful of the tradition, educationally valuable, and historically accurate.
        """

        # Call Ollama API
        logger.info(f"Generating lesson with Ollama model: {model}")
        response = requests.post(
            OLLAMA_API_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "top_k": 40
                }
            },
            timeout=60
        )
        
        if response.status_code != 200:
            logger.error(f"Ollama API error: {response.status_code} - {response.text}")
            return None
        
        # Parse the response
        result = response.json()
        output_text = result.get("response", "")
        
        # Extract JSON from the response
        try:
            # Find JSON object in the text
            json_start = output_text.find("{")
            json_end = output_text.rfind("}") + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = output_text[json_start:json_end]
                lesson_data = json.loads(json_str)
                
                # Ensure all required fields are present
                required_fields = ["title", "shloka", "translation", "explanation", "activity", "question"]
                for field in required_fields:
                    if field not in lesson_data:
                        lesson_data[field] = f"Missing {field} information"
                
                # Add subject and topic if not present
                if "subject" not in lesson_data:
                    lesson_data["subject"] = subject
                if "topic" not in lesson_data:
                    lesson_data["topic"] = topic
                
                logger.info(f"Successfully generated lesson with Ollama: {lesson_data['title']}")
                return lesson_data
            else:
                logger.error("Could not find JSON object in Ollama response")
                return None
        
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from Ollama response: {str(e)}")
            logger.debug(f"Raw response: {output_text}")
            return None
    
    except Exception as e:
        logger.error(f"Error generating lesson with Ollama: {str(e)}")
        return None

def create_enhanced_lesson(subject: str, topic: str) -> Dict[str, Any]:
    """
    Create an enhanced lesson by combining data from multiple sources
    
    Args:
        subject: The subject of the lesson
        topic: The topic of the lesson
        
    Returns:
        Dict[str, Any]: The enhanced lesson data
    """
    logger.info(f"Creating enhanced lesson for subject: {subject}, topic: {topic}")
    
    # Check if lesson already exists in knowledge store
    existing_lesson = get_lesson(subject, topic)
    if existing_lesson:
        logger.info(f"Found existing lesson in knowledge store for {subject}/{topic}")
        return existing_lesson
    
    # Initialize lesson data
    lesson_data = {
        "subject": subject,
        "topic": topic,
        "title": f"Lesson on {subject}: {topic}",
        "shloka": "",
        "translation": "",
        "explanation": "",
        "activity": "",
        "question": "",
        "wikipedia_info": None,
        "sources": []
    }
    
    # Step 1: Get Wikipedia information
    try:
        wiki_data = get_relevant_wikipedia_info(subject, topic)
        if wiki_data["wikipedia"]["title"]:
            lesson_data["wikipedia_info"] = {
                "title": wiki_data["wikipedia"]["title"],
                "summary": wiki_data["wikipedia"]["summary"],
                "url": wiki_data["wikipedia"]["url"],
                "related_articles": wiki_data["wikipedia"]["related_articles"]
            }
            lesson_data["sources"].append("Wikipedia")
            
            # Create a one-paragraph summary from Wikipedia
            if wiki_data["wikipedia"]["summary"]:
                summary = wiki_data["wikipedia"]["summary"]
                # Limit to first paragraph or first 500 characters
                first_para = summary.split("\n")[0]
                if len(first_para) > 500:
                    first_para = first_para[:497] + "..."
                lesson_data["wikipedia_summary"] = first_para
    except Exception as e:
        logger.error(f"Error fetching Wikipedia information: {str(e)}")
    
    # Step 2: Generate content with Ollama
    ollama_running, model = check_ollama_service()
    if ollama_running and model:
        ollama_lesson = generate_with_ollama(subject, topic, model)
        if ollama_lesson:
            # Update lesson data with Ollama-generated content
            for key in ["title", "shloka", "translation", "explanation", "activity", "question"]:
                if key in ollama_lesson and ollama_lesson[key]:
                    lesson_data[key] = ollama_lesson[key]
            
            lesson_data["sources"].append(f"Ollama ({model})")
    else:
        logger.warning("Ollama service is not available. Using fallback content.")
    
    # Step 3: If we have Wikipedia info but no Ollama content, enhance the explanation
    if lesson_data["wikipedia_info"] and not lesson_data["sources"]:
        if "wikipedia_summary" in lesson_data:
            lesson_data["explanation"] = f"According to Wikipedia: {lesson_data['wikipedia_summary']}\n\nThis topic is significant in {subject} because it represents an important aspect of ancient Indian knowledge systems."
    
    # Step 4: If we still don't have content, use mock data
    if not lesson_data["explanation"]:
        lesson_data = generate_mock_lesson(subject, topic)
        lesson_data["sources"].append("Mock Data")
    
    # Save the lesson to knowledge store
    save_lesson(lesson_data)
    
    return lesson_data

def generate_mock_lesson(subject: str, topic: str) -> Dict[str, Any]:
    """
    Generate a mock lesson when other methods fail
    
    Args:
        subject: The subject of the lesson
        topic: The topic of the lesson
        
    Returns:
        Dict[str, Any]: The mock lesson data
    """
    # Convert subject and topic to lowercase for case-insensitive matching
    subject_lower = subject.lower()
    topic_lower = topic.lower()
    
    # Define mock lessons for different subjects
    mock_lessons = {
        "ved": {
            "sound": {
                "title": "The Sacred Sound in Vedic Tradition",
                "shloka": "ॐ अग्निमीळे पुरोहितं यज्ञस्य देवम् ऋत्विजम्",
                "translation": "Om, I praise Agni, the priest of the sacrifice, the divine, the ritual performer.",
                "explanation": "In Vedic tradition, sound (shabda) is considered not just a physical phenomenon but a spiritual one. The primordial sound OM (ॐ) is considered the source of all creation. Mantras are sacred sound formulas that, when recited with proper pronunciation, rhythm, and devotion, create powerful vibrations affecting both the chanter and the environment. The concept of Nada Brahma suggests that the entire universe is a manifestation of sound vibration.",
                "activity": "Recite the shloka aloud thrice, paying attention to the vibration you feel in different parts of your body. Then, sit in silence for 2 minutes and observe the subtle sounds around you.",
                "question": "How does the concept of sound in Vedic tradition differ from the modern scientific understanding of sound?"
            }
        },
        "yoga": {
            "asana": {
                "title": "The Philosophy of Asanas in Yoga",
                "shloka": "स्थिरसुखमासनम्॥४६॥",
                "translation": "Posture (asana) should be steady and comfortable.",
                "explanation": "This sutra from Patanjali's Yoga Sutras defines the essence of asana practice. Unlike modern interpretations that focus primarily on physical benefits, traditional yoga views asanas as tools for preparing the body for meditation. The steadiness (sthira) refers to the ability to hold a position, while comfort (sukha) ensures that the practitioner can remain in the pose without distraction or discomfort, allowing the mind to remain calm.",
                "activity": "Practice Sukhasana (Easy Pose) for 5 minutes, focusing on finding both steadiness and comfort. Observe how your breath affects your ability to maintain both qualities.",
                "question": "How does the traditional understanding of asana differ from modern yoga practice, and why is this distinction important?"
            }
        },
        "ganita": {
            "algebra": {
                "title": "Bījagaṇita: The Science of Algebra in Ancient India",
                "shloka": "यथा शिखा मयूराणां नागानां मणयो यथा। तद्वद्वेदाङ्गशास्त्राणां गणितं मूर्धनि स्थितम्॥",
                "translation": "Just as the crest on the peacock, or the gem on the head of a snake, so is mathematics at the head of all sciences.",
                "explanation": "Bījagaṇita (algebra) in ancient Indian mathematics was a sophisticated system for solving equations and working with unknowns. The term 'bīja' means 'seed' or 'element' and 'gaṇita' means 'calculation'. Mathematicians like Brahmagupta (7th century) and Bhaskara II (12th century) made significant contributions to algebra, developing methods for solving linear and quadratic equations, as well as indeterminate equations. The 'chakravala' method for solving Pell's equation was discovered centuries before European mathematicians tackled similar problems.",
                "activity": "Try to solve this ancient Indian algebraic problem: If a number is multiplied by 3, and then 12 is added, the result is 30. What is the original number?",
                "question": "How did the development of algebra in ancient India differ from its development in other civilizations, and what unique insights did Indian mathematicians contribute?"
            }
        },
        "ayurveda": {
            "doshas": {
                "title": "The Tridosha Theory: Foundation of Ayurvedic Medicine",
                "shloka": "वातपित्तकफा दोषा धातवस्तु रसादयः। मलामूत्रपुरीषाणि दोषधातुमलाः स्मृताः॥",
                "translation": "Vata, pitta and kapha are the doshas; rasa and others are the dhatus; urine and feces are the malas; these are remembered as the dosha, dhatu and mala.",
                "explanation": "The Tridosha theory forms the cornerstone of Ayurvedic medicine, proposing that three biological energies or doshas—Vata (air and space), Pitta (fire and water), and Kapha (earth and water)—govern all physiological and psychological functions. Each person has a unique constitution or prakriti determined by the proportion of these doshas at conception. Health is maintained when the doshas are in balance according to one's natural constitution, while imbalance leads to disease.",
                "activity": "Observe your daily patterns of hunger, energy, and sleep for three days. Note which dosha characteristics seem most prominent in your constitution based on Ayurvedic descriptions.",
                "question": "How might the Tridosha theory complement modern medical understanding of human physiology and personalized medicine?"
            }
        }
    }
    
    # Try to find a matching lesson in our pre-defined lessons
    if subject_lower in mock_lessons and topic_lower in mock_lessons[subject_lower]:
        lesson = mock_lessons[subject_lower][topic_lower].copy()
        lesson["subject"] = subject
        lesson["topic"] = topic
        return lesson
    
    # If no match, return a generic lesson
    return {
        "subject": subject,
        "topic": topic,
        "title": f"Lesson on {subject}: {topic}",
        "shloka": "ॐ सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः। सर्वे भद्राणि पश्यन्तु मा कश्चिद्दुःखभाग्भवेत्॥",
        "translation": "May all be happy, may all be free from disease, may all see auspiciousness, may none suffer.",
        "explanation": f"This lesson explores the topic of {topic} within the context of {subject}. In ancient Indian knowledge systems, every subject was approached with a holistic perspective that integrated practical knowledge with spiritual wisdom. The study of {topic} would have been undertaken not merely for intellectual understanding but for its application in creating harmony and balance in life.",
        "activity": f"Research the relationship between {subject} and {topic} in traditional Indian knowledge systems. Write down three key insights you discover.",
        "question": f"How does the understanding of {topic} in {subject} differ from modern interpretations?"
    }

if __name__ == "__main__":
    # Test the module
    test_subject = "Ganita"
    test_topic = "Algebra"
    
    print(f"Testing enhanced lesson generation for {test_subject}/{test_topic}")
    lesson = create_enhanced_lesson(test_subject, test_topic)
    
    print(f"Generated lesson: {lesson['title']}")
    print(f"Sources: {lesson['sources']}")
    
    if "wikipedia_info" in lesson and lesson["wikipedia_info"]:
        print(f"Wikipedia article: {lesson['wikipedia_info']['title']}")
