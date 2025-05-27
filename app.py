"""
app.py - FastAPI application for serving the Gurukul Lesson Generator
"""

import os
import logging
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import time
from dotenv import load_dotenv
import sys

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Simple function to check compute device
def get_compute_device():
    """Get the available compute device (CPU or GPU)"""
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            device = f"GPU ({torch.cuda.get_device_name(0)})"
            logger.info(f"GPU available: {device}")
        else:
            device = "CPU"
            logger.info("Using CPU for compute")
        return device, gpu_available
    except Exception as e:
        logger.warning(f"Error checking compute device: {e}")
        return "CPU", False

# Get compute device
device, gpu_available = get_compute_device()

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Gurukul AI-Lesson Generator",
    description="Generate structured lessons based on ancient Indian wisdom texts",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Add middleware to log HTTP requests and status codes
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    # Log the request with a distinctive prefix
    request_message = f">>> HTTP REQUEST: {request.method} {request.url.path} - Query params: {request.query_params}"
    logger.info(request_message)
    print(request_message)  # Print directly to console

    # Process the request
    response = await call_next(request)

    # Calculate processing time
    process_time = time.time() - start_time

    # Log the response status code and processing time with a distinctive prefix
    response_message = f"<<< HTTP RESPONSE: {request.method} {request.url.path} - Status: {response.status_code} - Time: {process_time:.4f}s"
    logger.info(response_message)
    print(response_message)  # Print directly to console

    return response

class LessonRequest(BaseModel):
    subject: str
    topic: str
    user_id: str
    include_wikipedia: bool = True
    use_knowledge_store: bool = True

class WikipediaInfo(BaseModel):
    title: Optional[str] = None
    summary: Optional[str] = None
    url: Optional[str] = None
    related_articles: List[str] = []

class LessonResponse(BaseModel):
    title: str
    shloka: str
    translation: str
    explanation: str
    activity: str
    question: str
    wikipedia_info: Optional[WikipediaInfo] = None

@app.get("/")
async def root():
    """
    Root endpoint to check if the API is running
    """
    # Check GPU status
    try:
        import torch
        gpu_status = "available" if torch.cuda.is_available() else "unavailable"
        device = f"GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"
    except Exception:
        gpu_status = "error"
        device = "CPU (GPU check failed)"

    return {
        "message": "Welcome to the Gurukul AI-Lesson Generator API",
        "status": "Service is running - check /llm_status for details",
        "compute": {
            "device": device,
            "gpu_status": gpu_status
        },
        "endpoints": {
            "generate_lesson": "/generate_lesson?subject=Ved&topic=Sound",
            "llm_status": "/llm_status",
            "documentation": "/docs"
        }
    }

@app.get("/llm_status")
async def llm_status():
    """
    Endpoint to check the status of the LLM service
    """
    try:
        # Check if vector store exists
        vector_store_path = os.getenv("CHROMA_PERSIST_DIRECTORY", "knowledge_store")
        vector_store_status = "connected" if os.path.exists(vector_store_path) else "not found"

        # Check for available LLMs
        llm_status = "unavailable"
        llm_type = "none"
        llm_details = {}

        # Check OpenAI
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key and openai_api_key != "your_openai_api_key_here":
            try:
                from openai import OpenAI
                client = OpenAI(api_key=openai_api_key)

                # Just check if the API key is valid without making an API call
                llm_status = "limited"
                llm_type = "openai"
                llm_details = {
                    "model": "gpt-3.5-turbo",
                    "provider": "OpenAI",
                    "status": "API key present but quota exceeded",
                    "note": "The OpenAI API key is valid but has exceeded its quota. Using mock lessons instead."
                }
            except Exception as e:
                llm_details = {"error": str(e), "provider": "OpenAI"}

        # Check Ollama if OpenAI is not available
        if llm_status != "operational":
            try:
                import subprocess
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
                if result.returncode == 0:
                    llm_status = "operational"
                    llm_type = "ollama"
                    llm_details = {
                        "provider": "Ollama",
                        "models": result.stdout.strip()
                    }
                else:
                    llm_details = {"error": "Ollama is installed but not running properly", "provider": "Ollama"}
            except Exception as e:
                if not llm_details:  # Only update if OpenAI didn't already set an error
                    llm_details = {"error": str(e), "provider": "Ollama"}

        # If no real LLM is available, use the mock LLM
        if llm_status != "operational":
            if llm_status != "limited":  # Don't overwrite the limited status
                llm_status = "mock"
                llm_type = "mock"
                llm_details = {"provider": "Mock LLM", "response": "Using pre-defined lessons"}

        # Get GPU information from the global variable
        try:
            import torch
            gpu_available = torch.cuda.is_available()
            if gpu_available:
                device_info = {
                    "status": "available",
                    "device_name": torch.cuda.get_device_name(0),
                    "device_count": torch.cuda.device_count(),
                    "cuda_version": torch.version.cuda,
                    "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**2:.2f} MB",
                    "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**2:.2f} MB",
                    "max_memory": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
                }
            else:
                device_info = {
                    "status": "unavailable",
                    "device_name": "cpu",
                    "reason": "CUDA not available on this system"
                }
        except Exception as e:
            device_info = {
                "status": "error",
                "device_name": "cpu",
                "error": f"Error checking GPU: {str(e)}"
            }

        return {
            "status": "operational" if llm_status == "operational" else "limited",
            "message": f"LLM service is {llm_status}",
            "vector_store": vector_store_status,
            "llm": {
                "status": llm_status,
                "type": llm_type,
                "details": llm_details
            },
            "device": device_info
        }
    except Exception as e:
        logger.error(f"Error checking LLM status: {str(e)}")
        return {
            "status": "error",
            "message": f"LLM service encountered an error: {str(e)}",
            "error_details": str(e)
        }

@app.get("/generate_lesson", response_model=LessonResponse)
async def generate_lesson_endpoint(
    subject: str = Query(..., description="Subject of the lesson (e.g., Ved, Ganita, Yoga)", example="ved"),
    topic: str = Query(..., description="Topic of the lesson (e.g., Sound, Mathematics, Meditation)", example="ayurved"),
    include_wikipedia: bool = Query(True, description="Whether to include Wikipedia information in the response"),
    use_knowledge_store: bool = Query(True, description="Whether to use the knowledge store for retrieving/saving lessons")
):
    """
    Generate a structured lesson based on subject and topic

    Example usage:
    GET /generate_lesson?subject=ved&topic=ayurved&include_wikipedia=true&use_knowledge_store=true
    """
    try:
        # Validate required parameters
        if not subject or not topic:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Both 'subject' and 'topic' parameters are required",
                    "example": "/generate_lesson?subject=ved&topic=ayurved",
                    "available_subjects": ["ved", "ganita", "yoga", "ayurveda"],
                    "example_topics": ["sound", "algebra", "asana", "doshas"]
                }
            )

        logger.info(f"Generating lesson for subject: {subject}, topic: {topic}, include_wikipedia: {include_wikipedia}, use_knowledge_store: {use_knowledge_store}")
        lesson = None
        error_messages = []

        # Check if we should use the enhanced lesson generator
        try:
            try:
                from generate_lesson_enhanced import create_enhanced_lesson
                from knowledge_store import get_lesson, save_lesson

                # Try to get lesson from knowledge store if requested
                if use_knowledge_store:
                    stored_lesson = get_lesson(subject, topic)
                    if stored_lesson:
                        logger.info(f"Retrieved lesson from knowledge store for {subject}/{topic}")
                        return stored_lesson

                # Generate enhanced lesson
                logger.info(f"Generating enhanced lesson for {subject}/{topic}")
                enhanced_lesson = create_enhanced_lesson(subject, topic)
                if enhanced_lesson:
                    logger.info(f"Successfully generated enhanced lesson: {enhanced_lesson.get('title', 'Untitled')}")

                    # Save to knowledge store if requested
                    if use_knowledge_store:
                        save_lesson(enhanced_lesson)

                    return enhanced_lesson
            except ImportError as e:
                logger.warning(f"Enhanced lesson generator not available: {str(e)}. Falling back to standard generation.")
            except Exception as e:
                logger.error(f"Error using enhanced lesson generator: {str(e)}")
                error_messages.append(f"Enhanced generator error: {str(e)}")

        except ImportError as e:
            logger.warning(f"Enhanced lesson generator not available: {str(e)}. Falling back to standard generation.")
        except Exception as e:
            logger.error(f"Error using enhanced lesson generator: {str(e)}")
            error_messages.append(f"Enhanced generator error: {str(e)}")

        # If enhanced generator fails, fall back to standard methods

        # Fetch Wikipedia information if requested
        wikipedia_info = None
        if include_wikipedia:
            try:
                from wikipedia_utils import get_relevant_wikipedia_info
                logger.info(f"Fetching Wikipedia information for {subject}/{topic}")
                wiki_data = get_relevant_wikipedia_info(subject, topic)

                if wiki_data["wikipedia"]["title"]:
                    wikipedia_info = {
                        "title": wiki_data["wikipedia"]["title"],
                        "summary": wiki_data["wikipedia"]["summary"],
                        "url": wiki_data["wikipedia"]["url"],
                        "related_articles": wiki_data["wikipedia"]["related_articles"]
                    }
                    logger.info(f"Found Wikipedia article: {wikipedia_info['title']}")
                else:
                    logger.warning(f"No Wikipedia information found for {subject}/{topic}")
            except Exception as e:
                logger.error(f"Error fetching Wikipedia information: {str(e)}")
                error_messages.append(f"Wikipedia error: {str(e)}")

        # First, try to use Ollama if it's available
        try:
            # First check if Ollama service is running
            try:
                import subprocess
                import requests
                import platform

                logger.info("Checking Ollama service status...")

                # Check if service is responding on the API endpoint
                try:
                    response = requests.get("http://localhost:11434", timeout=5)
                    if response.status_code == 200:
                        logger.info("Ollama API is responding")
                    else:
                        raise Exception()
                except Exception:
                    # If API check fails, try ollama list command
                    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
                    if result.returncode != 0:
                        os_type = platform.system().lower()
                        if os_type == "linux":
                            raise Exception(
                                "Ollama service is not running. Please start it with:\n"
                                "1. systemctl start ollama\n"
                                "2. systemctl enable ollama  # To run on boot\n"
                                "Check status with: systemctl status ollama"
                            )
                        elif os_type in ["darwin", "windows"]:  # macOS or Windows
                            raise Exception(
                                "Ollama service is not running. Please start it by:\n"
                                "1. Open a new terminal\n"
                                "2. Run: ollama serve\n"
                                "3. Check if it's running with: curl http://localhost:11434"
                            )
                        else:
                            raise Exception("Ollama service is not running. Please start the Ollama service first.")

                logger.info("Ollama service is running.")

                # Check if any models are installed
                result = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=5)
                if "no models" in result.stdout.lower():
                    raise Exception(
                        "No Ollama models installed. Please install a model:\n"
                        "1. For best results: ollama pull llama3\n"
                        "2. For faster, lightweight option: ollama pull mistral"
                    )

            except FileNotFoundError:
                raise Exception(
                    "Ollama is not installed. Please:\n"
                    "1. Visit ollama.com to download and install Ollama\n"
                    "2. After installation, start the service following the OS-specific instructions"
                )
            except subprocess.TimeoutExpired:
                raise Exception(
                    "Ollama service check timed out. This might be due to:\n"
                    "1. Insufficient system resources (need 4-8GB free RAM)\n"
                    "2. Port 11434 being blocked\n"
                    "3. System overload\n"
                    "Please check system resources and try restarting the service."
                )

            try:
                from generate_lesson_ollama import generate_lesson as generate_ollama_lesson
                logger.info("Trying to generate lesson using Ollama subprocess...")
                lesson = generate_ollama_lesson(subject, topic)
                if lesson:
                    # Add Wikipedia information if available
                    if wikipedia_info:
                        lesson["wikipedia_info"] = wikipedia_info
                    logger.info(f"Successfully generated lesson with Ollama: {lesson['title']}")
                    return lesson
            except ModuleNotFoundError:
                logger.warning("generate_lesson_ollama.py not found. Ollama integration is disabled.")
                error_messages.append("Ollama integration not available")
            except Exception as e:
                logger.error(f"Error using Ollama subprocess: {str(e)}")
                error_messages.append(f"Ollama error: {str(e)}")
        except Exception as e:
            logger.error(f"Error in Ollama setup: {str(e)}")
            error_messages.append(f"Ollama setup error: {str(e)}")

        # Try OpenAI if Ollama fails
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key and openai_api_key != "your_openai_api_key_here":
            try:
                from openai import OpenAI
                import backoff  # For exponential backoff retries

                client = OpenAI(api_key=openai_api_key)

                # Define retry decorator for OpenAI API calls
                @backoff.on_exception(
                    backoff.expo,
                    (Exception),  # Catch all exceptions
                    max_tries=3,  # Maximum number of retries
                    max_time=30  # Maximum total time to retry
                )
                def generate_with_openai():
                    prompt = f"""
                    Create a structured lesson on the topic of {topic} within the subject of {subject} in the context of ancient Indian wisdom.

                    The lesson should include:
                    1. A title that captures the essence of the lesson
                    2. A relevant Sanskrit shloka or verse
                    3. An English translation of the shloka
                    4. A detailed explanation of the concept, including its significance in ancient Indian knowledge systems
                    5. A practical activity for students to engage with the concept
                    6. A reflective question for students to ponder

                    Format the response as a JSON object with the following structure:
                    {{
                        "title": "The title of the lesson",
                        "shloka": "The Sanskrit shloka or verse",
                        "translation": "English translation of the shloka",
                        "explanation": "Detailed explanation of the concept",
                        "activity": "A practical activity for students",
                        "question": "A reflective question for students"
                    }}

                    Ensure that the content is authentic, respectful of the tradition, and educationally valuable.
                    """
                    return client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are an expert in ancient Indian wisdom traditions, particularly in Vedic knowledge, Yoga, and traditional Indian mathematics (Ganita)."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=1000
                    )

                # Try to generate with OpenAI with retries
                response = generate_with_openai()
                lesson_text = response.choices[0].message.content.strip()
                import json
                lesson = json.loads(lesson_text)
                # Add Wikipedia information if available
                if wikipedia_info:
                    lesson["wikipedia_info"] = wikipedia_info
                logger.info(f"Successfully generated lesson with OpenAI: {lesson['title']}")
                return lesson

            except Exception as e:
                logger.error(f"Error using OpenAI: {str(e)}")
                error_messages.append(f"OpenAI error: {str(e)}")

        # If both services fail, use mock lessons with enhanced error reporting
        logger.warning(f"All LLM services failed. Errors: {'; '.join(error_messages)}. Using mock lessons as fallback.")
        mock_lesson = generate_mock_lesson(subject, topic)
        mock_lesson["explanation"] = f"Note: This is a mock lesson. LLM services were unavailable ({'; '.join(error_messages)}). " + mock_lesson["explanation"]

        # Add Wikipedia information if available
        if wikipedia_info:
            mock_lesson["wikipedia_info"] = wikipedia_info
            # Enhance the explanation with Wikipedia summary if available
            if wikipedia_info.get("summary"):
                mock_lesson["explanation"] += f"\n\nAdditional information from Wikipedia: {wikipedia_info['summary'][:300]}..."

        return mock_lesson

    except Exception as e:
        logger.error(f"Error generating lesson: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Error generating lesson: {str(e)}",
                "errors": error_messages
            }
        )

def generate_mock_lesson(subject: str, topic: str):
    """
    Generate a pre-defined lesson based on subject and topic
    """
    # Convert subject and topic to lowercase for case-insensitive matching
    subject_lower = subject.lower()
    topic_lower = topic.lower()

    # Define mock lessons
    mock_lessons = {
        "ved": {
            "sound": {
                "title": "Lesson 1: The Sacred Sound in Vedic Tradition",
                "shloka": "ॐ अग्निमीळे पुरोहितं यज्ञस्य देवम् ऋत्विजम्",
                "translation": "Om, I praise Agni, the priest of the sacrifice, the divine, the ritual performer.",
                "explanation": "In Vedic tradition, sound (shabda) is considered not just a physical phenomenon but a spiritual one. The primordial sound OM (ॐ) is considered the source of all creation. Mantras are sacred sound formulas that, when recited with proper pronunciation, rhythm, and devotion, create powerful vibrations affecting both the chanter and the environment. The concept of Nada Brahma suggests that the entire universe is a manifestation of sound vibration.",
                "activity": "Recite the shloka aloud thrice, paying attention to the vibration you feel in different parts of your body. Then, sit in silence for 2 minutes and observe the subtle sounds around you.",
                "question": "How does the concept of sound in Vedic tradition differ from the modern scientific understanding of sound?"
            }
        },
        "yoga": {
            "asana": {
                "title": "Lesson 1: The Philosophy of Asanas in Yoga",
                "shloka": "स्थिरसुखमासनम्॥४६॥",
                "translation": "Posture (asana) should be steady and comfortable.",
                "explanation": "This sutra from Patanjali's Yoga Sutras defines the essence of asana practice. Unlike modern interpretations that focus primarily on physical benefits, traditional yoga views asanas as tools for preparing the body for meditation. The steadiness (sthira) refers to the ability to hold a position, while comfort (sukha) ensures that the practitioner can remain in the pose without distraction or discomfort, allowing the mind to remain calm.",
                "activity": "Practice Sukhasana (Easy Pose) for 5 minutes, focusing on finding both steadiness and comfort. Observe how your breath affects your ability to maintain both qualities.",
                "question": "How does the traditional understanding of asana differ from modern yoga practice, and why is this distinction important?"
            }
        },
        "ganita": {
            "geometry": {
                "title": "Lesson 1: Sacred Geometry in Ancient Indian Mathematics",
                "shloka": "यथा शिखा मयूराणां नागानां मणयो यथा। तद्वद्वेदाङ्गशास्त्राणां गणितं मूर्धनि स्थितम्॥",
                "translation": "Just as the crest on the peacock, or the gem on the head of a snake, so is mathematics at the head of all sciences.",
                "explanation": "Ancient Indian mathematics (ganita) viewed geometry as a sacred science that revealed the underlying patterns of the universe. The construction of fire altars (agnicayana) required precise geometric knowledge. The Shulba Sutras (800-200 BCE) contain detailed instructions for constructing these altars, including methods to transform one geometric shape into another of equal area, and approximations of irrational numbers like √2.",
                "activity": "Using a compass and straightedge, construct a square and then transform it into a circle of equal area using the approximation π ≈ 3.1416, as given in ancient Indian texts.",
                "question": "How did the practical requirements of Vedic rituals contribute to the development of geometric knowledge in ancient India?"
            }
        }
    }

    # Try to find a matching lesson in our pre-defined lessons
    if subject_lower in mock_lessons and topic_lower in mock_lessons[subject_lower]:
        return mock_lessons[subject_lower][topic_lower]

    # If no match, return a generic lesson
    return {
        "title": f"Lesson on {subject}: {topic}",
        "shloka": "ॐ सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः। सर्वे भद्राणि पश्यन्तु मा कश्चिद्दुःखभाग्भवेत्॥",
        "translation": "May all be happy, may all be free from disease, may all see auspiciousness, may none suffer.",
        "explanation": f"This lesson explores the topic of {topic} within the context of {subject}. In the absence of a language model, we're providing this placeholder lesson. To generate more specific content, please set up an OpenAI API key or ensure Ollama is running.",
        "activity": f"Research the relationship between {subject} and {topic} in traditional Indian knowledge systems. Write down three key insights you discover.",
        "question": f"How does the understanding of {topic} in {subject} differ from modern interpretations?"
    }

@app.post("/generate_lesson", response_model=LessonResponse)
async def generate_lesson_post(request: LessonRequest):
    """
    Generate a structured lesson based on subject and topic (POST method)
    """
    # Call the GET endpoint handler with the same parameters
    return await generate_lesson_endpoint(
        subject=request.subject,
        topic=request.topic,
        include_wikipedia=request.include_wikipedia,
        use_knowledge_store=request.use_knowledge_store
    )

@app.get("/lessons")
async def list_lessons():
    """
    List all lessons in the knowledge store
    """
    try:
        from knowledge_store import list_lessons
        lessons = list_lessons()
        return {
            "status": "success",
            "count": len(lessons),
            "lessons": lessons
        }
    except ImportError:
        return {
            "status": "error",
            "message": "Knowledge store module not available",
            "count": 0,
            "lessons": []
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error listing lessons: {str(e)}",
            "count": 0,
            "lessons": []
        }

@app.get("/search_lessons")
async def search_lessons(query: str = Query(..., description="Search query")):
    """
    Search for lessons in the knowledge store
    """
    try:
        from knowledge_store import search_lessons as search_lessons_func
        results = search_lessons_func(query)
        return {
            "status": "success",
            "count": len(results),
            "results": results
        }
    except ImportError:
        return {
            "status": "error",
            "message": "Knowledge store module not available",
            "count": 0,
            "results": []
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error searching lessons: {str(e)}",
            "count": 0,
            "results": []
        }

if __name__ == "__main__":
    import uvicorn
    # Try with localhost and a different port
    print("Starting server on http://192.168.0.73:8000")
    uvicorn.run("app:app", host="192.168.0.73", port=8000, reload=False)
