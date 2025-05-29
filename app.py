"""
app.py - FastAPI application for serving the Gurukul Lesson Generator
"""

import os
import logging
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, Query, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import time
from dotenv import load_dotenv
import sys
import uvicorn
import uuid
from datetime import datetime, timedelta
from enum import Enum

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

class CreateLessonRequest(BaseModel):
    subject: str
    topic: str
    user_id: str
    include_wikipedia: bool = True
    force_regenerate: bool = True  # Changed default to True for dynamic generation

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

# New models for async lesson generation
class GenerationStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"

class LessonGenerationTask(BaseModel):
    task_id: str
    subject: str
    topic: str
    user_id: str
    status: GenerationStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    include_wikipedia: bool = True

class LessonGenerationResponse(BaseModel):
    task_id: str
    status: GenerationStatus
    message: str
    estimated_completion_time: Optional[str] = None
    poll_url: str

class LessonStatusResponse(BaseModel):
    task_id: str
    status: GenerationStatus
    progress_message: str
    created_at: datetime
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    lesson_data: Optional[LessonResponse] = None

# Global storage for generation tasks (in production, use Redis or database)
generation_tasks: Dict[str, LessonGenerationTask] = {}
generation_results: Dict[str, Dict[str, Any]] = {}

# Background task function for lesson generation
async def generate_lesson_background(task_id: str, subject: str, topic: str, user_id: str, include_wikipedia: bool = True):
    """
    Background task to generate a lesson asynchronously
    """
    try:
        # Update task status to in_progress
        if task_id in generation_tasks:
            generation_tasks[task_id].status = GenerationStatus.IN_PROGRESS
            logger.info(f"Starting background generation for task {task_id}: {subject}/{topic}")

        # Generate the lesson using enhanced lesson generator with force_fresh=True for dynamic generation
        try:
            from generate_lesson_enhanced import create_enhanced_lesson
            generated_lesson = create_enhanced_lesson(subject, topic)
        except ImportError:
            # Enhanced module is not available
            logger.error("Enhanced lesson generator module not available")
            raise Exception("Enhanced lesson generator module not available")
        except ValueError as e:
            # No content sources available (Ollama not working, no Wikipedia content)
            logger.error(f"Unable to generate lesson content: {str(e)}")
            raise Exception(f"Unable to generate lesson content: {str(e)}")
        except Exception as e:
            # Other generation errors
            logger.error(f"Error in lesson generation: {str(e)}")
            raise Exception(f"Error in lesson generation: {str(e)}")

        # Add user information to the generated lesson
        if isinstance(generated_lesson, dict):
            generated_lesson["created_by"] = user_id
            generated_lesson["generation_method"] = "async_background"
            generated_lesson["task_id"] = task_id

        # Store the result
        generation_results[task_id] = generated_lesson

        # Update task status to completed
        if task_id in generation_tasks:
            generation_tasks[task_id].status = GenerationStatus.COMPLETED
            generation_tasks[task_id].completed_at = datetime.now()
            logger.info(f"Completed background generation for task {task_id}: {generated_lesson.get('title', 'Untitled')}")

    except Exception as e:
        logger.error(f"Error in background generation for task {task_id}: {str(e)}")

        # Update task status to failed
        if task_id in generation_tasks:
            generation_tasks[task_id].status = GenerationStatus.FAILED
            generation_tasks[task_id].error_message = str(e)
            generation_tasks[task_id].completed_at = datetime.now()

# Cleanup function to remove old completed tasks
def cleanup_old_tasks():
    """Remove tasks older than 1 hour to prevent memory leaks"""
    cutoff_time = datetime.now() - timedelta(hours=1)
    tasks_to_remove = []

    for task_id, task in generation_tasks.items():
        if task.completed_at and task.completed_at < cutoff_time:
            tasks_to_remove.append(task_id)

    for task_id in tasks_to_remove:
        generation_tasks.pop(task_id, None)
        generation_results.pop(task_id, None)
        logger.info(f"Cleaned up old task: {task_id}")

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
            "get_lesson": "/lessons/{subject}/{topic} - Retrieve existing lesson",
            "create_lesson_async": "/lessons - Create new lesson (POST, async)",
            "check_generation_status": "/lessons/status/{task_id} - Check lesson generation status",
            "list_active_tasks": "/lessons/tasks - List active generation tasks",
            "search_lessons": "/search_lessons?query=sound",
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

@app.get("/lessons/{subject}/{topic}", response_model=LessonResponse)
async def get_lesson_endpoint(
    subject: str,
    topic: str
):
    """
    Retrieve an existing lesson from the knowledge store

    This endpoint only retrieves existing lessons and does not generate new content.
    Use POST /lessons to create new lessons.

    Args:
        subject: Subject of the lesson (e.g., ved, ganita, yoga)
        topic: Topic of the lesson (e.g., sound, algebra, asana)

    Returns:
        LessonResponse: The existing lesson data

    Raises:
        404: If lesson is not found in knowledge store
        500: If there's an error retrieving the lesson

    Example usage:
        GET /lessons/ved/sound
    """
    try:
        logger.info(f"Retrieving lesson for subject: {subject}, topic: {topic}")

        # Try to get lesson from knowledge store
        try:
            from knowledge_store import get_lesson
            stored_lesson = get_lesson(subject, topic)

            if not stored_lesson:
                raise HTTPException(
                    status_code=404,
                    detail={
                        "message": f"Lesson not found for subject '{subject}' and topic '{topic}'",
                        "suggestion": "Use POST /lessons to create a new lesson",
                        "available_endpoints": {
                            "create_lesson": "POST /lessons",
                            "search_lessons": "GET /search_lessons?query=your_search"
                        }
                    }
                )

            logger.info(f"Successfully retrieved lesson: {stored_lesson.get('title', 'Untitled')}")

            return stored_lesson

        except ImportError:
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Knowledge store module not available",
                    "error": "Cannot retrieve lessons - knowledge store is not configured"
                }
            )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error retrieving lesson: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Error retrieving lesson: {str(e)}",
                "subject": subject,
                "topic": topic
            }
        )







@app.post("/lessons", response_model=LessonGenerationResponse)
async def create_lesson_endpoint(request: CreateLessonRequest, background_tasks: BackgroundTasks):
    """
    Create a new lesson by generating content using AI models (Async)

    This endpoint starts lesson generation in the background and returns immediately.
    The lesson generation happens asynchronously to prevent timeout issues.
    Use the returned task_id to poll for completion status.

    Args:
        request: CreateLessonRequest containing:
            - subject: Subject of the lesson (e.g., ved, ganita, yoga)
            - topic: Topic of the lesson (e.g., sound, algebra, asana)
            - user_id: ID of the user creating the lesson
            - include_wikipedia: Whether to include Wikipedia information
            - force_regenerate: Always True for dynamic generation

    Returns:
        LessonGenerationResponse: Task information for polling status

    Example usage:
        POST /lessons
        {
            "subject": "english",
            "topic": "verbs",
            "user_id": "user123",
            "include_wikipedia": true
        }
    """
    try:
        # Validate required parameters
        if not request.subject or not request.topic or not request.user_id:
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Subject, topic, and user_id are required",
                    "example": {
                        "subject": "english",
                        "topic": "verbs",
                        "user_id": "user123"
                    },
                    "available_subjects": ["ved", "ganita", "yoga", "ayurveda", "english", "maths"],
                    "example_topics": ["sound", "algebra", "asana", "doshas", "verbs", "geometry"]
                }
            )

        # Clean up old tasks periodically
        cleanup_old_tasks()

        # Generate unique task ID
        task_id = str(uuid.uuid4())

        # Create task record
        task = LessonGenerationTask(
            task_id=task_id,
            subject=request.subject,
            topic=request.topic,
            user_id=request.user_id,
            status=GenerationStatus.PENDING,
            created_at=datetime.now(),
            include_wikipedia=request.include_wikipedia
        )

        # Store task
        generation_tasks[task_id] = task

        # Start background generation
        background_tasks.add_task(
            generate_lesson_background,
            task_id=task_id,
            subject=request.subject,
            topic=request.topic,
            user_id=request.user_id,
            include_wikipedia=request.include_wikipedia
        )

        logger.info(f"Started async lesson generation - Task ID: {task_id}, Subject: {request.subject}, Topic: {request.topic}")

        # Return task information
        return LessonGenerationResponse(
            task_id=task_id,
            status=GenerationStatus.PENDING,
            message=f"Lesson generation started for {request.subject}/{request.topic}",
            estimated_completion_time="30-60 seconds",
            poll_url=f"/lessons/status/{task_id}"
        )

    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Error starting lesson generation: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Error starting lesson generation: {str(e)}",
                "subject": request.subject if hasattr(request, 'subject') else 'unknown',
                "topic": request.topic if hasattr(request, 'topic') else 'unknown'
            }
        )

@app.get("/lessons/status/{task_id}", response_model=LessonStatusResponse)
async def get_lesson_generation_status(task_id: str):
    """
    Get the status of a lesson generation task

    Args:
        task_id: The unique task identifier returned from POST /lessons

    Returns:
        LessonStatusResponse: Current status and lesson data if completed
    """
    try:
        # Check if task exists
        if task_id not in generation_tasks:
            raise HTTPException(
                status_code=404,
                detail={
                    "message": f"Task {task_id} not found",
                    "suggestion": "The task may have expired or the task_id is invalid"
                }
            )

        task = generation_tasks[task_id]

        # Prepare response based on status
        if task.status == GenerationStatus.COMPLETED:
            # Get the generated lesson data
            lesson_data = generation_results.get(task_id)
            lesson_response = None

            if lesson_data:
                # Convert to LessonResponse format
                lesson_response = LessonResponse(
                    title=lesson_data.get("title", ""),
                    shloka=lesson_data.get("shloka", ""),
                    translation=lesson_data.get("translation", ""),
                    explanation=lesson_data.get("explanation", ""),
                    activity=lesson_data.get("activity", ""),
                    question=lesson_data.get("question", ""),
                    wikipedia_info=lesson_data.get("wikipedia_info")
                )

            return LessonStatusResponse(
                task_id=task_id,
                status=task.status,
                progress_message="Lesson generation completed successfully",
                created_at=task.created_at,
                completed_at=task.completed_at,
                lesson_data=lesson_response
            )

        elif task.status == GenerationStatus.FAILED:
            return LessonStatusResponse(
                task_id=task_id,
                status=task.status,
                progress_message="Lesson generation failed",
                created_at=task.created_at,
                completed_at=task.completed_at,
                error_message=task.error_message
            )

        elif task.status == GenerationStatus.IN_PROGRESS:
            return LessonStatusResponse(
                task_id=task_id,
                status=task.status,
                progress_message="Lesson generation is in progress...",
                created_at=task.created_at
            )

        else:  # PENDING
            return LessonStatusResponse(
                task_id=task_id,
                status=task.status,
                progress_message="Lesson generation is queued and will start shortly",
                created_at=task.created_at
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error retrieving task status: {str(e)}"
        )

@app.get("/lessons/tasks")
async def list_active_generation_tasks():
    """
    List all active lesson generation tasks

    Returns:
        Dict: Information about all active generation tasks
    """
    try:
        # Clean up old tasks first
        cleanup_old_tasks()

        active_tasks = []
        for task_id, task in generation_tasks.items():
            active_tasks.append({
                "task_id": task_id,
                "subject": task.subject,
                "topic": task.topic,
                "user_id": task.user_id,
                "status": task.status.value,
                "created_at": task.created_at.isoformat(),
                "completed_at": task.completed_at.isoformat() if task.completed_at else None,
                "error_message": task.error_message
            })

        return {
            "status": "success",
            "total_tasks": len(active_tasks),
            "tasks": active_tasks,
            "status_counts": {
                "pending": len([t for t in active_tasks if t["status"] == "pending"]),
                "in_progress": len([t for t in active_tasks if t["status"] == "in_progress"]),
                "completed": len([t for t in active_tasks if t["status"] == "completed"]),
                "failed": len([t for t in active_tasks if t["status"] == "failed"])
            }
        }

    except Exception as e:
        logger.error(f"Error listing active tasks: {str(e)}")
        return {
            "status": "error",
            "message": f"Error listing active tasks: {str(e)}",
            "total_tasks": 0,
            "tasks": []
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
    uvicorn.run("app:app", host="192.168.0.70", port=8000, reload=False)
