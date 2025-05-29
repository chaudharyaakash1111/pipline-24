# Async Lesson Generation API

This document describes the new asynchronous lesson generation system that prevents timeout issues and supports dynamic lesson creation.

## Overview

The new system provides:
- **Asynchronous lesson generation** to prevent frontend timeouts
- **Dynamic lesson creation** - each request generates fresh, unique content
- **Status tracking** for monitoring generation progress
- **Automatic versioning** of lessons in the knowledge store
- **Background task management** with cleanup

## Key Features

### 1. Dynamic Generation
- Every POST request to `/lessons` generates completely new content
- No more conflicts with existing lessons
- Automatic versioning (1.0, 1.1, 1.2, etc.)
- Latest lesson always overwrites previous versions

### 2. Async Processing
- POST requests return immediately with a task ID
- Generation happens in the background
- Frontend can poll for completion status
- Prevents timeout issues during long generation processes

### 3. Status Tracking
- Real-time status updates (pending → in_progress → completed/failed)
- Detailed progress messages
- Error handling and reporting
- Automatic cleanup of old tasks

## API Endpoints

### 1. Start Lesson Generation (POST /lessons)

**Request:**
```json
{
    "subject": "english",
    "topic": "verbs", 
    "user_id": "user123",
    "include_wikipedia": true
}
```

**Response:**
```json
{
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "pending",
    "message": "Lesson generation started for english/verbs",
    "estimated_completion_time": "30-60 seconds",
    "poll_url": "/lessons/status/550e8400-e29b-41d4-a716-446655440000"
}
```

### 2. Check Generation Status (GET /lessons/status/{task_id})

**Response (In Progress):**
```json
{
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "in_progress",
    "progress_message": "Lesson generation is in progress...",
    "created_at": "2025-05-29T14:55:20.885000",
    "completed_at": null,
    "error_message": null,
    "lesson_data": null
}
```

**Response (Completed):**
```json
{
    "task_id": "550e8400-e29b-41d4-a716-446655440000",
    "status": "completed",
    "progress_message": "Lesson generation completed successfully",
    "created_at": "2025-05-29T14:55:20.885000",
    "completed_at": "2025-05-29T14:56:15.123000",
    "error_message": null,
    "lesson_data": {
        "title": "Understanding English Verbs",
        "shloka": "...",
        "translation": "...",
        "explanation": "...",
        "activity": "...",
        "question": "...",
        "wikipedia_info": {...}
    }
}
```

### 3. Retrieve Latest Lesson (GET /lessons/{subject}/{topic})

Returns the most recently generated lesson from the knowledge store.

### 4. List Active Tasks (GET /lessons/tasks)

**Response:**
```json
{
    "status": "success",
    "total_tasks": 2,
    "tasks": [
        {
            "task_id": "550e8400-e29b-41d4-a716-446655440000",
            "subject": "english",
            "topic": "verbs",
            "user_id": "user123",
            "status": "completed",
            "created_at": "2025-05-29T14:55:20.885000",
            "completed_at": "2025-05-29T14:56:15.123000",
            "error_message": null
        }
    ],
    "status_counts": {
        "pending": 0,
        "in_progress": 1,
        "completed": 1,
        "failed": 0
    }
}
```

## Usage Examples

### Frontend Implementation

```javascript
// 1. Start lesson generation
async function generateLesson(subject, topic, userId) {
    const response = await fetch('/lessons', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            subject: subject,
            topic: topic,
            user_id: userId,
            include_wikipedia: true
        })
    });
    
    const result = await response.json();
    return result.task_id;
}

// 2. Poll for completion
async function pollForCompletion(taskId) {
    while (true) {
        const response = await fetch(`/lessons/status/${taskId}`);
        const status = await response.json();
        
        if (status.status === 'completed') {
            return status.lesson_data;
        } else if (status.status === 'failed') {
            throw new Error(status.error_message);
        }
        
        // Wait 3 seconds before next poll
        await new Promise(resolve => setTimeout(resolve, 3000));
    }
}

// 3. Complete workflow
async function createLessonAsync(subject, topic, userId) {
    try {
        const taskId = await generateLesson(subject, topic, userId);
        const lessonData = await pollForCompletion(taskId);
        return lessonData;
    } catch (error) {
        console.error('Lesson generation failed:', error);
        throw error;
    }
}
```

### Python Client

```python
import requests
import time

def generate_lesson_async(subject, topic, user_id, base_url="http://192.168.0.70:8000"):
    # Start generation
    response = requests.post(f"{base_url}/lessons", json={
        "subject": subject,
        "topic": topic,
        "user_id": user_id,
        "include_wikipedia": True
    })
    
    task_id = response.json()["task_id"]
    
    # Poll for completion
    while True:
        status_response = requests.get(f"{base_url}/lessons/status/{task_id}")
        status_data = status_response.json()
        
        if status_data["status"] == "completed":
            return status_data["lesson_data"]
        elif status_data["status"] == "failed":
            raise Exception(status_data["error_message"])
        
        time.sleep(3)
```

## Migration Guide

### From Synchronous to Asynchronous

**Old synchronous approach:**
```python
# This would timeout for long generations
response = requests.post("/lessons", json=lesson_request)
lesson_data = response.json()  # Could timeout
```

**New asynchronous approach:**
```python
# Start generation (returns immediately)
response = requests.post("/lessons", json=lesson_request)
task_id = response.json()["task_id"]

# Poll for completion
while True:
    status = requests.get(f"/lessons/status/{task_id}").json()
    if status["status"] == "completed":
        lesson_data = status["lesson_data"]
        break
    time.sleep(3)
```

## Configuration

### Task Cleanup
- Tasks older than 1 hour are automatically cleaned up
- Prevents memory leaks in long-running applications
- Configurable in the `cleanup_old_tasks()` function

### Generation Settings
- Default estimated completion time: 30-60 seconds
- Automatic retry logic for failed generations
- Wikipedia integration enabled by default

## Error Handling

### Common Error Scenarios

1. **Task Not Found (404)**
   - Task ID expired or invalid
   - Check if task was cleaned up

2. **Generation Failed (500)**
   - LLM service unavailable
   - Check error_message in status response

3. **Validation Error (400)**
   - Missing required fields (subject, topic, user_id)
   - Invalid parameter values

### Best Practices

1. **Always check task status** before assuming completion
2. **Implement timeout logic** in your polling (max 2-3 minutes)
3. **Handle failed generations gracefully** with retry logic
4. **Store task IDs** for later reference if needed

## Testing

Run the provided test script:
```bash
python test_async_lesson_generation.py
```

This will demonstrate the complete async workflow and verify all endpoints are working correctly.
