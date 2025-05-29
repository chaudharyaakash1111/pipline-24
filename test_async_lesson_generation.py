#!/usr/bin/env python3
"""
Test script for the new async lesson generation API

This script demonstrates how to use the new async lesson generation endpoints:
1. POST /lessons - Start lesson generation
2. GET /lessons/status/{task_id} - Check generation status
3. GET /lessons/{subject}/{topic} - Retrieve the final lesson

Usage:
    python test_async_lesson_generation.py
"""

import requests
import time
import json
from typing import Dict, Any

# Configuration
API_BASE_URL = "http://192.168.0.70:8000"
TEST_SUBJECT = "english"
TEST_TOPIC = "verbs"
TEST_USER_ID = "test_user_123"

def start_lesson_generation() -> Dict[str, Any]:
    """Start async lesson generation"""
    url = f"{API_BASE_URL}/lessons"
    payload = {
        "subject": TEST_SUBJECT,
        "topic": TEST_TOPIC,
        "user_id": TEST_USER_ID,
        "include_wikipedia": True,
        "force_regenerate": True
    }
    
    print(f"🚀 Starting lesson generation for {TEST_SUBJECT}/{TEST_TOPIC}...")
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Generation started successfully!")
        print(f"   Task ID: {result['task_id']}")
        print(f"   Status: {result['status']}")
        print(f"   Poll URL: {result['poll_url']}")
        print(f"   Estimated time: {result['estimated_completion_time']}")
        return result
    else:
        print(f"❌ Failed to start generation: {response.status_code}")
        print(f"   Error: {response.text}")
        return {}

def check_generation_status(task_id: str) -> Dict[str, Any]:
    """Check the status of lesson generation"""
    url = f"{API_BASE_URL}/lessons/status/{task_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"❌ Failed to check status: {response.status_code}")
        print(f"   Error: {response.text}")
        return {}

def poll_until_complete(task_id: str, max_wait_time: int = 120) -> Dict[str, Any]:
    """Poll the generation status until completion or timeout"""
    print(f"⏳ Polling for completion (max {max_wait_time}s)...")
    
    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        status_result = check_generation_status(task_id)
        
        if not status_result:
            break
            
        status = status_result.get("status", "unknown")
        progress_message = status_result.get("progress_message", "")
        
        print(f"   Status: {status} - {progress_message}")
        
        if status == "completed":
            print("✅ Generation completed successfully!")
            return status_result
        elif status == "failed":
            error_msg = status_result.get("error_message", "Unknown error")
            print(f"❌ Generation failed: {error_msg}")
            return status_result
        
        # Wait before next poll
        time.sleep(3)
    
    print(f"⏰ Timeout reached after {max_wait_time}s")
    return {}

def get_final_lesson() -> Dict[str, Any]:
    """Retrieve the final lesson from the knowledge store"""
    url = f"{API_BASE_URL}/lessons/{TEST_SUBJECT}/{TEST_TOPIC}"
    response = requests.get(url)
    
    if response.status_code == 200:
        print("✅ Successfully retrieved final lesson!")
        return response.json()
    else:
        print(f"❌ Failed to retrieve lesson: {response.status_code}")
        print(f"   Error: {response.text}")
        return {}

def list_active_tasks() -> Dict[str, Any]:
    """List all active generation tasks"""
    url = f"{API_BASE_URL}/lessons/tasks"
    response = requests.get(url)
    
    if response.status_code == 200:
        result = response.json()
        print(f"📋 Active tasks: {result['total_tasks']}")
        for task in result.get('tasks', []):
            print(f"   - {task['task_id'][:8]}... | {task['subject']}/{task['topic']} | {task['status']}")
        return result
    else:
        print(f"❌ Failed to list tasks: {response.status_code}")
        return {}

def main():
    """Main test function"""
    print("🧪 Testing Async Lesson Generation API")
    print("=" * 50)
    
    # Step 1: List current active tasks
    print("\n1. Checking active tasks...")
    list_active_tasks()
    
    # Step 2: Start lesson generation
    print("\n2. Starting lesson generation...")
    generation_result = start_lesson_generation()
    
    if not generation_result:
        print("❌ Cannot continue - failed to start generation")
        return
    
    task_id = generation_result.get("task_id")
    if not task_id:
        print("❌ No task ID received")
        return
    
    # Step 3: Poll until completion
    print(f"\n3. Polling for completion...")
    final_status = poll_until_complete(task_id)
    
    # Step 4: Check if lesson data is available in the status response
    if final_status.get("status") == "completed" and final_status.get("lesson_data"):
        print("\n4. Lesson data received in status response:")
        lesson_data = final_status["lesson_data"]
        print(f"   Title: {lesson_data.get('title', 'N/A')}")
        print(f"   Explanation length: {len(lesson_data.get('explanation', ''))} characters")
    
    # Step 5: Retrieve final lesson from knowledge store
    print(f"\n5. Retrieving lesson from knowledge store...")
    final_lesson = get_final_lesson()
    
    if final_lesson:
        print(f"   Title: {final_lesson.get('title', 'N/A')}")
        print(f"   Version: {final_lesson.get('metadata', {}).get('version', 'N/A')}")
        print(f"   Last updated: {final_lesson.get('metadata', {}).get('last_updated', 'N/A')}")
    
    # Step 6: List active tasks again
    print("\n6. Final active tasks check...")
    list_active_tasks()
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    main()
