#!/usr/bin/env python3
"""
Test script for the refactored lesson generation API endpoints
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_root_endpoint():
    """Test the root endpoint to see updated API structure"""
    print("🔍 Testing root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("✅ Root endpoint working")
        print("📋 Available endpoints:")
        for key, value in data.get("endpoints", {}).items():
            print(f"  - {key}: {value}")
    else:
        print("❌ Root endpoint failed")
    print()

def test_get_existing_lesson():
    """Test GET endpoint for retrieving existing lessons"""
    print("🔍 Testing GET /lessons/{subject}/{topic} - Existing lesson...")
    response = requests.get(f"{BASE_URL}/lessons/Veda/Sound")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("✅ Successfully retrieved existing lesson")
        print(f"📖 Title: {data.get('title', 'N/A')}")
        print(f"🕉️  Shloka: {data.get('shloka', 'N/A')[:50]}...")
        print(f"📚 Has Wikipedia info: {'Yes' if data.get('wikipedia_info') else 'No'}")
    else:
        print("❌ Failed to retrieve existing lesson")
        print(f"Error: {response.text}")
    print()

def test_get_nonexistent_lesson():
    """Test GET endpoint for non-existent lessons (should return 404)"""
    print("🔍 Testing GET /lessons/{subject}/{topic} - Non-existent lesson...")
    response = requests.get(f"{BASE_URL}/lessons/nonexistent/topic")
    print(f"Status: {response.status_code}")
    if response.status_code == 404:
        data = response.json()
        print("✅ Correctly returned 404 for non-existent lesson")
        print(f"💡 Suggestion: {data.get('detail', {}).get('suggestion', 'N/A')}")
    else:
        print("❌ Expected 404 but got different status")
    print()

def test_create_new_lesson():
    """Test POST endpoint for creating new lessons"""
    print("🔍 Testing POST /lessons - Create new lesson...")
    
    # Use a unique topic to avoid conflicts
    unique_topic = f"test_topic_{int(time.time())}"
    
    payload = {
        "subject": "yoga",
        "topic": unique_topic,
        "user_id": "test_user_123",
        "include_wikipedia": True,
        "force_regenerate": False
    }
    
    response = requests.post(
        f"{BASE_URL}/lessons",
        headers={"Content-Type": "application/json"},
        json=payload
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("✅ Successfully created new lesson")
        print(f"📖 Title: {data.get('title', 'N/A')}")
        print(f"👤 Created by: {data.get('created_by', 'N/A')}")
        print(f"🔧 Generation method: {data.get('generation_method', 'N/A')}")
        return unique_topic
    else:
        print("❌ Failed to create new lesson")
        print(f"Error: {response.text}")
        return None
    print()

def test_create_duplicate_lesson(topic):
    """Test POST endpoint conflict detection"""
    if not topic:
        print("⏭️  Skipping duplicate test - no topic from previous test")
        return
        
    print("🔍 Testing POST /lessons - Duplicate lesson (should return 409)...")
    
    payload = {
        "subject": "yoga",
        "topic": topic,
        "user_id": "test_user_456",
        "include_wikipedia": True,
        "force_regenerate": False
    }
    
    response = requests.post(
        f"{BASE_URL}/lessons",
        headers={"Content-Type": "application/json"},
        json=payload
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 409:
        data = response.json()
        print("✅ Correctly returned 409 for duplicate lesson")
        print(f"💡 Suggestion: {data.get('detail', {}).get('suggestion', 'N/A')}")
    else:
        print("❌ Expected 409 but got different status")
        print(f"Response: {response.text}")
    print()

def test_force_regenerate(topic):
    """Test POST endpoint with force_regenerate=True"""
    if not topic:
        print("⏭️  Skipping force regenerate test - no topic from previous test")
        return
        
    print("🔍 Testing POST /lessons - Force regenerate existing lesson...")
    
    payload = {
        "subject": "yoga",
        "topic": topic,
        "user_id": "test_user_789",
        "include_wikipedia": True,
        "force_regenerate": True
    }
    
    response = requests.post(
        f"{BASE_URL}/lessons",
        headers={"Content-Type": "application/json"},
        json=payload
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("✅ Successfully force-regenerated lesson")
        print(f"👤 Created by: {data.get('created_by', 'N/A')}")
    else:
        print("❌ Failed to force-regenerate lesson")
        print(f"Error: {response.text}")
    print()

def test_legacy_endpoints():
    """Test that legacy endpoints still work but show deprecation"""
    print("🔍 Testing legacy GET /generate_lesson...")
    response = requests.get(f"{BASE_URL}/generate_lesson?subject=ved&topic=sound")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print("✅ Legacy GET endpoint still works")
    else:
        print("❌ Legacy GET endpoint failed")
    print()

def main():
    """Run all tests"""
    print("🚀 Testing Refactored Lesson Generation API")
    print("=" * 50)
    
    try:
        # Test basic functionality
        test_root_endpoint()
        test_get_existing_lesson()
        test_get_nonexistent_lesson()
        
        # Test lesson creation and conflict detection
        created_topic = test_create_new_lesson()
        test_create_duplicate_lesson(created_topic)
        test_force_regenerate(created_topic)
        
        # Test legacy endpoints
        test_legacy_endpoints()
        
        print("🎉 All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to the API server.")
        print("Make sure the server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
