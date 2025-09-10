"""Debug script to check query processing issues."""

import requests
import json

def check_system_status():
    """Check if the system is properly initialized."""
    try:
        response = requests.get("http://localhost:5000/api/status")
        if response.status_code == 200:
            status = response.json()
            print("System Status:")
            print(json.dumps(status, indent=2))
            return status
        else:
            print(f"Status check failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"Failed to check status: {e}")
        return None

def check_files():
    """Check uploaded files."""
    try:
        response = requests.get("http://localhost:5000/api/files")
        if response.status_code == 200:
            files = response.json()
            print("\nUploaded Files:")
            print(json.dumps(files, indent=2))
            return files
        else:
            print(f"Files check failed: {response.status_code}")
            return None
    except Exception as e:
        print(f"Failed to check files: {e}")
        return None

def test_query_with_debug():
    """Test query with specific file_id."""
    file_id = "eb342bd4-68be-4613-b3a4-1f2e7c1d6228"
    query = "这个文件内容主要是讲什么的"
    
    payload = {
        "file_id": file_id,
        "query": query
    }
    
    try:
        print(f"\nTesting query with file_id: {file_id}")
        print(f"Query: {query}")
        
        response = requests.post(
            "http://localhost:5000/api/query",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"\nResponse Status: {response.status_code}")
        print("Response Body:")
        
        if response.headers.get('content-type', '').startswith('application/json'):
            result = response.json()
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(response.text)
            
        return response
        
    except Exception as e:
        print(f"Query test failed: {e}")
        return None

if __name__ == "__main__":
    print("=" * 60)
    print("Debug Query Processing Issues")
    print("=" * 60)
    
    # Check system status
    status = check_system_status()
    
    # Check uploaded files
    files = check_files()
    
    # Test the query
    response = test_query_with_debug()
    
    print("\n" + "=" * 60)
    print("Debug Summary:")
    if status:
        print(f"- Agent Available: {status.get('agent_available', False)}")
        print(f"- Orchestrator Ready: {status.get('orchestrator_ready', False)}")
        print(f"- MCP Initialized: {status.get('mcp_initialized', False)}")
    
    if files and 'files' in files:
        print(f"- Total Files: {len(files['files'])}")
        target_file = None
        for f in files['files']:
            if f['file_id'] == 'eb342bd4-68be-4613-b3a4-1f2e7c1d6228':
                target_file = f
                break
        if target_file:
            print(f"- Target File Found: {target_file['filename']}")
            print(f"- File Processed: {target_file.get('processed', False)}")
        else:
            print("- Target File NOT Found")
    
    print("=" * 60)