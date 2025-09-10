"""Test script for Excel preview functionality."""

import pandas as pd
import requests
import json
from pathlib import Path

# Create a sample Excel file for testing
def create_test_excel():
    """Create a test Excel file with sample data."""
    
    # Sample data
    data = {
        'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones'],
        'Price': [999.99, 25.50, 75.00, 299.99, 149.99],
        'Quantity': [10, 50, 30, 15, 25],
        'Category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Accessories'],
        'In Stock': [True, True, False, True, True]
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Create Excel file with multiple sheets
    test_file = Path('test_data.xlsx')
    
    with pd.ExcelWriter(test_file, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='Products', index=False)
        
        # Add another sheet with more data
        df2 = pd.DataFrame({
            'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May'],
            'Sales': [1000, 1500, 1200, 1800, 2000],
            'Profit': [200, 300, 240, 360, 400]
        })
        df2.to_excel(writer, sheet_name='Sales', index=False)
    
    return str(test_file)

def test_excel_to_html():
    """Test the Excel to HTML conversion directly."""
    print("Creating test Excel file...")
    test_file = create_test_excel()
    
    try:
        # Test the utility function directly
        from backend.utils.excel_to_html import excel_to_html, get_excel_info
        
        print(f"Testing with file: {test_file}")
        
        # Test basic conversion
        print("\n=== Testing basic Excel to HTML conversion ===")
        result = excel_to_html(test_file)
        
        if result['success']:
            print("SUCCESS: Conversion successful!")
            print(f"Metadata: {result['metadata']}")
            print(f"HTML length: {len(result['html'])} characters")
            print("First 200 characters of HTML:")
            print(result['html'][:200] + "...")
        else:
            print(f"ERROR: Conversion failed: {result['error']}")
        
        # Test file info
        print("\n=== Testing Excel file info ===")
        info_result = get_excel_info(test_file)
        
        if info_result['success']:
            print("SUCCESS: File info retrieval successful!")
            print(f"File info: {json.dumps(info_result['info'], indent=2)}")
        else:
            print(f"ERROR: File info failed: {info_result['error']}")
        
        # Test specific sheet
        print("\n=== Testing specific sheet conversion ===")
        result2 = excel_to_html(test_file, sheet_name='Sales')
        
        if result2['success']:
            print("SUCCESS: Sheet-specific conversion successful!")
            print(f"Sheet metadata: {result2['metadata']}")
        else:
            print(f"ERROR: Sheet conversion failed: {result2['error']}")
            
    except Exception as e:
        print(f"ERROR: Test failed with exception: {e}")
    finally:
        # Clean up
        if Path(test_file).exists():
            Path(test_file).unlink()
            print(f"\n🧹 Cleaned up test file: {test_file}")

def test_api_endpoints():
    """Test the API endpoints (requires server to be running)."""
    base_url = "http://localhost:5000"
    
    print("Testing API endpoints...")
    print("Note: This requires the Flask server to be running on localhost:5000")
    
    try:
        # Test status endpoint
        response = requests.get(f"{base_url}/api/status")
        if response.status_code == 200:
            print("SUCCESS: Status endpoint working")
        else:
            print(f"ERROR: Status endpoint failed: {response.status_code}")
            
    except requests.ConnectionError:
        print("ERROR: Cannot connect to Flask server. Please start the server first.")
        print("Run: python backend/app.py")

if __name__ == "__main__":
    print("Testing Excel Preview Functionality")
    print("=" * 50)
    
    test_excel_to_html()
    
    print("\n" + "=" * 50)
    test_api_endpoints()
    
    print("\nTesting complete!")