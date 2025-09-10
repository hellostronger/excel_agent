"""Simple script to run the frontend Flask application."""

import os
import sys
from pathlib import Path

# Set UTF-8 encoding for Windows
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())

def main():
    """Run the frontend application."""
    print("🚀 启动Excel智能分析系统前端...")
    
    # Check if we're in the correct directory
    current_dir = Path.cwd()
    if current_dir.name != 'excel_agent':
        print("❌ 请在 excel_agent 目录下运行此脚本")
        sys.exit(1)
    
    # Check dependencies
    try:
        import flask
        import pandas
        import numpy
        print("✅ 核心依赖已安装")
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("请运行: pip install -r frontend_requirements.txt")
        sys.exit(1)
    
    # Set environment variables
    os.environ['FLASK_APP'] = 'backend/app.py'
    os.environ['FLASK_ENV'] = 'development'
    
    # Change to backend directory and run
    backend_dir = current_dir / 'backend'
    if not backend_dir.exists():
        print("❌ 找不到 backend 目录")
        sys.exit(1)
    
    os.chdir(backend_dir)
    
    print("📡 启动Flask服务器...")
    print("🌐 请在浏览器中访问: http://localhost:5000")
    print("⭐ 支持的文件格式: .xlsx, .xls, .xlsm")
    print("🔄 系统将自动检测是否有完整的Agent系统，如无则运行在演示模式")
    print("\n按 Ctrl+C 停止服务器\n")
    
    # Import and run the Flask app
    sys.path.insert(0, str(backend_dir))
    from app import app
    app.run(debug=False, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main()