#!/usr/bin/env python3
"""
Excel Intelligence Agent 运行脚本

使用方式:
python run_agent.py
"""

import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_environment():
    """设置环境变量"""
    env_file = project_root / ".env"
    if not env_file.exists():
        print("警告: .env 文件不存在，请复制 .env.example 并配置")
        print("cp .env.example .env")
        return False
    
    # 简单的环境变量加载
    with open(env_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip()
    
    return True

def main():
    """主函数"""
    print("=== Excel Intelligence Agent ===")
    print("多智能体Excel分析系统")
    
    # 设置环境
    if not setup_environment():
        return
    
    try:
        # 导入主Agent
        from excel_intelligence_agent.agent import excel_intelligence_agent
        
        print(f"\n✓ Agent初始化成功")
        print(f"  - Agent名称: {excel_intelligence_agent.name}")
        print(f"  - 描述: {excel_intelligence_agent.description}")
        
        # 显示可用工具
        if hasattr(excel_intelligence_agent, 'tools'):
            print(f"  - 可用工具: {len(excel_intelligence_agent.tools)}个")
            for tool in excel_intelligence_agent.tools[:3]:  # 显示前3个
                if hasattr(tool, '__name__'):
                    print(f"    * {tool.__name__}")
        
        print(f"\n✓ 系统就绪！")
        print(f"你可以通过以下方式使用:")
        print(f"1. 在Python中导入: from excel_intelligence_agent.agent import excel_intelligence_agent")
        print(f"2. 使用ADK CLI: python -m google.adk.cli")
        print(f"3. 创建会话并发送消息")
        
        # 交互模式
        print(f"\n=== 交互模式 ===")
        print(f"输入Excel文件路径开始分析，或输入'quit'退出")
        
        while True:
            user_input = input(f"\n请输入: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            
            if not user_input:
                continue
            
            try:
                # 这里需要根据ADK的实际API来调用
                print(f"正在分析: {user_input}")
                print(f"(注意: 实际的Agent调用需要根据ADK文档来实现)")
                
                # 示例调用 - 需要根据实际ADK API调整
                # session = excel_intelligence_agent.start_session()
                # response = session.send_message(f"分析文件: {user_input}")
                # print(f"回复: {response}")
                
            except Exception as e:
                print(f"错误: {e}")
    
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请检查:")
        print("1. 是否安装了所需依赖: poetry install 或 pip install google-adk")
        print("2. 是否正确配置了环境变量")
        print("3. 是否有网络连接")
    
    except Exception as e:
        print(f"运行错误: {e}")

if __name__ == "__main__":
    main()