"""
Excel智能分析系统 - 日志配置文件

这个配置文件允许你调整系统的日志级别和行为。

日志级别说明：
- DEBUG: 最详细的日志，包括模型输入输出、所有内部状态
- INFO: 一般信息，包括Agent执行步骤、API请求响应
- WARNING: 警告信息
- ERROR: 错误信息
- CRITICAL: 严重错误

日志文件位置：
- API日志: excel_agent_api.log
- Agent日志: tmp/excel_agent.log (由系统配置决定)
"""

import os
from pathlib import Path

# ================================
# 日志配置 - 用户可修改
# ================================

# 控制台日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
CONSOLE_LOG_LEVEL = "INFO"

# 文件日志级别 (通常设置为DEBUG以记录详细信息)
FILE_LOG_LEVEL = "DEBUG"

# 是否启用详细的模型调用日志 (True/False)
ENABLE_DETAILED_LLM_LOGS = True

# 是否启用API请求响应日志 (True/False)
ENABLE_API_LOGS = True

# 是否启用Agent执行详情日志 (True/False)
ENABLE_AGENT_LOGS = True

# 日志文件最大大小 (MB)
MAX_LOG_FILE_SIZE = 10

# 日志文件保留天数
LOG_RETENTION_DAYS = 7

# ================================
# 高级配置 - 谨慎修改
# ================================

# 日志格式
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"

# 彩色日志格式 (用于控制台)
COLORED_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan> | "
    "<level>{message}</level>"
)

# 需要静默的外部库日志
SUPPRESS_LOGGERS = [
    "httpx",
    "httpcore", 
    "urllib3",
    "requests"
]

# ================================
# 日志应用函数
# ================================

def apply_logging_config():
    """应用日志配置到系统"""
    import logging
    from loguru import logger
    
    # 设置标准库日志级别
    logging.getLogger().setLevel(getattr(logging, CONSOLE_LOG_LEVEL))
    
    # 静默外部库
    for logger_name in SUPPRESS_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    print(f"✅ 日志配置已应用:")
    print(f"   - 控制台日志级别: {CONSOLE_LOG_LEVEL}")
    print(f"   - 文件日志级别: {FILE_LOG_LEVEL}")
    print(f"   - 详细模型日志: {'启用' if ENABLE_DETAILED_LLM_LOGS else '禁用'}")
    print(f"   - API请求日志: {'启用' if ENABLE_API_LOGS else '禁用'}")
    print(f"   - Agent执行日志: {'启用' if ENABLE_AGENT_LOGS else '禁用'}")


# ================================
# 日志查看工具
# ================================

def tail_logs(lines=50):
    """查看最新的日志记录"""
    log_files = [
        "excel_agent_api.log",
        "tmp/excel_agent.log",
        Path.cwd() / "tmp" / "excel_agent.log"
    ]
    
    print("🔍 最新日志记录:")
    print("=" * 80)
    
    for log_file in log_files:
        if Path(log_file).exists():
            print(f"\n📄 {log_file}:")
            print("-" * 40)
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    all_lines = f.readlines()
                    recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines
                    for line in recent_lines:
                        print(line.rstrip())
            except Exception as e:
                print(f"❌ 无法读取日志文件: {e}")
            break
    else:
        print("❌ 未找到日志文件")


def clear_logs():
    """清理日志文件"""
    log_files = [
        "excel_agent_api.log",
        "tmp/excel_agent.log",
        Path.cwd() / "tmp" / "excel_agent.log"
    ]
    
    cleared = 0
    for log_file in log_files:
        if Path(log_file).exists():
            try:
                Path(log_file).unlink()
                print(f"🗑️  已删除日志文件: {log_file}")
                cleared += 1
            except Exception as e:
                print(f"❌ 无法删除日志文件 {log_file}: {e}")
    
    if cleared == 0:
        print("ℹ️  未找到需要清理的日志文件")
    else:
        print(f"✅ 已清理 {cleared} 个日志文件")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "tail":
            lines = int(sys.argv[2]) if len(sys.argv) > 2 else 50
            tail_logs(lines)
        elif command == "clear":
            clear_logs()
        elif command == "config":
            apply_logging_config()
        else:
            print("用法:")
            print("  python logging_config.py tail [lines]  # 查看最新日志")
            print("  python logging_config.py clear        # 清理日志文件")
            print("  python logging_config.py config       # 显示配置信息")
    else:
        print("Excel智能分析系统 - 日志配置")
        print("=" * 40)
        apply_logging_config()
        print("\n可用命令:")
        print("  python logging_config.py tail [lines]  # 查看最新日志")
        print("  python logging_config.py clear        # 清理日志文件")
        print("  python logging_config.py config       # 显示配置信息")