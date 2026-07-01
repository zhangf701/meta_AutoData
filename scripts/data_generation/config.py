"""Configuration for the data generation pipeline."""

import os

# Base directories
BASE_DIR = r"D:\coding\meta_AutoData"
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts", "data_generation")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Source JSON files (Markdown text stored as .json extension)
STANDARDS_DIR = os.path.join(DATA_DIR, "国标规范json")
DESIGN_DIR = os.path.join(DATA_DIR, "待审查文件json")

SOURCE_FILES = [
    os.path.join(STANDARDS_DIR, "GB+38755-电力系统安全稳定导则2019.json"),
    os.path.join(STANDARDS_DIR, "DLT 5429-2009 电力系统设计技术规程.json"),
    os.path.join(STANDARDS_DIR, "DLT 5218-2012 220kV-750kV变电站设计技术规程.json"),
    os.path.join(DESIGN_DIR, "500kV变电站设计方案.json"),
    os.path.join(DESIGN_DIR, "500kV变电站设计方案反馈单.json"),
    os.path.join(DESIGN_DIR, "江门电网专项规划2035.json"),
] + [
    # 审查纪要 .md 文件（真实反馈的三段式模板：问题→审查意见→答复）
    os.path.join(DESIGN_DIR, f)
    for f in sorted(os.listdir(DESIGN_DIR))
    if f.endswith(".md")
]

# Files with encoding issues
GBK_ENCODED_FILES = [
    os.path.join(STANDARDS_DIR, "DLT 5218-2012 220kV-750kV变电站设计技术规程.json"),
]

# Output paths
OUTPUT_DIR = os.path.join(DATA_DIR, "questions")
CLAUSES_OUTPUT = os.path.join(SCRIPTS_DIR, "clauses_v1.json")
QUESTIONS_OUTPUT = os.path.join(OUTPUT_DIR, "generated_eval_set_v1.json")
GOLD_OUTPUT = os.path.join(OUTPUT_DIR, "generated_gold_chunks_v1.json")
QUALITY_REPORT_DIR = os.path.join(SCRIPTS_DIR, "quality_reports")

# Reference files (for format verification only, NOT used as data source)
REF_EVAL_SET = os.path.join(OUTPUT_DIR, "eval_set_70_v3.json")
REF_GOLD_CHUNKS = os.path.join(OUTPUT_DIR, "gold_chunks_verified_v3.json")

# Few-shot examples for L2 generation
L2_FEWSHOT_FILE = os.path.join(BASE_DIR, "日志", "L2综合推理问题集报告.md")

# Model configurations
WEAK_MODEL = "qwen2.5:3b"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT = 120  # seconds
OLLAMA_START_TIMEOUT = 60    # seconds to wait for ollama serve auto-start
OLLAMA_EXECUTABLE = r"C:\Users\Z\AppData\Local\Programs\Ollama\ollama.exe"

STRONG_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Generation targets
TARGET_COUNTS = {"L1": 100, "L2": 100, "L3": 100}
MAX_L3_NARROW_ITERATIONS = 3

# Quality gate thresholds
L1_WEAK_REJECT_THRESHOLD = 0.7   # Reject L1 if weak >= 0.7 (too easy)
L2_ACCEPT_WEAK_MAX = 0.4         # Accept L2 only if weak < 0.4
L2_ACCEPT_STRONG_MIN = 0.7       # Accept L2 only if strong >= 0.7
L2_ACCEPT_GAP_MIN = 0.2          # Accept L2 only if gap >= 0.2
L3_ACCEPT_WEAK_MIN = 0.1         # Accept L3 only if weak > 0 (has learnable signal)
L3_ACCEPT_STRONG_MIN = 0.5       # Accept L3 only if strong >= 0.5

# Scenario length requirements (Chinese characters)
MIN_L2_SCENARIO_CHARS = 150
MIN_L3_SCENARIO_CHARS = 300

# API key from environment
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")

# Scoring package path (for importing rubric_score)
SCORING_PACKAGE_PATH = r"D:\coding\power_grid_rag\scripts\scoring"

# Ensure directories exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(QUALITY_REPORT_DIR, exist_ok=True)
