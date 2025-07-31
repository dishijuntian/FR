#!/bin/bash

# 航班排名系统启动脚本
# 适配简化版main.py，所有配置通过config/conf.yaml控制
# 错误时等待10秒，日志输出到logs目录

set -e  # 遇到错误立即退出

# 颜色输出定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# 创建日志目录和文件
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/run_$(date +%Y%m%d_%H%M%S).log"

# 创建日志目录
mkdir -p "$LOG_DIR"

# 日志函数 - 同时输出到终端和文件
log_info() {
    local message="[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo -e "${BLUE}$message${NC}"
    echo "$message" >> "$LOG_FILE"
}

log_warn() {
    local message="[WARN] $(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo -e "${YELLOW}$message${NC}"
    echo "$message" >> "$LOG_FILE"
}

log_error() {
    local message="[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo -e "${RED}$message${NC}"
    echo "$message" >> "$LOG_FILE"
}

log_success() {
    local message="[SUCCESS] $(date '+%Y-%m-%d %H:%M:%S') - $1"
    echo -e "${GREEN}$message${NC}"
    echo "$message" >> "$LOG_FILE"
}

# 等待并关闭函数
wait_and_exit() {
    local exit_code=$1
    local wait_time=10
    
    if [ $exit_code -ne 0 ]; then
        echo ""
        log_error "程序执行失败，退出码: $exit_code"
        echo -e "${YELLOW}将在 ${wait_time} 秒后自动关闭窗口...${NC}"
        echo -e "${YELLOW}按 Ctrl+C 立即关闭${NC}"
        
        # 倒计时
        for ((i=wait_time; i>0; i--)); do
            echo -ne "\r剩余时间: $i 秒 "
            sleep 1
        done
        echo ""
    fi
    
    log_info "日志文件保存在: $LOG_FILE"
    exit $exit_code
}

# 初始化日志
initialize_log() {
    echo "================================================================" >> "$LOG_FILE"
    echo "航班排名系统运行日志" >> "$LOG_FILE"
    echo "启动时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
    echo "项目路径: $PROJECT_ROOT" >> "$LOG_FILE"
    echo "日志文件: $LOG_FILE" >> "$LOG_FILE"
    echo "================================================================" >> "$LOG_FILE"
}

# 激活conda环境（如果存在）
activate_conda_env() {
    log_info "检查conda环境..."
    
    # 检查是否有conda
    if command -v conda &> /dev/null; then
        log_info "找到conda"
        
        # 尝试激活torch环境
        if conda env list | grep -q "^torch "; then
            log_info "激活conda环境: torch"
            source "$(conda info --base)/etc/profile.d/conda.sh"
            conda activate torch
            log_info "conda环境torch已激活"
        else
            log_warn "未找到torch环境，使用当前环境"
        fi
    else
        log_info "未找到conda，使用系统Python"
    fi
}

# 查找Python解释器
find_python() {
    log_info "查找Python解释器..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
        log_info "使用 Python3: $PYTHON_VERSION"
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1)
        if [[ $PYTHON_VERSION == *"Python 3"* ]]; then
            PYTHON_CMD="python"
            log_info "使用 Python: $PYTHON_VERSION"
        else
            log_error "需要 Python 3.x，当前版本: $PYTHON_VERSION"
            return 1
        fi
    else
        log_error "未找到 Python 解释器，请确保已安装 Python 3.x"
        return 1
    fi
    
    return 0
}

# 查找主程序文件
find_main_program() {
    log_info "查找主程序文件..."
    
    # 按优先级查找main.py
    local main_locations=(
        "$PROJECT_ROOT/src/main.py"
        "$PROJECT_ROOT/main.py"
    )
    
    MAIN_PY=""
    for location in "${main_locations[@]}"; do
        if [[ -f "$location" ]]; then
            MAIN_PY="$location"
            log_info "找到主程序: $MAIN_PY"
            return 0
        fi
    done
    
    log_error "未找到 main.py 文件"
    log_error "请确保 main.py 位于以下位置之一:"
    for location in "${main_locations[@]}"; do
        log_error "  - $location"
    done
    return 1
}

# 检查必需文件
check_required_files() {
    log_info "检查必需文件..."
    
    # 检查配置文件
    CONFIG_FILE="$PROJECT_ROOT/config/conf.yaml"
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "配置文件不存在: $CONFIG_FILE"
        log_error "请创建配置文件"
        return 1
    else
        log_info "找到配置文件: $CONFIG_FILE"
    fi
    
    # 检查核心模块
    local core_locations=(
        "$PROJECT_ROOT/core/Core.py"
        "$PROJECT_ROOT/src/core/Core.py"
    )
    
    local core_found=false
    for location in "${core_locations[@]}"; do
        if [[ -f "$location" ]]; then
            log_info "找到核心模块: $location"
            core_found=true
            break
        fi
    done
    
    if [[ "$core_found" == false ]]; then
        log_error "未找到核心模块 Core.py"
        log_error "请确保 Core.py 位于以下位置之一:"
        for location in "${core_locations[@]}"; do
            log_error "  - $location"
        done
        return 1
    fi
    
    return 0
}

# 检查Python包依赖
check_python_packages() {
    log_info "检查Python包依赖..."
    
    # 必需的包列表
    local required_packages=(
        "yaml"
        "pandas"
        "numpy"
        "pathlib"
    )
    
    # 可选的机器学习包
    local optional_packages=(
        "xgboost"
        "lightgbm" 
        "torch"
        "sklearn"
    )
    
    local missing_required=()
    local missing_optional=()
    
    # 检查必需包
    for package in "${required_packages[@]}"; do
        if ! $PYTHON_CMD -c "import $package" &> /dev/null 2>> "$LOG_FILE"; then
            missing_required+=("$package")
            log_warn "缺少必需包: $package"
        fi
    done
    
    # 检查可选包
    for package in "${optional_packages[@]}"; do
        if ! $PYTHON_CMD -c "import $package" &> /dev/null 2>> "$LOG_FILE"; then
            missing_optional+=("$package")
            log_warn "缺少可选包: $package"
        fi
    done
    
    if [ ${#missing_required[@]} -gt 0 ]; then
        log_error "缺少 ${#missing_required[@]} 个必需的Python包"
        log_error "请运行以下命令安装:"
        log_error "pip install ${missing_required[*]}"
        return 1
    fi
    
    if [ ${#missing_optional[@]} -gt 0 ]; then
        log_warn "缺少 ${#missing_optional[@]} 个可选的Python包"
        log_warn "建议安装以获得完整功能:"
        log_warn "pip install ${missing_optional[*]}"
    fi
    
    log_success "必需的Python包检查完成"
    return 0
}

# 设置运行环境
setup_environment() {
    log_info "设置运行环境..."
    
    # 添加项目根目录到PYTHONPATH
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    log_info "PYTHONPATH: $PYTHONPATH"
    
    # 切换到项目根目录
    cd "$PROJECT_ROOT"
    log_info "工作目录: $(pwd)"
    
    # 创建必要目录
    local dirs_to_create=(
        "$PROJECT_ROOT/data"
        "$PROJECT_ROOT/logs"
    )
    
    for dir in "${dirs_to_create[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_info "创建目录: $dir"
        fi
    done
}

# 运行主程序
run_main_program() {
    log_info "启动航班排名系统主程序..."
    log_info "使用配置文件: config/conf.yaml"
    
    echo "================================================================"
    echo -e "${GREEN}开始执行航班排名系统${NC}"
    echo "================================================================"
    
    # 运行主程序，同时将输出写入日志
    if $PYTHON_CMD "$MAIN_PY" 2>&1 | tee -a "$LOG_FILE"; then
        local exit_code=${PIPESTATUS[0]}
        if [ $exit_code -eq 0 ]; then
            echo "================================================================"
            log_success "航班排名系统执行完成"
            echo "================================================================"
            return 0
        else
            log_error "主程序执行失败，退出码: $exit_code"
            return $exit_code
        fi
    else
        log_error "主程序启动失败"
        return 1
    fi
}

# 主函数
main() {
    # 初始化日志
    initialize_log
    
    echo "================================================================"
    log_info "航班排名系统启动脚本"
    log_info "项目路径: $PROJECT_ROOT"
    log_info "日志文件: $LOG_FILE"
    echo "================================================================"
    
    # 执行所有检查
    activate_conda_env || log_warn "conda环境激活失败，继续使用当前环境"
    find_python || wait_and_exit 1
    find_main_program || wait_and_exit 1
    check_required_files || wait_and_exit 1
    check_python_packages || wait_and_exit 1
    setup_environment || wait_and_exit 1
    
    # 运行主程序
    if run_main_program; then
        wait_and_exit 0
    else
        wait_and_exit 1
    fi
}

# 错误处理 - 捕获中断信号
cleanup() {
    log_error "程序被用户中断"
    wait_and_exit 130
}

# 意外错误处理
error_handler() {
    local exit_code=$?
    log_error "脚本执行过程中发生意外错误，退出码: $exit_code"
    wait_and_exit $exit_code
}

# 设置信号处理
trap cleanup INT TERM
trap error_handler ERR

# 执行主函数
main "$@"
