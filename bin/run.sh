#!/bin/bash

# 航班排名系统启动脚本
<<<<<<< HEAD
# 自动检测项目结构并启动系统

set -e  # 遇到错误立即退出

# 颜色输出定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# 获取脚本所在目录的绝对路径
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

log_info "脚本目录: $SCRIPT_DIR"
log_info "项目根目录: $PROJECT_ROOT"

# 查找主程序文件
MAIN_PY=""
if [[ -f "$PROJECT_ROOT/main.py" ]]; then
    MAIN_PY="$PROJECT_ROOT/main.py"
    log_info "找到主程序: $MAIN_PY"
elif [[ -f "$PROJECT_ROOT/src/main.py" ]]; then
    MAIN_PY="$PROJECT_ROOT/src/main.py"
    log_info "找到主程序: $MAIN_PY"
else
    log_error "未找到 main.py 文件"
    log_error "请确保 main.py 位于项目根目录或 src/ 目录下"
    exit 1
fi

# 检查Python环境
check_python() {
    log_info "检查Python环境..."
    
    # 检查Python3是否可用
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
            log_error "需要 Python 3.x，当前: $PYTHON_VERSION"
            exit 1
        fi
    else
        log_error "未找到 Python 解释器"
        exit 1
    fi
}

# 检查依赖文件
check_dependencies() {
    log_info "检查项目依赖..."
    
    # 检查配置文件
    CONFIG_FILE="$PROJECT_ROOT/config/conf.yaml"
    if [[ ! -f "$CONFIG_FILE" ]]; then
        log_warn "配置文件不存在: $CONFIG_FILE"
        log_warn "将使用默认配置或命令行指定的配置"
    else
        log_info "找到配置文件: $CONFIG_FILE"
    fi
    
    # 检查核心模块
    CORE_MODULE="$PROJECT_ROOT/core/Core.py"
    if [[ ! -f "$CORE_MODULE" ]]; then
        log_error "核心模块不存在: $CORE_MODULE"
        exit 1
    fi
    
    # 检查工具模块
    UTILS_MODULE="$PROJECT_ROOT/src/utils/ConfigUtils.py"
    if [[ ! -f "$UTILS_MODULE" ]]; then
        log_error "工具模块不存在: $UTILS_MODULE"
        exit 1
    fi
    
    log_success "依赖检查完成"
}

# 检查Python包依赖
check_python_packages() {
    log_info "检查Python包依赖..."
    
    # 必需的包列表
    REQUIRED_PACKAGES=(
        "yaml"
        "pandas"
        "numpy"
        "pathlib"
        "argparse"
        "psutil"
    )
    
    for package in "${REQUIRED_PACKAGES[@]}"; do
        if ! $PYTHON_CMD -c "import $package" &> /dev/null; then
            log_warn "缺少Python包: $package"
            log_warn "请运行: pip install $package"
        fi
    done
}

# 设置环境变量
setup_environment() {
    log_info "设置环境变量..."
    
    # 添加项目根目录到PYTHONPATH
    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    log_info "PYTHONPATH: $PYTHONPATH"
    
    # 设置工作目录
    cd "$PROJECT_ROOT"
    log_info "工作目录: $(pwd)"
}
=======
# 提供交互式菜单选项和直接参数启动两种方式

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE}" )" && pwd )"
cd "$SCRIPT_DIR" || { echo "❌ 无法进入项目目录"; exit 1; }

# 设置Python路径
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# 主程序路径
MAIN_PY="src/main.py"
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3

# 显示帮助信息
show_help() {
    echo "航班排名系统启动脚本"
<<<<<<< HEAD
    echo ""
    echo "用法: $0 [选项] [-- [Python程序选项]]"
    echo ""
    echo "脚本选项:"
    echo "  -h, --help              显示此帮助信息"
    echo "  -c, --check             只检查环境，不启动程序"
    echo "  -v, --verbose           详细输出"
    echo "  --python PYTHON_CMD     指定Python解释器"
    echo ""
    echo "Python程序选项 (在 -- 之后):"
    echo "  --config CONFIG         配置文件路径"
    echo "  --mode MODE             运行模式 (full|data|training|prediction)"
    echo "  --status                显示状态报告"
    echo "  --force                 强制重新处理"
    echo "  --gpu auto|on|off       GPU设置"
    echo "  --verbose               详细输出"
    echo "  --dry-run               干运行模式"
    echo ""
    echo "示例:"
    echo "  $0                                    # 运行完整流水线"
    echo "  $0 -- --mode data                     # 只执行数据处理"
    echo "  $0 -- --config my_config.yaml        # 使用指定配置"
    echo "  $0 -- --status                       # 查看状态报告"
    echo "  $0 -c                                 # 只检查环境"
}

# 主函数
main() {
    local check_only=false
    local verbose=false
    local python_args=()
    
    # 解析脚本参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -c|--check)
                check_only=true
                shift
                ;;
            -v|--verbose)
                verbose=true
                shift
                ;;
            --python)
                PYTHON_CMD="$2"
                shift 2
                ;;
            --)
                shift
                python_args=("$@")
                break
                ;;
            *)
                python_args+=("$1")
                shift
                ;;
        esac
    done
    
    log_info "开始启动航班排名系统..."
    echo "================================================================"
    
    # 执行检查
    check_python
    check_dependencies
    
    if [[ "$verbose" == "true" ]]; then
        check_python_packages
    fi
    
    setup_environment
    
    if [[ "$check_only" == "true" ]]; then
        log_success "环境检查完成，系统可以正常启动"
        exit 0
    fi
    
    echo "================================================================"
    log_info "启动程序: $MAIN_PY"
    
    if [[ ${#python_args[@]} -gt 0 ]]; then
        log_info "程序参数: ${python_args[*]}"
    fi
    
    echo "================================================================"
    
    # 启动主程序
    exec $PYTHON_CMD "$MAIN_PY" "${python_args[@]}"
}

# 错误处理
trap 'log_error "脚本执行被中断"; exit 130' INT
trap 'log_error "脚本执行失败"; exit 1' ERR

# 执行主函数
main "$@"
=======
    echo "用法:"
    echo "  $0 [选项]"
    echo "  $0 [命令行参数]"
    echo
    echo "选项:"
    echo "  -h, --help     显示此帮助信息"
    echo "  -i, --interactive  进入交互式菜单"
    echo
    echo "命令行参数 (直接传递给主程序):"
    echo "  --config       指定配置文件路径"
    echo "  --mode         运行模式 (full, data, training, prediction)"
    echo "  --status       显示状态报告"
    echo "  --force        强制重新处理"
    echo "  --no-verify    跳过数据验证"
    echo "  --segments     指定数据段"
    echo "  --model        指定预测模型"
    echo "  --verbose      详细输出模式"
    echo
    echo "示例:"
    echo "  $0                          # 默认运行完整流水线"
    echo "  $0 --mode training          # 只运行模型训练"
    echo "  $0 --config myconfig.yaml   # 使用自定义配置"
    echo "  $0 -i                       # 进入交互式菜单"
}

# 运行主程序
run_main() {
    echo "▶️ 启动航班排名系统..."
    echo "🏃 执行命令: python3 $MAIN_PY $*"
    echo "------------------------------------------------------------"
    
    python3 "$MAIN_PY" "$@"
    local exit_status=$?
    
    echo "------------------------------------------------------------"
    if [ $exit_status -eq 0 ]; then
        echo "✅ 程序执行成功"
    else
        echo "❌ 程序执行失败 (错误码: $exit_status)" >&2
    fi
    
    return $exit_status
}

# 交互式菜单
interactive_menu() {
    while true; do
        echo
        echo "============================================================"
        echo "航班排名系统 - 主菜单"
        echo "============================================================"
        echo "1. 完整流水线运行 (默认)"
        echo "2. 仅数据处理"
        echo "3. 仅模型训练"
        echo "4. 仅模型预测"
        echo "5. 查看系统状态"
        echo "6. 高级选项"
        echo "7. 退出"
        echo "------------------------------------------------------------"
        
        read -p "请选择操作 [1-7] (默认1): " choice
        choice=${choice:-1}  # 默认值为1
        
        case $choice in
            1)
                run_main
                ;;
            2)
                run_main --mode data
                ;;
            3)
                run_main --mode training
                ;;
            4)
                run_main --mode prediction
                ;;
            5)
                run_main --status
                ;;
            6)
                advanced_menu
                ;;
            7)
                echo "退出系统"
                exit 0
                ;;
            *)
                echo "无效选择，请重新输入"
                ;;
        esac
        
        # 每次操作后暂停
        read -rp "按回车键继续..."
    done
}

# 高级选项菜单
advanced_menu() {
    while true; do
        echo
        echo "============================================================"
        echo "高级选项"
        echo "============================================================"
        echo "1. 强制重新处理所有数据"
        echo "2. 使用自定义配置文件"
        echo "3. 指定数据段"
        echo "4. 指定预测模型"
        echo "5. 跳过数据验证"
        echo "6. 详细输出模式"
        echo "7. 返回主菜单"
        echo "------------------------------------------------------------"
        
        read -p "请选择操作 [1-7]: " choice
        
        case $choice in
            1)
                run_main --force
                ;;
            2)
                read -p "请输入配置文件路径: " config_file
                run_main --config "$config_file"
                ;;
            3)
                read -p "请输入数据段 (空格分隔): " segments
                run_main --segments $segments
                ;;
            4)
                read -p "请选择模型 (XGBRanker/LGBMRanker): " model
                run_main --model "$model"
                ;;
            5)
                run_main --no-verify
                ;;
            6)
                run_main --verbose
                ;;
            7)
                return
                ;;
            *)
                echo "无效选择，请重新输入"
                ;;
        esac
        
        # 每次操作后暂停
        read -rp "按回车键继续..."
    done
}

# 解析参数
if [ $# -eq 0 ]; then
    # 没有参数时，运行默认完整流水线
    run_main
    exit $?
fi

# 检查参数
case $1 in
    -h|--help)
        show_help
        exit 0
        ;;
    -i|--interactive)
        interactive_menu
        exit 0
        ;;
    *)
        # 将参数传递给主程序
        run_main "$@"
        exit $?
        ;;
esac
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
