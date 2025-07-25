#!/bin/bash

# 航班排名系统启动脚本
# 提供交互式菜单选项和直接参数启动两种方式

# 获取脚本所在目录（项目根目录）
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE}" )" && pwd )"
cd "$SCRIPT_DIR" || { echo "❌ 无法进入项目目录"; exit 1; }

# 设置Python路径
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# 主程序路径
MAIN_PY="src/main.py"

# 显示帮助信息
show_help() {
    echo "航班排名系统启动脚本"
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