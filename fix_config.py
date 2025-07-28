#!/usr/bin/env python3
"""
配置修复脚本

该脚本帮助用户快速修复配置问题，特别是数据路径配置

使用方法:
    python fix_config.py
"""

import os
import sys

def get_current_config():
    """获取当前配置"""
    config_file = os.path.join('src', 'config.py')
    
    if not os.path.exists(config_file):
        print("❌ 找不到配置文件 src/config.py")
        return None, None
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找当前的DATA_BASE_PATH
        for line in content.split('\n'):
            if 'DATA_BASE_PATH' in line and '=' in line and not line.strip().startswith('#'):
                # 提取路径
                path_part = line.split('=')[1].strip().strip('"').strip("'")
                return content, path_part
        
        return content, None
        
    except Exception as e:
        print(f"❌ 读取配置文件失败: {e}")
        return None, None

def update_config(content, old_path, new_path):
    """更新配置文件"""
    try:
        # 替换路径
        old_line = f'DATA_BASE_PATH = "{old_path}"'
        new_line = f'DATA_BASE_PATH = "{new_path.replace(os.sep, "/")}"'
        
        updated_content = content.replace(old_line, new_line)
        
        # 如果没有找到完全匹配，尝试其他格式
        if updated_content == content:
            old_line = f"DATA_BASE_PATH = '{old_path}'"
            new_line = f'DATA_BASE_PATH = "{new_path.replace(os.sep, "/")}"'
            updated_content = content.replace(old_line, new_line)
        
        # 写入文件
        config_file = os.path.join('src', 'config.py')
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(updated_content)
        
        print(f"✅ 配置已更新: {new_path}")
        return True
        
    except Exception as e:
        print(f"❌ 更新配置失败: {e}")
        return False

def find_data_directory():
    """自动查找可能的数据目录"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 向上查找可能的数据目录
    possible_paths = []
    
    # 查找当前目录及其父目录
    search_dirs = [current_dir]
    parent = os.path.dirname(current_dir)
    if parent != current_dir:
        search_dirs.append(parent)
        grandparent = os.path.dirname(parent)
        if grandparent != parent:
            search_dirs.append(grandparent)
    
    for search_dir in search_dirs:
        for item in os.listdir(search_dir):
            item_path = os.path.join(search_dir, item)
            if os.path.isdir(item_path):
                # 查找包含aeroclub或recsys的目录
                if 'aeroclub' in item.lower() or 'recsys' in item.lower() or 'data' in item.lower():
                    possible_paths.append(item_path)
                
                # 查找包含train和test目录的数据目录
                encode_path = os.path.join(item_path, 'encode')
                if os.path.exists(encode_path):
                    train_path = os.path.join(encode_path, 'train')
                    test_path = os.path.join(encode_path, 'test')
                    if os.path.exists(train_path) and os.path.exists(test_path):
                        possible_paths.append(item_path)
    
    return list(set(possible_paths))  # 去重

def main():
    """主函数"""
    print("🔧 配置修复脚本")
    print("="*30)
    
    # 1. 获取当前配置
    print("\n📋 检查当前配置...")
    content, current_path = get_current_config()
    
    if content is None:
        print("无法读取配置文件")
        input("按Enter键退出...")
        return
    
    print(f"当前数据路径: {current_path}")
    
    # 2. 检查路径是否存在
    if current_path and os.path.exists(current_path):
        print("✅ 当前路径配置正确，无需修改")
        input("按Enter键退出...")
        return
    
    print(f"❌ 路径不存在: {current_path}")
    
    # 3. 自动查找可能的数据目录
    print("\n🔍 自动查找数据目录...")
    possible_paths = find_data_directory()
    
    if possible_paths:
        print("找到以下可能的数据目录:")
        for i, path in enumerate(possible_paths, 1):
            print(f"  {i}. {path}")
        print(f"  {len(possible_paths) + 1}. 手动输入路径")
        
        while True:
            try:
                choice = input(f"\n请选择 (1-{len(possible_paths) + 1}): ").strip()
                choice_num = int(choice)
                
                if 1 <= choice_num <= len(possible_paths):
                    new_path = possible_paths[choice_num - 1]
                    break
                elif choice_num == len(possible_paths) + 1:
                    new_path = input("请输入数据目录路径: ").strip()
                    break
                else:
                    print("请输入有效的选择")
            except ValueError:
                print("请输入数字")
    else:
        print("未找到可能的数据目录")
        new_path = input("请手动输入数据目录路径: ").strip()
    
    # 4. 验证新路径
    if not os.path.exists(new_path):
        print(f"❌ 路径不存在: {new_path}")
        create = input("是否创建该路径? (y/n): ").strip().lower()
        if create == 'y':
            try:
                os.makedirs(new_path, exist_ok=True)
                print(f"✅ 路径已创建: {new_path}")
            except Exception as e:
                print(f"❌ 创建路径失败: {e}")
                input("按Enter键退出...")
                return
        else:
            print("未更新配置")
            input("按Enter键退出...")
            return
    
    # 5. 更新配置
    print(f"\n📝 更新配置...")
    if update_config(content, current_path, new_path):
        print("✅ 配置更新成功!")
        print(f"新的数据路径: {new_path}")
        
        # 检查数据文件结构
        print("\n📁 检查数据文件结构...")
        train_path = os.path.join(new_path, 'encode', 'train')
        test_path = os.path.join(new_path, 'encode', 'test')
        
        if os.path.exists(train_path):
            train_files = [f for f in os.listdir(train_path) if f.endswith('.parquet')]
            print(f"✅ 训练文件目录存在，找到 {len(train_files)} 个文件")
        else:
            print(f"⚠️  训练文件目录不存在: {train_path}")
        
        if os.path.exists(test_path):
            test_files = [f for f in os.listdir(test_path) if f.endswith('.parquet')]
            print(f"✅ 测试文件目录存在，找到 {len(test_files)} 个文件")
        else:
            print(f"⚠️  测试文件目录不存在: {test_path}")
    else:
        print("❌ 配置更新失败")
    
    input("\n按Enter键退出...")

if __name__ == "__main__":
    main()