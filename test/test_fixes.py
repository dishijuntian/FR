#!/usr/bin/env python3
"""
测试数据分割修复的脚本
"""

import os
import sys
import logging
from datetime import datetime

# 添加当前目录到路径
current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_path)

def setup_test_logging():
    """设置测试日志"""
    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[console_handler],
        force=True
    )

def test_data_segment_fix():
    """测试DataSegment修复"""
    print("=" * 60)
    print("测试 DataSegment 修复")
    print("=" * 60)
    
    try:
        # 使用修复后的DataSegment
        from src.data.DataSegment import DataSegment
        
        segmenter = DataSegment(chunk_size=100000, n_processes=2)
        print("✓ DataSegment_Fixed 导入成功")
        
        # 测试基本方法
        pattern = segmenter._get_segment_pattern(1)
        print(f"✓ _get_segment_pattern(1) = '{pattern}'")
        
        # 检查关键方法是否存在
        required_methods = [
            '_get_ranker_segment_classification',
            'process_file',
            'verify_segmentation',
            'get_output_files'
        ]
        
        for method in required_methods:
            if hasattr(segmenter, method):
                print(f"✓ 方法 {method} 存在")
            else:
                print(f"✗ 方法 {method} 缺失")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ DataSegment 测试失败: {e}")
        return False

def test_data_processor_fix():
    """测试DataProcessor修复"""
    print("\n" + "=" * 60)
    print("测试 DataProcessor 修复")
    print("=" * 60)
    
    try:
        # 使用修复后的DataProcessor
        from src.data.DataProcessor import DataProcessor
        
        processor = DataProcessor(
            base_dir="data/aeroclub-recsys-2025",
            chunk_size=100000,
        )
        print("✓ DataProcessor_Fixed 导入成功")
        
        # 检查关键方法
        required_methods = [
            'process_pipeline',
            'segment_data',
            'concatenate_segments',
            'get_pipeline_status',
            '_verify_existing_segmentation'
        ]
        
        for method in required_methods:
            if hasattr(processor, method):
                print(f"✓ 方法 {method} 存在")
            else:
                print(f"✗ 方法 {method} 缺失")
                return False
        
        # 测试状态检查
        status = processor.get_pipeline_status()
        print(f"✓ 状态检查成功: {list(status.keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ DataProcessor 测试失败: {e}")
        return False

def test_diagnostics():
    """测试诊断工具"""
    print("\n" + "=" * 60)
    print("测试诊断工具")
    print("=" * 60)
    
    try:
        from data_diagnostics import DataDiagnostics
        
        diagnostics = DataDiagnostics()
        print("✓ DataDiagnostics 导入成功")
        
        # 检查关键方法
        required_methods = [
            'analyze_segment_patterns',
            'compare_segmentation_results',
            'diagnose_missing_data',
            'generate_diagnostic_report'
        ]
        
        for method in required_methods:
            if hasattr(diagnostics, method):
                print(f"✓ 方法 {method} 存在")
            else:
                print(f"✗ 方法 {method} 缺失")
                return False
        
        return True
        
    except Exception as e:
        print(f"✗ 诊断工具测试失败: {e}")
        return False

def test_file_structure():
    """检查文件结构"""
    print("\n" + "=" * 60)
    print("检查数据文件结构")
    print("=" * 60)
    
    base_dir = "data/aeroclub-recsys-2025"
    
    # 检查基础文件
    required_files = [
        os.path.join(base_dir, "train.parquet"),
        os.path.join(base_dir, "test.parquet"),
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / 1024**2
            print(f"✓ {file_path} 存在 ({size_mb:.1f}MB)")
        else:
            print(f"✗ {file_path} 不存在")
    
    # 检查目录结构
    required_dirs = [
        os.path.join(base_dir, "encoded"),
        os.path.join(base_dir, "segmented"),
        os.path.join(base_dir, "encoded", "train"),
        os.path.join(base_dir, "encoded", "test"),
        os.path.join(base_dir, "segmented", "train"),
        os.path.join(base_dir, "segmented", "test"),
    ]
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✓ 目录 {dir_path} 存在")
        else:
            print(f"! 目录 {dir_path} 不存在（将被创建）")

def run_quick_diagnosis():
    """运行快速诊断（如果数据存在）"""
    print("\n" + "=" * 60)
    print("运行快速诊断")
    print("=" * 60)
    
    try:
        from data_diagnostics import DataDiagnostics
        
        diagnostics = DataDiagnostics()
        base_dir = "data/aeroclub-recsys-2025"
        
        for data_type in ['train', 'test']:
            encoded_file = os.path.join(base_dir, "encoded", data_type, f"{data_type}_encoded.parquet")
            
            if os.path.exists(encoded_file):
                print(f"\n检查 {data_type} 数据...")
                
                # 快速模式分析
                pattern_analysis = diagnostics.analyze_segment_patterns(data_type)
                if pattern_analysis:
                    print(f"  总行数: {pattern_analysis['total_rows']:,}")
                    print(f"  总ranker_id: {pattern_analysis['total_ranker_ids']:,}")
                    print("  级别分布:")
                    for level in [3, 2, 1, 0, -1]:
                        count = pattern_analysis['level_distribution'][level]
                        if count > 0:
                            level_name = f"Segment {level}" if level >= 0 else "无有效Segment"
                            print(f"    {level_name}: {count:,} ranker_id")
                
                # 如果分割文件存在，比较结果
                segment_dir = os.path.join(base_dir, "segmented", data_type)
                if os.path.exists(segment_dir):
                    comparison = diagnostics.compare_segmentation_results(data_type)
                    if comparison:
                        integrity = comparison['integrity']
                        print(f"  分割完整性:")
                        print(f"    行数匹配: {'✓' if integrity['rows_match'] else '✗'}")
                        print(f"    ranker_id匹配: {'✓' if integrity['rankers_match'] else '✗'}")
                        if not integrity['rows_match']:
                            orig = comparison['original']['total_rows']
                            seg = comparison['segmented']['total_rows']
                            print(f"    差异: {orig - seg} 行")
            else:
                print(f"  {data_type} 编码文件不存在，跳过诊断")
                
    except Exception as e:
        print(f"诊断失败: {e}")

def main():
    """主测试函数"""
    print("数据分割修复测试")
    print("时间:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    setup_test_logging()
    
    # 切换到正确的工作目录
    current_path = os.path.dirname(os.path.abspath(__file__))
    main_path = os.path.abspath(os.path.join(current_path, "..", ".."))
    if os.path.exists(main_path):
        os.chdir(main_path)
        print(f"工作目录: {main_path}")
    
    # 运行测试
    tests = [
        ("文件结构检查", test_file_structure),
        ("DataSegment修复", test_data_segment_fix),
        ("DataProcessor修复", test_data_processor_fix),
        ("诊断工具", test_diagnostics),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} 测试异常: {e}")
            results.append((test_name, False))
    
    # 运行诊断（如果基础测试通过）
    if all(result for _, result in results):
        run_quick_diagnosis()
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！可以运行修复后的处理流水线。")
        print("\n下一步:")
        print("1. 运行 python DataProcessor_Fixed.py")
        print("2. 或者在代码中导入修复后的类")
    else:
        print("\n⚠️  有测试失败，请检查修复的代码文件。")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)