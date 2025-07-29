import pandas as pd
import numpy as np
import os
import pyarrow.parquet as pq
from typing import Dict, List, Set
import logging

class DataDiagnostics:
    """数据诊断工具类"""
    
    def __init__(self, base_dir: str = "data/aeroclub-recsys-2025"):
        self.base_dir = base_dir
        self.encoded_dir = os.path.join(base_dir, "encoded")
        self.segment_dir = os.path.join(base_dir, "segmented")
        
        # 设置简单日志
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s | %(levelname)s | %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def analyze_segment_patterns(self, data_type: str) -> Dict:
        """分析segment模式分布"""
        encoded_file = os.path.join(self.encoded_dir, data_type, f"{data_type}_encoded.parquet")
        
        if not os.path.exists(encoded_file):
            self.logger.error(f"编码文件不存在: {encoded_file}")
            return {}
        
        self.logger.info(f"分析 {data_type} 的segment模式...")
        
        # 读取数据
        df = pd.read_parquet(encoded_file)
        self.logger.info(f"加载数据: {len(df):,} 行, {df['ranker_id'].nunique():,} ranker_id")
        
        # 分析每个ranker_id的segment级别
        ranker_analysis = {}
        
        for ranker_id in df['ranker_id'].unique():
            ranker_data = df[df['ranker_id'] == ranker_id]
            segment_levels = []
            
            for level in [0, 1, 2, 3]:
                pattern = f'segments{level}'
                segment_cols = df.filter(regex=pattern).columns
                
                if len(segment_cols) > 0:
                    segment_data = ranker_data[segment_cols]
                    has_valid_data = ~((segment_data == -1) | segment_data.isnull()).all(axis=1).all()
                    if has_valid_data:
                        segment_levels.append(level)
            
            ranker_analysis[ranker_id] = {
                'levels': segment_levels,
                'max_level': max(segment_levels) if segment_levels else -1,
                'row_count': len(ranker_data)
            }
        
        # 统计分析
        level_distribution = {0: 0, 1: 0, 2: 0, 3: 0, -1: 0}  # -1表示无有效segment
        multiple_levels = 0
        total_rows_by_level = {0: 0, 1: 0, 2: 0, 3: 0, -1: 0}
        
        for ranker_id, info in ranker_analysis.items():
            max_level = info['max_level']
            level_distribution[max_level] += 1
            total_rows_by_level[max_level] += info['row_count']
            
            if len(info['levels']) > 1:
                multiple_levels += 1
        
        result = {
            'total_ranker_ids': len(ranker_analysis),
            'total_rows': len(df),
            'level_distribution': level_distribution,
            'rows_by_level': total_rows_by_level,
            'multiple_levels_count': multiple_levels,
            'sample_analysis': dict(list(ranker_analysis.items())[:5])  # 前5个样本
        }
        
        return result
    
    def compare_segmentation_results(self, data_type: str) -> Dict:
        """比较分割前后的数据"""
        encoded_file = os.path.join(self.encoded_dir, data_type, f"{data_type}_encoded.parquet")
        segment_dir = os.path.join(self.segment_dir, data_type)
        
        if not os.path.exists(encoded_file):
            self.logger.error(f"编码文件不存在: {encoded_file}")
            return {}
        
        self.logger.info(f"比较 {data_type} 分割前后数据...")
        
        # 原始数据统计
        original_df = pd.read_parquet(encoded_file)
        original_stats = {
            'total_rows': len(original_df),
            'total_ranker_ids': original_df['ranker_id'].nunique(),
            'ranker_ids': set(original_df['ranker_id'].unique())
        }
        
        # 分割数据统计
        segment_stats = {}
        total_segmented_rows = 0
        all_segmented_rankers = set()
        
        for level in [0, 1, 2, 3]:
            segment_file = os.path.join(segment_dir, f"{data_type}_segment_{level}.parquet")
            if os.path.exists(segment_file):
                segment_df = pd.read_parquet(segment_file)
                rows = len(segment_df)
                rankers = set(segment_df['ranker_id'].unique()) if rows > 0 else set()
                
                segment_stats[level] = {
                    'rows': rows,
                    'ranker_ids': len(rankers),
                    'ranker_set': rankers
                }
                
                total_segmented_rows += rows
                all_segmented_rankers.update(rankers)
            else:
                segment_stats[level] = {
                    'rows': 0,
                    'ranker_ids': 0,
                    'ranker_set': set()
                }
        
        # 找出丢失的ranker_id
        missing_rankers = original_stats['ranker_ids'] - all_segmented_rankers
        extra_rankers = all_segmented_rankers - original_stats['ranker_ids']
        
        # 检查ranker_id重复
        ranker_overlaps = {}
        for level1 in [0, 1, 2, 3]:
            for level2 in range(level1 + 1, 4):
                overlap = segment_stats[level1]['ranker_set'] & segment_stats[level2]['ranker_set']
                if overlap:
                    ranker_overlaps[f"segment_{level1}_vs_{level2}"] = len(overlap)
        
        comparison = {
            'original': original_stats,
            'segmented': {
                'total_rows': total_segmented_rows,
                'total_ranker_ids': len(all_segmented_rankers),
                'by_segment': {level: {'rows': stats['rows'], 'ranker_ids': stats['ranker_ids']} 
                              for level, stats in segment_stats.items()}
            },
            'integrity': {
                'rows_match': original_stats['total_rows'] == total_segmented_rows,
                'rankers_match': original_stats['total_ranker_ids'] == len(all_segmented_rankers),
                'missing_rankers': len(missing_rankers),
                'extra_rankers': len(extra_rankers),
                'ranker_overlaps': ranker_overlaps
            },
            'missing_sample': list(missing_rankers)[:10] if missing_rankers else [],
            'extra_sample': list(extra_rankers)[:10] if extra_rankers else []
        }
        
        return comparison
    
    def diagnose_missing_data(self, data_type: str) -> Dict:
        """诊断数据丢失的具体原因"""
        encoded_file = os.path.join(self.encoded_dir, data_type, f"{data_type}_encoded.parquet")
        
        if not os.path.exists(encoded_file):
            return {}
        
        self.logger.info(f"诊断 {data_type} 数据丢失原因...")
        
        df = pd.read_parquet(encoded_file)
        
        # 检查每个ranker_id的segment数据完整性
        problematic_rankers = []
        
        for ranker_id in df['ranker_id'].unique()[:100]:  # 限制检查数量
            ranker_data = df[df['ranker_id'] == ranker_id]
            
            diagnosis = {
                'ranker_id': ranker_id,
                'row_count': len(ranker_data),
                'segments_found': [],
                'all_segments_missing': True
            }
            
            for level in [0, 1, 2, 3]:
                pattern = f'segments{level}'
                segment_cols = df.filter(regex=pattern).columns
                
                if len(segment_cols) > 0:
                    segment_data = ranker_data[segment_cols]
                    # 检查是否有任何非-1且非null的值
                    has_valid = ~((segment_data == -1) | segment_data.isnull()).all(axis=1).any()
                    
                    if has_valid:
                        diagnosis['segments_found'].append(level)
                        diagnosis['all_segments_missing'] = False
            
            if diagnosis['all_segments_missing'] or not diagnosis['segments_found']:
                problematic_rankers.append(diagnosis)
        
        return {
            'total_checked': min(100, df['ranker_id'].nunique()),
            'problematic_count': len(problematic_rankers),
            'problematic_sample': problematic_rankers[:10]
        }
    
    def generate_diagnostic_report(self, data_type: str) -> str:
        """生成完整的诊断报告"""
        report_lines = [
            f"数据诊断报告 - {data_type.upper()}",
            "=" * 50,
            ""
        ]
        
        # Segment模式分析
        pattern_analysis = self.analyze_segment_patterns(data_type)
        if pattern_analysis:
            report_lines.extend([
                "1. Segment模式分析:",
                f"   总ranker_id数: {pattern_analysis['total_ranker_ids']:,}",
                f"   总行数: {pattern_analysis['total_rows']:,}",
                f"   多级别ranker_id: {pattern_analysis['multiple_levels_count']:,}",
                "",
                "   按最高级别分布:"
            ])
            
            for level in [3, 2, 1, 0, -1]:
                count = pattern_analysis['level_distribution'][level]
                rows = pattern_analysis['rows_by_level'][level]
                if count > 0:
                    level_name = f"Segment {level}" if level >= 0 else "无有效Segment"
                    report_lines.append(f"     {level_name}: {count:,} ranker_id, {rows:,} 行")
            
            report_lines.append("")
        
        # 分割结果比较
        comparison = self.compare_segmentation_results(data_type)
        if comparison:
            report_lines.extend([
                "2. 分割结果比较:",
                f"   原始数据: {comparison['original']['total_rows']:,} 行, {comparison['original']['total_ranker_ids']:,} ranker_id",
                f"   分割数据: {comparison['segmented']['total_rows']:,} 行, {comparison['segmented']['total_ranker_ids']:,} ranker_id",
                "",
                "   分割详情:"
            ])
            
            for level in [3, 2, 1, 0]:
                stats = comparison['segmented']['by_segment'][level]
                if stats['rows'] > 0:
                    report_lines.append(f"     Segment {level}: {stats['rows']:,} 行, {stats['ranker_ids']:,} ranker_id")
            
            integrity = comparison['integrity']
            report_lines.extend([
                "",
                "   完整性检查:",
                f"     行数匹配: {'✓' if integrity['rows_match'] else '✗'}",
                f"     ranker_id匹配: {'✓' if integrity['rankers_match'] else '✗'}",
                f"     丢失ranker_id: {integrity['missing_rankers']}",
                f"     多余ranker_id: {integrity['extra_rankers']}",
                f"     ranker_id重叠: {len(integrity['ranker_overlaps'])} 对"
            ])
            
            if integrity['ranker_overlaps']:
                report_lines.append("     重叠详情:")
                for overlap_key, count in integrity['ranker_overlaps'].items():
                    report_lines.append(f"       {overlap_key}: {count} ranker_id")
            
            report_lines.append("")
        
        # 数据丢失诊断
        missing_diagnosis = self.diagnose_missing_data(data_type)
        if missing_diagnosis:
            report_lines.extend([
                "3. 数据丢失诊断:",
                f"   检查样本: {missing_diagnosis['total_checked']} ranker_id",
                f"   问题ranker_id: {missing_diagnosis['problematic_count']}",
                ""
            ])
            
            if missing_diagnosis['problematic_sample']:
                report_lines.append("   问题样本:")
                for sample in missing_diagnosis['problematic_sample'][:5]:
                    report_lines.append(f"     ranker_id {sample['ranker_id']}: {sample['row_count']} 行, segments: {sample['segments_found']}")
                report_lines.append("")
        
        return "\n".join(report_lines)
    
    def save_diagnostic_report(self, data_type: str, output_file: str = None) -> str:
        """保存诊断报告到文件"""
        if output_file is None:
            output_file = os.path.join(self.base_dir, f"diagnostic_report_{data_type}.txt")
        
        report = self.generate_diagnostic_report(data_type)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"诊断报告已保存: {output_file}")
        return output_file


def main():
    """诊断脚本主函数"""
    import sys
    
    diagnostics = DataDiagnostics()
    
    data_type = sys.argv[1] if len(sys.argv) > 1 else 'train'
    
    print(f"开始诊断 {data_type} 数据...")
    
    # 生成并显示报告
    report = diagnostics.generate_diagnostic_report(data_type)
    print(report)
    
    # 保存报告
    output_file = diagnostics.save_diagnostic_report(data_type)
    print(f"\n报告已保存到: {output_file}")


if __name__ == "__main__":
    main()