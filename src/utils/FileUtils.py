"""
文件操作工具类
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, Union


class FileUtils:    
    @staticmethod
    def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """获取文件信息"""
        file_path = Path(file_path)
        if not file_path.exists():
            return {'exists': False}
        
        try:
            info = {
                'exists': True,
                'size_mb': file_path.stat().st_size / 1024**2,
                'modified': file_path.stat().st_mtime
            }
            
            # 如果是parquet文件，获取更多信息
            if file_path.suffix == '.parquet':
                import pyarrow.parquet as pq
                pf = pq.ParquetFile(file_path)
                info.update({
                    'rows': pf.metadata.num_rows,
                    'columns': len(pf.schema.names)
                })
            
            return info
        except Exception as e:
            return {'exists': True, 'error': str(e)}
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """确保目录存在"""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def get_file_hash(file_path: Union[str, Path]) -> str:
        """计算文件MD5哈希"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

