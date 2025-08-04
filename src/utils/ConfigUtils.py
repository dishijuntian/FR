"""
配置工具类
"""

from typing import Dict, List


class ConfigUtils:
    
    @staticmethod
    def merge_configs(base_config: Dict, override_config: Dict) -> Dict:
        """合并配置字典"""
        result = base_config.copy()
        
        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigUtils.merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @staticmethod
    def validate_config_section(config: Dict, section: str, required_keys: List[str]) -> bool:
        """验证配置节是否包含必要的键"""
        if section not in config:
            return False
        
        section_config = config[section]
        return all(key in section_config for key in required_keys)
