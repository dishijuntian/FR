import os
from Trainer import FlightRankingTrainer
from Predictor import FlightRankingPredictor

def main():
    # 设置根目录路径
    CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
    MAIN_PATH = os.path.abspath(os.path.join(CURRENT_PATH, "..", ".."))
    os.chdir(MAIN_PATH)
    print(f"工作目录设置为: {MAIN_PATH}")
    
    # 固定配置参数
    segments = [0, 1, 2]  # 要处理的segment ID列表
    model_name = 'XGBRanker'  # 默认模型名称
    use_gpu = True  # 默认启用GPU加速
    
    # 执行完整流程：训练 + 预测
    print("\n" + "="*50)
    print("开始训练模型...")
    trainer = FlightRankingTrainer(use_gpu=use_gpu)
    trainer.train_all(segments=segments)
    
    print("\n" + "="*50)
    print("开始预测结果...")
    predictor = FlightRankingPredictor(use_gpu=use_gpu)
    predictor.predict_all(segments=segments, model_name=model_name)
    
    print("\n" + "="*50)
    print("流程执行完成！")

if __name__ == "__main__":
    main()