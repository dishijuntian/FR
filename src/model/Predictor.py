import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json
import numpy as np
import pandas as pd
import torch
from .Manager import FlightRankingModelsManager

class FlightRankingPredictor:
    def __init__(self, config: Dict, logger=None):
        self.config = config
        self.logger = logger
        
        self.data_path = Path(config['paths']['model_input_dir'])
        self.model_save_path = Path(config['paths']['model_save_dir'])
        self.output_path = Path(config['paths']['output_dir'])
        
        prediction_config = config['prediction']
        self.segments = prediction_config['segments']
        self.use_gpu = prediction_config['use_gpu'] and torch.cuda.is_available()
        
        self.best_models_config = self._load_best_models_config()
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        if self.use_gpu:
            self.logger.info(f"GPU Predictor: {torch.cuda.get_device_name(0)}")
        else:
            self.logger.info("CPU Predictor")
        
        self.logger.info(f"Best models: {self.best_models_config}")
    
    def _load_best_models_config(self) -> Dict:
        config_path = self.model_save_path / "best_models_config.json"
        
        if not config_path.exists():
            self.logger.warning("Config file not found, inferring from reports")
            return self._infer_best_models_from_reports()
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _infer_best_models_from_reports(self) -> Dict:
        best_models = {}
        for segment_id in self.segments:
            report_path = self.model_save_path / f"segment_{segment_id}" / "training_report.json"
            if report_path.exists():
                try:
                    with open(report_path, 'r') as f:
                        report = json.load(f)
                    best_model = report.get('best_model_name')
                    if best_model:
                        best_models[str(segment_id)] = best_model
                except Exception as e:
                    self.logger.warning(f"Failed to read report for segment_{segment_id}: {e}")
        return best_models
    
    def load_segment_best_model(self, segment_id: int) -> Tuple[FlightRankingModelsManager, str]:
        segment_dir = self.model_save_path / f"segment_{segment_id}"
        if not segment_dir.exists():
            raise FileNotFoundError(f"Segment dir not found: {segment_dir}")
        
        best_model_name = self.best_models_config.get(str(segment_id))
        if not best_model_name:
            raise ValueError(f"No best model config for segment_{segment_id}")
        
        model_file = segment_dir / f"{best_model_name}.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        models_manager = FlightRankingModelsManager(use_gpu=self.use_gpu, logger=self.logger)
        success = models_manager.load_model(best_model_name, str(segment_dir))
        
        if not success:
            raise ValueError(f"Failed to load {best_model_name} for segment_{segment_id}")
        
        self.logger.info(f"Loaded {best_model_name} for segment_{segment_id}")
        return models_manager, best_model_name
    
    def generate_rankings(self, scores: np.ndarray, groups: np.ndarray) -> np.ndarray:
        if torch.cuda.is_available() and self.use_gpu:
            try:
                scores_tensor = torch.FloatTensor(scores).cuda()
                groups_tensor = torch.LongTensor(groups).cuda()
                rankings = torch.zeros(len(scores), dtype=torch.long).cuda()
                
                unique_groups = torch.unique(groups_tensor)
                for group_id in unique_groups:
                    group_mask = groups_tensor == group_id
                    group_scores = scores_tensor[group_mask]
                    group_indices = torch.where(group_mask)[0]
                    
                    _, sort_indices = torch.sort(group_scores, descending=True)
                    group_rankings = torch.arange(1, len(group_scores) + 1, device=group_scores.device)
                    rankings[group_indices[sort_indices]] = group_rankings
                
                result = rankings.cpu().numpy()
                del scores_tensor, groups_tensor, rankings
                torch.cuda.empty_cache()
                return result
                
            except Exception as e:
                self.logger.warning(f"GPU ranking failed, using CPU: {e}")
                torch.cuda.empty_cache()
        
        # CPU fallback
        unique_groups = np.unique(groups)
        rankings = np.zeros(len(scores), dtype=int)
        
        for group_id in unique_groups:
            group_mask = groups == group_id
            group_scores = scores[group_mask]
            group_indices = np.where(group_mask)[0]
            
            sort_indices = np.argsort(-group_scores)
            group_rankings = np.arange(1, len(group_scores) + 1)
            rankings[group_indices[sort_indices]] = group_rankings
        
        return rankings
    
    def predict_segment(self, segment_id: int, save_individual: bool = False) -> Tuple:
        self.logger.info(f"Predicting segment_{segment_id}")
        start_time = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Load test data
        test_file = self.data_path / "test" / f"test_segment_{segment_id}.parquet"
        if not test_file.exists():
            raise FileNotFoundError(f"Test file not found: {test_file}")
        
        df = pd.read_parquet(test_file)
        
        # Load model
        models_manager, best_model_name = self.load_segment_best_model(segment_id)
        
        # Prepare data
        X, _, groups, _, _ = models_manager.prepare_data(df, target_col='selected', is_training=False)
        
        # Predict
        try:
            if best_model_name in ['GraphRanker', 'TransformerRanker']:
                predictions = models_manager.predict_model(best_model_name, X, groups)
            else:
                predictions = models_manager.predict_model(best_model_name, X)
            self.logger.info(f"Prediction completed: {len(predictions)} samples")
        except Exception as e:
            self.logger.error(f"✗ Prediction failed for {best_model_name}: {e}")
            # Fallback to random
            self.logger.warning("Using random prediction as fallback")
            predictions = np.random.random(len(X))
        
        # Generate rankings
        rankings = self.generate_rankings(predictions, groups)
        
        # Create results
        results = df[['Id', 'ranker_id']].copy()
        results['selected'] = rankings
        
        prediction_time = time.time() - start_time
        
        if save_individual:
            output_file = self.output_path / f"predictions_segment_{segment_id}.csv"
            results.to_csv(output_file, index=False)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info(f"✓ segment_{segment_id} predicted ({best_model_name}, {prediction_time:.1f}s)")
        
        return results, best_model_name, prediction_time
    
    def predict_all_segments(self) -> pd.DataFrame:
        self.logger.info(f"Predicting all segments: {self.segments}")
        
        all_results = []
        prediction_summary = {}
        total_start_time = time.time()
        successful_predictions = 0
        failed_predictions = 0
        
        for segment_id in self.segments:
            try:
                result, best_model_name, prediction_time = self.predict_segment(segment_id, save_individual=True)
                all_results.append(result)
                successful_predictions += 1
                
                prediction_summary[f'segment_{segment_id}'] = {
                    'best_model': best_model_name,
                    'prediction_time': prediction_time,
                    'samples_count': len(result),
                    'rankers_count': result['ranker_id'].nunique(),
                    'status': 'success'
                }
                
            except Exception as e:
                self.logger.error(f"✗ segment_{segment_id} prediction failed: {e}")
                failed_predictions += 1
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                prediction_summary[f'segment_{segment_id}'] = {
                    'status': 'failed',
                    'error': str(e)
                }
                
                # Random fallback
                try:
                    test_file = self.data_path / "test" / f"test_segment_{segment_id}.parquet"
                    if test_file.exists():
                        df = pd.read_parquet(test_file)
                        results = df[['Id', 'ranker_id']].copy()
                        
                        # Generate random rankings
                        rankings = np.zeros(len(df), dtype=int)
                        for group_id in df['ranker_id'].unique():
                            group_mask = df['ranker_id'] == group_id
                            group_size = group_mask.sum()
                            rankings[group_mask] = np.random.permutation(range(1, group_size + 1))
                        results['selected'] = rankings
                        
                        all_results.append(results)
                        prediction_summary[f'segment_{segment_id}']['status'] = 'random_fallback'
                        self.logger.warning(f"segment_{segment_id} using random fallback")
                except Exception:
                    continue
        
        if not all_results:
            raise ValueError("All segment predictions failed")
        
        # Combine results
        final_submission = pd.concat(all_results, ignore_index=True)
        final_submission = final_submission.sort_values('Id').reset_index(drop=True)
        
        total_time = time.time() - total_start_time
        output_file = self.output_path / "final_submission.csv"
        final_submission.to_csv(output_file, index=False)
        
        # Generate report
        self._generate_prediction_report(
            final_submission, prediction_summary, total_time, 
            successful_predictions, failed_predictions
        )
        
        self.logger.info(f"✓ All predictions completed ({total_time:.1f}s)")
        self.logger.info(f"✓ Final result: {output_file}")
        self.logger.info(f"✓ Total records: {len(final_submission):,}")
        self.logger.info(f"✓ Success rate: {successful_predictions}/{len(self.segments)}")
        
        return final_submission
    
    def _generate_prediction_report(self, results: pd.DataFrame, prediction_summary: Dict,
                                   total_time: float, successful_predictions: int, failed_predictions: int):
        total_segments = successful_predictions + failed_predictions
        
        # Model usage
        model_usage = {}
        for segment_info in prediction_summary.values():
            if segment_info['status'] in ['success', 'random_fallback']:
                model_name = segment_info.get('best_model', 'random')
                model_usage[model_name] = model_usage.get(model_name, 0) + 1
        
        # Stats
        total_samples = len(results)
        total_rankers = int(results['ranker_id'].nunique())
        group_sizes = results.groupby('ranker_id').size()
        
        report = {
            'prediction_summary': {
                'total_segments': total_segments,
                'successful_segments': successful_predictions,
                'failed_segments': failed_predictions,
                'success_rate': successful_predictions / total_segments if total_segments > 0 else 0,
                'total_time': total_time,
                'avg_time_per_segment': total_time / max(successful_predictions, 1)
            },
            'data_statistics': {
                'total_samples': total_samples,
                'total_rankers': total_rankers,
                'avg_options_per_ranker': total_samples / total_rankers if total_rankers > 0 else 0,
                'min_options': int(group_sizes.min()) if len(group_sizes) > 0 else 0,
                'max_options': int(group_sizes.max()) if len(group_sizes) > 0 else 0
            },
            'model_usage': model_usage,
            'segment_details': prediction_summary,
            'best_models_config': self.best_models_config
        }
        
        report_path = self.output_path / "prediction_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self.logger.info(f"\nPrediction Summary:")
        self.logger.info(f"  Success rate: {report['prediction_summary']['success_rate']:.1%}")
        self.logger.info(f"  Total samples: {total_samples:,}")
        self.logger.info(f"  Total rankers: {total_rankers:,}")
        
        if model_usage:
            self.logger.info(f"\nModel usage:")
            for model_name, count in sorted(model_usage.items(), key=lambda x: x[1], reverse=True):
                self.logger.info(f"  {model_name}: {count} segments")