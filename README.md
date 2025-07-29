<<<<<<< HEAD
# Flight Ranking Competition - Business Travel Recommendation System ✈️

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org/)
[![Competition](https://img.shields.io/badge/competition-Kaggle-20BEFF.svg)](https://kaggle.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://claude.xiaoai.shop/chat/LICENSE)

## 🎯 Project Overview

This project tackles the challenge of building an intelligent flight ranking model that predicts which flight option business travelers will choose from search results. Unlike leisure travelers who primarily focus on cost, business travelers must balance multiple competing factors: corporate travel policies, meeting schedules, expense compliance, and personal convenience.

### Competition Goal

* **Objective** : Build a group-wise ranking model to predict business traveler flight preferences
* **Problem Type** : Group-wise ranking problem within user search sessions
* **Evaluation Metric** : HitRate@3 - fraction of sessions where correct flight appears in top-3 predictions
* **Dataset Size** : 18M+ training samples, 6.9M+ test samples

## 🏆 Competition Details

### Key Challenges

* **Complex Decision Patterns** : Business travelers balance policies, schedules, and convenience
* **Variable Group Sizes** : Handle 10-1000+ flight options per search session
* **Sparse Signal** : Only one positive example per group (selected flight)
* **Real-world Scale** : Dataset contains actual flight search sessions

### Evaluation Criteria

* **Primary Metric** : HitRate@3 (only groups with >10 options)
* **Score Range** : 0 to 1 (1 = correct flight always in top-3)
* **Bonus Threshold** : HitRate@3 ≥ 0.7 doubles prize money

## 📊 Dataset Structure

### Main Files

* `train.parquet` - Training data (18,145,372 rows)
* `test.parquet` - Test data (6,897,776 rows)
* `sample_submission.parquet` - Submission format example
* `jsons_raw.tar.kaggle` - Raw JSON data (150K files, ~50GB)

### Key Features

#### User & Company Information

* User demographics, VIP status, frequent flyer programs
* Corporate tariff codes and travel policies
* Booking independence indicators

#### Flight Details

* Pricing: Total price, taxes, policy compliance
* Timing: Departure/arrival times, duration
* Route: Airports, airlines, aircraft types
* Service: Cabin class, baggage allowance, seat availability
* Flexibility: Cancellation/exchange rules and penalties

#### Search Context

* Search route (single/round trip)
* Request timestamp
* Session grouping (`ranker_id`)

## 🏗️ Technical Architecture

### Project Structure

```
flight-ranking-system/
├── bin/
│   └── start.sh              # Interactive startup script
├── src/
│   ├── main.py              # Main application entry
│   ├── core/
│   │   ├── data_processor.py # Data loading and preprocessing
│   │   ├── feature_engineer.py # Feature engineering pipeline
│   │   ├── model_trainer.py  # Model training and validation
│   │   └── predictor.py     # Prediction and ranking
│   ├── models/
│   │   ├── xgb_ranker.py    # XGBoost ranking models
│   │   ├── lgb_ranker.py    # LightGBM ranking models
│   │   └── ensemble.py      # Model ensemble methods
│   └── utils/
│       ├── metrics.py       # Evaluation metrics
│       ├── validation.py    # Cross-validation strategies
│       └── preprocessing.py # Data preprocessing utilities
├── config/
│   ├── conf.yaml           # Main configuration
│   └── model_configs/      # Model-specific configs
├── data/
│   ├── raw/                # Original competition data
│   ├── processed/          # Engineered features
│   └── submissions/        # Generated submissions
├── models/
│   ├── trained/           # Saved model artifacts
│   └── checkpoints/       # Training checkpoints
├── logs/                  # Execution logs
├── notebooks/             # Jupyter notebooks for EDA
└── requirements.txt       # Python dependencies
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/dishijuntian/FR.git
cd FR
pip install -r requirements.txt
```

### Basic Usage

#### 1. Default Full Pipeline

```bash
./bin/start.sh
```

#### 2. Interactive Menu Mode

```bash
./bin/start.sh --interactive
```

#### 3. Command Line Options

```bash
# Training only
./bin/start.sh --mode training

# Custom configuration
./bin/start.sh --config custom_config.yaml

# Force reprocessing
./bin/start.sh --force

# Show help
./bin/start.sh --help
```

### Interactive Menu Features

The startup script provides user-friendly menus:

**Main Menu:**

1. Complete pipeline execution (default)
2. Data processing only
3. Model training only
4. Model prediction only
5. System status check
6. Advanced options
7. Exit

**Advanced Options:**

* Force data reprocessing
* Custom configuration files
* Data segment specification
* Model selection
* Skip data validation
* Verbose output mode

## 📈 Technical Challenges & Solutions

### 1. Ranking Problem Complexity

* **Challenge** : Converting classification/regression to ranking
* **Solution** : Use ranking-aware loss functions and evaluation metrics

### 2. Variable Group Sizes

* **Challenge** : Handle groups from 10-1000+ options
* **Solution** : Scalable models and efficient ranking algorithms

### 3. Sparse Signal

* **Challenge** : Only 1 positive example per group
* **Solution** : Careful sampling and augmentation strategies

### 4. Business Logic Integration

* **Challenge** : Capture complex business travel preferences
* **Solution** : Domain expertise and interpretable features

### 5. Computational Efficiency

* **Challenge** : Large dataset with complex features
* **Solution** : Efficient algorithms and parallel processing

## 🔧 Configuration

### Main Configuration (`config/conf.yaml`)

```yaml
# Flight Ranking System Configuration

paths:
  data_dir: "data/aeroclub-recsys-2025"
  model_save_dir: "data/aeroclub-recsys-2025/models"
  output_dir: "data/aeroclub-recsys-2025/submissions"
  log_dir: "logs"

data_processing:
  chunk_size: 200000
  n_processes: null
  force_reprocess: false
  verify_results: true

training:
  segments: [0, 1, 2]
  use_gpu: true
  random_state: 42
  
  model_params:
    xgboost:
      n_estimators: 200
      max_depth: 8
      learning_rate: 0.05
      verbosity: 0
  
    lightgbm:
      n_estimators: 200
      max_depth: 8
      learning_rate: 0.05
      verbose: -1

prediction:
  segments: [0, 1, 2]
  model_name: "XGBRanker"
  use_gpu: true
  random_state: 42

pipeline:
  run_data_processing: true
  run_training: true
  run_prediction: true
  continue_on_failure: false

logging:
  level: "INFO"
  format: "%(asctime)s | %(levelname)8s | %(name)s | %(message)s"
  datefmt: "%Y-%m-%d %H:%M:%S"

advanced:
  memory_optimization: true
  parallel_optimization: true
  cleanup_temp_files: true

validation:
  check_data_integrity: true
  validate_model_performance: true
  validate_predictions: true

monitoring:
  enable_monitoring: true
  monitor_memory: true
  monitor_timing: true
```

## 📊 Evaluation & Metrics

### Primary Metric: HitRate@3

```python
def hitrate_at_3(y_true_ranks, y_pred_ranks):
    """
    Calculate HitRate@3 for ranking predictions
    Only considers groups with >10 options
    """
    correct_in_top3 = 0
    total_groups = 0
  
    for group in groups:
        if len(group) > 10:  # Filter condition
            total_groups += 1
            true_rank = get_true_rank(group)
            pred_rank = get_predicted_rank(group)
            if pred_rank <= 3:
                correct_in_top3 += 1
  
    return correct_in_top3 / total_groups
```

## 💻 Resource Requirements

### Software Stack

* **Data Processing** : pandas, numpy, polars
* **ML Libraries** : scikit-learn, xgboost, lightgbm
* **Deep Learning** : tensorflow/pytorch, transformers
* **Visualization** : matplotlib, seaborn, plotly
* **Ranking** : rankpy, xlearn, tensorflow-ranking

## 🚨 Risk Management

### Technical Risks

* **Data Quality Issues** : Implement robust validation
* **Overfitting** : Strong cross-validation strategy
* **Computational Limits** : Efficient algorithms and sampling
* **Model Complexity** : Start simple, increase complexity gradually

## 📝 Submission Format

### Training Data

```csv
Id,ranker_id,selected
100,abc123,0     # Flight option 1 - not chosen
101,abc123,0     # Flight option 2 - not chosen
102,abc123,1     # Flight option 3 - SELECTED by user
103,abc123,0     # Flight option 4 - not chosen
```

### Submission Format

```csv
Id,ranker_id,selected
100,abc123,4     # Rank 4 (worst option)
101,abc123,2     # Rank 2 (second best)
102,abc123,1     # Rank 1 (best - correct prediction!)
103,abc123,3     # Rank 3 (third best)
```

### Validation Requirements

* Preserve exact row order as test.csv
* Complete rankings for all flight options
* Valid permutations (1, 2, 3, ..., N) within each group
* No duplicate ranks per search session
* Integer values ≥ 1

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](License) file for details.

## 🎯 Competition Timeline

* **Data Exploration** : Weeks 1-2
* **Feature Engineering** : Weeks 2-3
* **Model Development** : Weeks 3-4
* **Advanced Techniques** : Weeks 4-5
* **Final Optimization** : Weeks 5-6
* **Submission Deadline** : [Competition End Date]

## 📞 Contact

* **Repository** : [https://github.com/dishijuntian/FR.git](https://github.com/dishijuntian/FR.git)
* **Issues** : Use GitHub Issues for bug reports and feature requests
* **Discussions** : Use GitHub Discussions for general questions
=======
# Flight Ranking Competition - Business Travel Recommendation System ✈️

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org/)
[![Competition](https://img.shields.io/badge/competition-Kaggle-20BEFF.svg)](https://kaggle.com/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://claude.xiaoai.shop/chat/LICENSE)

## 🎯 Project Overview

This project tackles the challenge of building an intelligent flight ranking model that predicts which flight option business travelers will choose from search results. Unlike leisure travelers who primarily focus on cost, business travelers must balance multiple competing factors: corporate travel policies, meeting schedules, expense compliance, and personal convenience.

### Competition Goal

* **Objective** : Build a group-wise ranking model to predict business traveler flight preferences
* **Problem Type** : Group-wise ranking problem within user search sessions
* **Evaluation Metric** : HitRate@3 - fraction of sessions where correct flight appears in top-3 predictions
* **Dataset Size** : 18M+ training samples, 6.9M+ test samples

## 🏆 Competition Details

### Key Challenges

* **Complex Decision Patterns** : Business travelers balance policies, schedules, and convenience
* **Variable Group Sizes** : Handle 10-1000+ flight options per search session
* **Sparse Signal** : Only one positive example per group (selected flight)
* **Real-world Scale** : Dataset contains actual flight search sessions

### Evaluation Criteria

* **Primary Metric** : HitRate@3 (only groups with >10 options)
* **Score Range** : 0 to 1 (1 = correct flight always in top-3)
* **Bonus Threshold** : HitRate@3 ≥ 0.7 doubles prize money

## 📊 Dataset Structure

### Main Files

* `train.parquet` - Training data (18,145,372 rows)
* `test.parquet` - Test data (6,897,776 rows)
* `sample_submission.parquet` - Submission format example
* `jsons_raw.tar.kaggle` - Raw JSON data (150K files, ~50GB)

### Key Features

#### User & Company Information

* User demographics, VIP status, frequent flyer programs
* Corporate tariff codes and travel policies
* Booking independence indicators

#### Flight Details

* Pricing: Total price, taxes, policy compliance
* Timing: Departure/arrival times, duration
* Route: Airports, airlines, aircraft types
* Service: Cabin class, baggage allowance, seat availability
* Flexibility: Cancellation/exchange rules and penalties

#### Search Context

* Search route (single/round trip)
* Request timestamp
* Session grouping (`ranker_id`)

## 🏗️ Technical Architecture

### Project Structure

```
flight-ranking-system/
├── bin/
│   └── start.sh              # Interactive startup script
├── src/
│   ├── main.py              # Main application entry
│   ├── core/
│   │   ├── data_processor.py # Data loading and preprocessing
│   │   ├── feature_engineer.py # Feature engineering pipeline
│   │   ├── model_trainer.py  # Model training and validation
│   │   └── predictor.py     # Prediction and ranking
│   ├── models/
│   │   ├── xgb_ranker.py    # XGBoost ranking models
│   │   ├── lgb_ranker.py    # LightGBM ranking models
│   │   └── ensemble.py      # Model ensemble methods
│   └── utils/
│       ├── metrics.py       # Evaluation metrics
│       ├── validation.py    # Cross-validation strategies
│       └── preprocessing.py # Data preprocessing utilities
├── config/
│   ├── conf.yaml           # Main configuration
│   └── model_configs/      # Model-specific configs
├── data/
│   ├── raw/                # Original competition data
│   ├── processed/          # Engineered features
│   └── submissions/        # Generated submissions
├── models/
│   ├── trained/           # Saved model artifacts
│   └── checkpoints/       # Training checkpoints
├── logs/                  # Execution logs
├── notebooks/             # Jupyter notebooks for EDA
└── requirements.txt       # Python dependencies
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/dishijuntian/FR.git
cd FR
pip install -r requirements.txt
```

### Basic Usage

#### 1. Default Full Pipeline

```bash
./bin/start.sh
```

#### 2. Interactive Menu Mode

```bash
./bin/start.sh --interactive
```

#### 3. Command Line Options

```bash
# Training only
./bin/start.sh --mode training

# Custom configuration
./bin/start.sh --config custom_config.yaml

# Force reprocessing
./bin/start.sh --force

# Show help
./bin/start.sh --help
```

### Interactive Menu Features

The startup script provides user-friendly menus:

**Main Menu:**

1. Complete pipeline execution (default)
2. Data processing only
3. Model training only
4. Model prediction only
5. System status check
6. Advanced options
7. Exit

**Advanced Options:**

* Force data reprocessing
* Custom configuration files
* Data segment specification
* Model selection
* Skip data validation
* Verbose output mode

## 📈 Technical Challenges & Solutions

### 1. Ranking Problem Complexity

* **Challenge** : Converting classification/regression to ranking
* **Solution** : Use ranking-aware loss functions and evaluation metrics

### 2. Variable Group Sizes

* **Challenge** : Handle groups from 10-1000+ options
* **Solution** : Scalable models and efficient ranking algorithms

### 3. Sparse Signal

* **Challenge** : Only 1 positive example per group
* **Solution** : Careful sampling and augmentation strategies

### 4. Business Logic Integration

* **Challenge** : Capture complex business travel preferences
* **Solution** : Domain expertise and interpretable features

### 5. Computational Efficiency

* **Challenge** : Large dataset with complex features
* **Solution** : Efficient algorithms and parallel processing

## 🔧 Configuration

### Main Configuration (`config/conf.yaml`)

```yaml
# Flight Ranking System Configuration

paths:
  data_dir: "data/aeroclub-recsys-2025"
  model_save_dir: "data/aeroclub-recsys-2025/models"
  output_dir: "data/aeroclub-recsys-2025/submissions"
  log_dir: "logs"

data_processing:
  chunk_size: 200000
  n_processes: null
  force_reprocess: false
  verify_results: true

training:
  segments: [0, 1, 2]
  use_gpu: true
  random_state: 42
  
  model_params:
    xgboost:
      n_estimators: 200
      max_depth: 8
      learning_rate: 0.05
      verbosity: 0
  
    lightgbm:
      n_estimators: 200
      max_depth: 8
      learning_rate: 0.05
      verbose: -1

prediction:
  segments: [0, 1, 2]
  model_name: "XGBRanker"
  use_gpu: true
  random_state: 42

pipeline:
  run_data_processing: true
  run_training: true
  run_prediction: true
  continue_on_failure: false

logging:
  level: "INFO"
  format: "%(asctime)s | %(levelname)8s | %(name)s | %(message)s"
  datefmt: "%Y-%m-%d %H:%M:%S"

advanced:
  memory_optimization: true
  parallel_optimization: true
  cleanup_temp_files: true

validation:
  check_data_integrity: true
  validate_model_performance: true
  validate_predictions: true

monitoring:
  enable_monitoring: true
  monitor_memory: true
  monitor_timing: true
```

## 📊 Evaluation & Metrics

### Primary Metric: HitRate@3

```python
def hitrate_at_3(y_true_ranks, y_pred_ranks):
    """
    Calculate HitRate@3 for ranking predictions
    Only considers groups with >10 options
    """
    correct_in_top3 = 0
    total_groups = 0
  
    for group in groups:
        if len(group) > 10:  # Filter condition
            total_groups += 1
            true_rank = get_true_rank(group)
            pred_rank = get_predicted_rank(group)
            if pred_rank <= 3:
                correct_in_top3 += 1
  
    return correct_in_top3 / total_groups
```

## 💻 Resource Requirements

### Software Stack

* **Data Processing** : pandas, numpy, polars
* **ML Libraries** : scikit-learn, xgboost, lightgbm
* **Deep Learning** : tensorflow/pytorch, transformers
* **Visualization** : matplotlib, seaborn, plotly
* **Ranking** : rankpy, xlearn, tensorflow-ranking

## 🚨 Risk Management

### Technical Risks

* **Data Quality Issues** : Implement robust validation
* **Overfitting** : Strong cross-validation strategy
* **Computational Limits** : Efficient algorithms and sampling
* **Model Complexity** : Start simple, increase complexity gradually

## 📝 Submission Format

### Training Data

```csv
Id,ranker_id,selected
100,abc123,0     # Flight option 1 - not chosen
101,abc123,0     # Flight option 2 - not chosen
102,abc123,1     # Flight option 3 - SELECTED by user
103,abc123,0     # Flight option 4 - not chosen
```

### Submission Format

```csv
Id,ranker_id,selected
100,abc123,4     # Rank 4 (worst option)
101,abc123,2     # Rank 2 (second best)
102,abc123,1     # Rank 1 (best - correct prediction!)
103,abc123,3     # Rank 3 (third best)
```

### Validation Requirements

* Preserve exact row order as test.csv
* Complete rankings for all flight options
* Valid permutations (1, 2, 3, ..., N) within each group
* No duplicate ranks per search session
* Integer values ≥ 1

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](License) file for details.

## 🎯 Competition Timeline

* **Data Exploration** : Weeks 1-2
* **Feature Engineering** : Weeks 2-3
* **Model Development** : Weeks 3-4
* **Advanced Techniques** : Weeks 4-5
* **Final Optimization** : Weeks 5-6
* **Submission Deadline** : [Competition End Date]

## 📞 Contact

* **Repository** : [https://github.com/dishijuntian/FR.git](https://github.com/dishijuntian/FR.git)
* **Issues** : Use GitHub Issues for bug reports and feature requests
* **Discussions** : Use GitHub Discussions for general questions
>>>>>>> e9addff7041905ba228e279124f71a4a54b1d4e3
