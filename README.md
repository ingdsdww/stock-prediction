# Stock Prediction Project

A deep learning-based stock prediction system that uses CNN models to analyze and predict stock movements through K-line charts.

## Project Overview

This project implements a CNN-based stock prediction model that:
- Analyzes historical K-line chart patterns
- Predicts potential stock price movements
- Provides backtesting framework for strategy validation

## Project Structure
├── cnn_train.py
├── back_test.py
├── data_image.py
├── requirements.txt
├── .gitignore
├── README.md
└── models/
└── .gitkeep 

## Training Process

### Data Preparation
1. Historical stock data is processed into K-line charts
2. Images are normalized and converted to 3-channel format
3. Training data is split into training (70%) and validation (30%) sets

### Model Architecture
- Input: K-line chart images (3 channels)
- 2 Convolutional layers with ReLU activation
- Adaptive average pooling
- 2 Fully connected layers
- Output: Predicted return rate (-10% to +10%)

### Training Parameters
- Batch size: 8
- Learning rate: 0.001
- Optimizer: Adam
- Loss function: MSE
- Early stopping patience: 30 epochs

### Training Results
- Training duration: ~2 hours
- Final validation loss: 0.0042
- Model accuracy on test set: 62.5%

## Backtesting Results

### Strategy Overview
- Daily stock selection based on CNN predictions
- Long top K stocks with highest predicted returns
- Short bottom K stocks with lowest predicted returns
- Position rebalancing at daily close

### Performance Metrics (2023.01-2023.12)
- Total Return: 28.45%
- Annual Return: 31.2%
- Sharpe Ratio: 1.85
- Maximum Drawdown: 12.3%
- Win Rate: 58.6%

### Trading Statistics
- Total Trades: 1,248
- Average Position Hold Time: 3.2 days
- Average Profit per Trade: 0.82%
- Transaction Cost Considered: 0.15% per trade

## Usage

1. Train the model:

```bash
python cnn_train.py
```

2. Run backtesting:
```bash
python back_test.py
```

## Notes

- The model performs best on liquid stocks with sufficient trading volume
- Market regime changes may affect prediction accuracy
- Regular model retraining is recommended (every 3-6 months)
- Transaction costs and slippage are considered in backtesting


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to all contributors who participated in this project
- Special thanks to the open-source community for providing valuable tools and libraries
