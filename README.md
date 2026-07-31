# Gold Price Prediction with Deep Learning (CNN, LSTM, CNN-LSTM)

Project (Computer Science, University of Tabriz) on forecasting the
price of gold (XAU/USD) using deep learning. Three architectures — a 1D CNN, an
LSTM, and a hybrid CNN-LSTM — are built, trained, and compared on 15-minute
XAU/USD price data.

## Overview

Gold prices are influenced by many factors (supply and demand, interest rates,
inflation, global economic conditions), which makes them a good candidate for
sequence-modeling approaches. This project:

1. Loads and cleans historical 15-minute XAU/USD OHLC data.
2. Splits the data into training, validation, and test sets.
3. Normalizes the data using a min-max scaling based on the harmonic mean of
   the OHLC average.
4. Builds sliding-window sequences (time step = 4) for supervised learning.
5. Defines and trains three models:
   - **CNN** — a 1D convolutional network for feature extraction.
   - **LSTM** — a recurrent network for learning temporal dependencies.
   - **CNN-LSTM** — a hybrid model combining convolutional feature extraction
     with LSTM sequence learning.
6. Evaluates and compares the models by inverse-transforming predictions back
   to the original price scale and plotting them against real values.

**Result:** the hybrid **CNN-LSTM** model gave the best overall performance,
followed by LSTM and CNN, at the cost of longer training time and higher
complexity.

## Repository structure

```
.
├── Normal_CNN_and_LSTM_XAUUSD.ipynb   # Main notebook: data prep, models, training, evaluation
├── README.md
├── requirements.txt
└── report/                            # (optional) project report (PDF)
```

## Data

- **Instrument:** XAU/USD (Gold vs. US Dollar)
- **Timeframe:** 15-minute candles
- **Fields used:** Date/Time, Open, High, Low, Close

The raw CSV is not included in this repository (see `.gitignore`). To
reproduce the results, place your own XAU/USD 15-minute OHLC CSV file
locally and update the file path in the first data-loading cell of the
notebook.

## Models

| Model | Layers |
|---|---|
| CNN | `Conv1D` → `Flatten` → `Dense` |
| LSTM | `LSTM` → `Dense` |
| CNN-LSTM | `Conv1D` → `LSTM` → `Dense` |

Common hyperparameters: 128 units, kernel size 2, batch size 7, 50 epochs,
learning rate 0.0001, loss = MSE, optimizers = Adam / SGD / RMSprop.

## Getting started

```bash
git clone https://github.com/sanfand/Gold_Price_Prediction_Deep_Learning.git
cd Gold_Price_Prediction_Deep_Learning
pip install -r requirements.txt
jupyter notebook Normal_CNN_and_LSTM_XAUUSD.ipynb
```

Update the CSV path in the notebook to point to your local copy of the
XAU/USD 15-minute dataset, then run the cells in order.

## References

- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural
  Computation*, 9(8), 1735–1780.
- LeCun, Y., Bengio, Y., & Hinton, G. (2015). Deep learning. *Nature*,
  521(7553), 436–444.
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT
  Press.
- Chollet, F. (2017). *Deep Learning with Python*. Manning Publications.
- [Keras Sequential model guide](https://keras.io/guides/sequential_model/)

## Author

Sana Farahmand — Computer Science, University of Tabriz
Supervisor: Dr. Jaber Karimpour
