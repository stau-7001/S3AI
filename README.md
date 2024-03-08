## Installation

We highly recommand that you use Anaconda for Installation
```
conda create -n S3AI
conda activate S3AI
pip install -r requirements.txt
```

## Data
The Sars-cov2 IC50 data is in the `data` folder.
* `data/` is the paired Ab-Ag data.
* `data/` is the Ag sequence data.

## Model inference 

To test S3AI on Sars-cov2 IC50 test data, please run
```
python main.py --config=configs/test_on_sarscov2.yml
```

## Model training

To train S3AI, please run
```
python main.py --config=configs/train.yml
```

## License

This project is licensed under the [MIT License](LICENSE).

