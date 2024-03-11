# S3AI
## Installation

We highly recommand that you use Anaconda for Installation
```
conda create -n S3AI
conda activate S3AI
pip install -r requirements.txt
```

## Data
The Sars-cov2 IC50 data is in the `data` folder.
* `data/updated_processed_data.csv` is the paired Ab-Ag data.
* `data/Ag_sequence.csv` is the Ag sequence data.

## Model inference 
### Download checkpoint
Download the checkpoint of S3AI from [here](10.6084/m9.figshare.25378708) and modify the paths in the code.

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

