This directory contains:
- 'utils.py' where we have a variety different helping functions to help us with processing.
- 'models.py' which consists of the definitions of the pytorch models we trained
- 'predict.py' which is used to run the trained models we provide on new datapoints
- 'notebooks' is a directory containing ipython notebooks for data exploration/vizualization

To run the models on test data with gold labels follow this example:

python predict.py --model_path cnn_adam_l1.pt --data_dir test_data/

To write out the predictions to stdout, per test instance,  use the --generate flag.

## Package requirments:
pytorch == 1.10.2
numpy == 1.20.3
scikit-learn == 0.22.2.post1
tqdm == 4.62.3
wandb == 0.12.4
pandas == 1.0.3

