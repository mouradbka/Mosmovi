## Introduction
This directory contains:
- 'utils.py' where we have a variety different helping functions to help us with processing.
- 'model_utils.py' has our Mixture Density Network (Bishop, 1994) implementation.
- 'models.py' which consists of the definitions of the pytorch models we trained
- 'predict.py' which is used to run the trained models we provide on new datapoints
- 'notebooks' is a directory containing ipython notebooks for data exploration/vizualization

To train your own model, choose the appropriate set of training flags from `runner.py`; explanations for each flag are in the file. 
To recreate the submitted model, run:

```
python runner.py --data_dir <data_path> --arch char_lstm --mdn --split_uids --subsample_ratio 1.0 --batch_size 128 --optimizer adamw  
--save_prefix <model_path> --lr 5e-4 --loss l1 --num_epochs 25 
--num_gaussians 30 --reg_penalty 0.001 --entropy_loss_weight 0.05 --confidence_validation_criterion
```

To run the models on test data with gold labels follow this example (for an MDN model):

```
python predict.py --data_dir <data_path> --model_path <model_path> --arch char_lstm --mdn --batch_size 128
```

## Model confidence
To write out the predictions to stdout, per test instance,  use the `--generate` flag. For printing confidences, 
both training and evaluation code should use the `--mdn` flag.

MDN models output, along with their predictions, the confidence of that prediction in a separate column. 
The number of confidence levels needed are specified with the flag `--num_confidence_bins`, set to a default of 5 (we recommend values between 5 and 10).
For both evaluation and generation, a higher confidence level implies that the model is more confident about the prediction for that subset of examples.
The output should look something like:

```
> test_mean: 1245.8035673871632, test_median: 520.50927734375
> conf: 0 - val_mean: 1794.3510998091208
> conf: 0 - val_median: 1105.5209350585938
> conf: 1 - val_mean: 1398.6538583901156
> conf: 1 - val_median: 905.3485107421875
> conf: 2 - val_mean: 1142.7724703758329
> conf: 2 - val_median: 377.4914245605469
> conf: 3 - val_mean: 1108.8275716482829
> conf: 3 - val_median: 274.7439270019531
> conf: 4 - val_mean: 746.1342747807423
> conf: 4 - val_median: 223.4276580810547
```

## Package requirements:
```
pytorch == 1.10.2
numpy == 1.20.3
scikit-learn == 0.22.2.post1
tqdm == 4.62.3
wandb == 0.12.4
pandas == 1.0.3
```