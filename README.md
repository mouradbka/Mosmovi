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
2022-03-23 16:08:18,243 test_mean: 1378.4861550968005, test_median: 359.2927551269531
2022-03-23 16:08:18,244 conf: 0 - val_mean: 2566.2339115645077
2022-03-23 16:08:18,244 conf: 0 - val_median: 2955.72021484375
2022-03-23 16:08:18,298 conf: 1 - val_mean: 1318.2091678031566
2022-03-23 16:08:18,299 conf: 1 - val_median: 1105.283203125
2022-03-23 16:08:18,400 conf: 2 - val_mean: 1501.5972616800073
2022-03-23 16:08:18,400 conf: 2 - val_median: 276.8335876464844
2022-03-23 16:08:18,435 conf: 3 - val_mean: 1342.0475045738883
2022-03-23 16:08:18,435 conf: 3 - val_median: 258.83815002441406
2022-03-23 16:08:18,444 conf: 4 - val_mean: 454.0487183505631
2022-03-23 16:08:18,444 conf: 4 - val_median: 173.92791748046875
```

Note that confidences are calculated at a batch level, for scalability (i.e. more accurate sample statistic estimates) --
therefore, while evaluating, we recommend setting the batch size as high as possible (default: 1024), given memory constraints.

We also recommend using the 'entropy_confidence' flag to for the calculation of confidence scores. We found this to
lead to more consistent results, compared to the alternative which relies on the maximum Probablity instead of the entropy
of the distribution.

## Package requirements:
```
pytorch == 1.10.2
numpy == 1.20.3
scikit-learn == 0.22.2.post1
tqdm == 4.62.3
wandb == 0.12.4
pandas == 1.0.3
```