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

To write out the predictions to stdout, per test instance,  use the `--generate` flag. For printing confidences, 
both training and evaluation code should use the `--mdn` flag.

## Package requirements:
```
pytorch == 1.10.2
numpy == 1.20.3
scikit-learn == 0.22.2.post1
tqdm == 4.62.3
wandb == 0.12.4
pandas == 1.0.3
```