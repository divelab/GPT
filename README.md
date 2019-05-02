# Label-free Prediction

This is the code for our recent work that develops a new method for label-free prediction from fluorescent images.
The code is created and modified based upon the work from Google.The code to their work is at
https://github.com/google/in-silico-labeling.


## Citing

If you use this code, please consider citing the original isl paper:

```bibtex
@article{christiansen2018isl,
  title={In silico labeling: Predicting fluorescent labels in unlabeled images},
  author={Christiansen, Eric M and Yang, Samuel J and Ando, D Michael and Javaherian, Ashkan and Skibinski, Gaia and Lipnick, Scott and Mount, Elliot and O’Neil, Alison and Shah, Kevan and Lee, Alicia K and Goyal, Piyush and Fedus, William and Poplin, Ryan and Esteva, Andre and Berndl, Marc and Rubin, Lee L and Nelson, Philip and Finkbeiner, Steven},
  journal={Cell},
  year={2018},
  publisher={Elsevier}
}
```
Citing bibtex for our work will be avialable upon the publishing of our paper.

## Dependencies
We have tested this code using:
* Ubuntu 18.04
* Python 3
* NumPy
* TensorFlow
* OpenCV

## Data
Data is available at https://github.com/google/in-silico-labeling/blob/master/data.md.


## Train and test

    python gunet\launch.py -- \
      --alsologtostderr \
      --base_directory $BASE_DIRECTORY \
      --mode EVAL_EVAL \
      --metric INFER_FULL \
      --stitch_crop_size 1500 \
      --restore_directory $(pwd)/checkpoints \
      --read_pngs \
      --dataset_eval_directory $(pwd)/data_sample/condition_b_sample \
      --infer_channel_whitelist DAPI_CONFOCAL,MAP2_CONFOCAL,NFH_CONFOCAL

In the above:

1.  `BASE_DIRECTORY` is the working directory for the model. It will be created
    if it doesn't already exist, and it's where the model predictions will be
    written. You can set it to whatever you want.
1.  `alsologtostderr` will cause progress information to be printed to the
    terminal.
1.  `stitch_crop_size` is the size of the crop for which we'll perform
    inference. If set to 1500 it may take an hour on a single machine, so try
    smaller numbers first.
1.  `infer_channel_whitelist` is the list of fluorescence channels we wish to
    infer. For the Condition B data, this should be a subset of `DAPI_CONFOCAL`,
    `MAP2_CONFOCAL`, and `NFH_CONFOCAL`.

## Licence
The code is MIT licensed, as found in the LICENSE file.
