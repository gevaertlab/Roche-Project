# Source code for the training of the models

Source python files for the training of the models and other miscellaneous tasks.

## Files

- **create_ref\*.py**: Creation of the reference files for the training of the models. These files contain the name of the WSI file, the RNA values (if available), the label of the sample and the TCGA project.
- **fusion.py**: Fusion model and main script for its training: 
    - **--config**: config file with training parameters.
    - **--checkpoint**: if a model checkpoint is provided it will be loaded.
    - **--seed**: random seed to use.
    - **--log**: use tensorboard for experiment logging.
    - **--parallel**: use DataParallel training.
    - **--fp16**: use mixed-precision training.- **fusion_utils.py**: utility functions to train and evaluate the fusion model.
- **fusion_utils.py**: utility functions to train and evaluate the fusion model.
- **heatmap.py**: File to generate the visualization of the model ona given slide.
    - **--wsi_path**: path to the WSI slide.
    - **--patch_size**: patch size.
    - **dezzom_factor**: dezoom  factor, 1.0 means the images are taken at 20x magnification, 2.0 means the images are taken at 10x magnification.
    - **--checkpoint**: path to the checkpoint of the image model.
    - **--cuda**: whether to use cuda or not.
    - **--problem**: problem type to load the classes.
    - **--grad_cam**: whether to use grad\_cam of the heatmap.
    - **--label**: the real label of the sample used.
    - **--output_dir**: path to save generated files.
    - **--suffix**: suffix for the image to be saved.
    - **--regression**: treating the prediction as a regression problem.
    - **--thresholds**: thresholds to use in case we are treating it as a regression problem.
- **main.py**: file for training the wsi models.
    - **--config**: json config file containing training information.
    - **--checkpoint:**: path to checkpoint.
    - **--save_dir**: path to folder where the checkpoints are saved.
    - **--flag**: flag to use for saving the checkpoints.
    - **--seed**: random seed to use.
    - **--log**: use tensorboard for experiment logging.
    - **--parallel**: use dataparallel training.
    - **--fp16**: use mixed-precision training.
    - **--bag_size**: number of tiles to use in the bag.
    - **--max_patch_per_wsi**: maximum number of tiles to use per slide.
    - **--class_weights**: whether to use weights on the cross-entropy loss
    - **--focal_loss**: use the focal loss.
    - **--attention**: use attention on the classification model.
    - **--over_sampling**: over-sample the minority class.
- **read_data.py**: classes and function to read the datasets.
- **resnet.py** implementation of the resnet network.
- **test_panda.py**: evaluation of a model on the panda challange dataset.
- **wsi_model.py**: wsi model architecture and training and evaluation functions.
