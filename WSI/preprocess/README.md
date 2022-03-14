# Preprocess functions

Preprocess utilities for reading the SVS files and getting a single LMDB database with the tiles for each slide.

## Files

- **move_svs.py**: Move the SVS files from their original folders when downloaeded from TCGA to a single folder for later pre-processing.
    - **--dir**: Directory containing the hierarchy of folders ans vsvs files.
- **patch_gen_grid.py**: Generate patches from a given folder of SVS images.
    - **--wsi_path**: Path to the input directory of WSI files.
    - **--patch_path**: Path to the output directory of patch images.
    - **--mask_path**: Path to the  directory of numpy masks
    - **--patch_size**: Size of the patches
    - **----max_patches_per_slide**: Maximum number of patches to take from each slide.
    - **--dezoom_factor**: dezoom  factor, 1.0 means the images are taken at 20x magnification, 2.0 means the images are taken at 10x magnification.

## Examples of usage

```[bash]
python3 move_svs.py --dir ../TCGA-CESC

python3 patch_gen_grid.py --wsi_path ../TCGA-KIRP --patch_path ../TCGA-KIRP_Patches256x256 --patch_size 256 --mask_path ../TCGA-KIRP_Masks
```
        
