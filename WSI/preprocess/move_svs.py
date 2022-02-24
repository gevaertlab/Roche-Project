from glob import glob
import os
import argparse
from tqdm import tqdm
import shutil

parser = argparse.ArgumentParser(description='Move svs files and delete their original folder')
parser.add_argument('--dir', type=str, help='Dir with the WSI subfolders')

args = parser.parse_args()


files = glob(args.dir + '/**/*.svs')
for file in tqdm(files):
    file_vector = file.split('/')
    if len(file_vector) != 4:
        print(file_vector)
        import pdb; pdb_.set_trace()
    name = file_vector[-1]
    # moving file to the root directory
    print('Moving {}'.format(name))
    new_path = file_vector[0] + '/' + file_vector[1] + '/' + name
    shutil.move(file, new_path)

    # remove directory
    shutil.rmtree(file_vector[0] + '/' + file_vector[1] + '/' + file_vector[2])

