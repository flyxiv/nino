SPRITE_DIRECTIONS = ['up', 'upright', 'right', 'downright', 'down', 'downleft', 'left', 'upleft']
SPRITE_DIRECTION_MOTIONS = ['idle', 'walk', 'run', 'attack', 'laydown', 'swim', 'jump']
SPRITE_NO_DIRECTION_MOTIONS = ['climb']

import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str, required=True, help='directory to create sprite folders')

    return parser.parse_args()

def create_sprite_folders(outdir):
    for direction in SPRITE_DIRECTIONS:
        for motion in SPRITE_DIRECTION_MOTIONS:
            folder_name = f'{motion}_{direction}'
            print(f'Creating folder: {folder_name}')

            if not os.path.exists(os.path.join(outdir, folder_name)):
                os.makedirs(os.path.join(outdir, folder_name), exist_ok=True)

    for motion in SPRITE_NO_DIRECTION_MOTIONS:
        folder_name = f'{motion}'
        if not os.path.exists(os.path.join(outdir, folder_name)):
            os.makedirs(os.path.join(outdir, folder_name), exist_ok=True)

if __name__ == '__main__':
    args = parse_arguments()
    create_sprite_folders(args.outdir)
    
