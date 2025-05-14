SPRITE_TABLE = {0: 'portraits', 1: 'idle_down', 2: 'idle_downleft', 3: 'idle_downright', 4: 'idle_left', 5: 'idle_right', 6: 'idle_up', 7: 'idle_upleft', 8: 'idle_upright', 9: 'run_down', 10: 'run_downleft', 11: 'run_downright', 12: 'run_left', 13: 'run_right', 14: 'run_up', 15: 'run_upleft', 16: 'run_upright', 17: 'sit_down', 18: 'sit_downleft', 19: 'sit_downright', 20: 'sit_left', 21: 'sit_right', 22: 'sit_up', 23: 'sit_upleft', 24: 'sit_upright', 25: 'others'}
SPRITE_IDS = {v: k for (k, v) in SPRITE_TABLE.items()}

SPRITE_NO_DIRECTIONS = {0: 'portrait', 1: 'idle', 2: 'run', 3: 'sit', 4: 'others'}
SPRITE_NO_DIRECTION_IDS = {v: k for (k, v) in SPRITE_NO_DIRECTIONS.items()} 

SPRITE_DIRECTIONS = ['up', 'upright', 'right', 'downright', 'down', 'downleft', 'left', 'upleft']
SPRITE_DIRECTION_MOTIONS = ['idle', 'walk', 'run', 'sit']
SPRITE_NO_DIRECTION_MOTIONS = ['climb']

VALID_SPRITE_DIRECTORY_NAMES = [f"{motion}_{direction}" for motion in SPRITE_DIRECTION_MOTIONS for direction in SPRITE_DIRECTIONS]

SPRITE_SIZE = (50, 100)

def sprite_name_to_no_direction_name(sprite_name: str) -> str:
    if sprite_name.startswith('portrait'):
        return 'portrait'
    else:
        return sprite_name.split('_')[0]


