from pathlib import Path

def is_video(input_path: Path | str) -> bool:
    if type(input_path) == str:
        return input_path.endswith(('.mp4', '.avi', '.mov', '.mkv'))
    return input_path.name.endswith(('.mp4', '.avi', '.mov', '.mkv'))

def is_image(input_path: Path | str) -> bool:
    if type(input_path) == str:
        return input_path.endswith(('.png', '.jpg', '.jpeg'))
    return input_path.name.endswith(('.png', '.jpg', '.jpeg'))