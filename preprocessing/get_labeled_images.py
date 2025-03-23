"""
Load labeled images from Label Studio project and convert them into COCO format. 
Must have a label studio running.

Run example: in nino base directory,

```
python -m preprocessing.get_labeled_images --project-id 1 --out-dir ./data/sprite_detection_dataset
```
"""

import label_studio_sdk
import yaml
import argparse


def extract_only_labeled_images(server_url: str, project_id: str, out_dir: str):
    """Get only labeled images from given label studio project.

        Args:
                server_url: url of the label studio server.
            project_id: project_id of the label studio project. 
                        out_dir: local directory the labeled images will be saved in
    """

    try:
        with open('./credentials.yml', 'r') as f:
            api_key = yaml.safe_load(f)['api_key']

    except FileNotFoundError:
        raise FileNotFoundError('File not found: credentials.yml')

    ls = label_studio_sdk.Client(url=server_url, api_key=api_key)
    project = ls.get_project('1')
    tasks = project.get_tasks()

    is_labeled_cnt = 0
    is_not_labeled_cnt = 0

    for task in tasks:
        if not task['is_labeled']:
            is_not_labeled_cnt += 1
        else:
            is_labeled_cnt += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--server-url', type=str, required=False,
                        default='http://localhost:8080', help='Label Studio Server URL')
    parser.add_argument('--project-id', type=str, required=False, default='1',
                        help='Project ID of the Label Studio project we want to extract image from')
    parser.add_argument('--out-dir', type=str, required=False,
                        default='./data/images', help='Output directory for the images')

    args = parser.parse_args()

    server_url = args.server_url
    project_id = args.project_id
    out_dir = args.out_dir

    extract_only_labeled_images(
        server_url=server_url, project_id=project_id, out_dir=out_dir)
