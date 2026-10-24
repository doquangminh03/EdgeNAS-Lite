"""Hash local YOLO-format KITTI images/labels and reject train/val overlap."""
import argparse
import hashlib
import json
from pathlib import Path


def build_manifest(root):
    root = Path(root).resolve()
    splits = {}
    for split in ('train', 'val'):
        rows = []
        for image in sorted((root / 'images' / split).rglob('*')):
            if image.name.startswith('._') or image.suffix.lower() not in {'.jpg','.jpeg','.png','.bmp'} or not image.is_file():
                continue
            label = root / 'labels' / split / image.relative_to(root / 'images' / split).with_suffix('.txt')
            if not label.is_file():
                raise ValueError('Missing label: ' + str(label))
            for line in label.read_text().splitlines():
                values = line.split()
                if not values:
                    continue
                import math
                if len(values) != 5 or not values[0].isdigit() or not 0 <= int(values[0]) <= 7:
                    raise ValueError('Invalid KITTI YOLO label: ' + str(label))
                coordinates = [float(v) for v in values[1:]]
                if any(not math.isfinite(v) or not 0 <= v <= 1 for v in coordinates):
                    raise ValueError('Invalid normalized box: ' + str(label))
            rows.append({'image':str(image.relative_to(root)), 'image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
                         'label':str(label.relative_to(root)), 'label_sha256':hashlib.sha256(label.read_bytes()).hexdigest()})
        if not rows:
            raise ValueError('Empty image split: ' + split)
        splits[split] = rows
    overlap = {row['image_sha256'] for row in splits['train']} & {row['image_sha256'] for row in splits['val']}
    overlap_ids = {Path(row['image']).stem for row in splits['train']} & {Path(row['image']).stem for row in splits['val']}
    if overlap or overlap_ids:
        raise ValueError('Train/val overlap by image hash or image ID.')
    serialized = json.dumps(splits, sort_keys=True,separators=(',',':')).encode()
    return {'schema_version':'kitti_yolo_manifest_v1', 'splits':splits, 'manifest_sha256':hashlib.sha256(serialized).hexdigest(),
            'selected_class_ids':[0,3,5], 'class_mapping':{'0':'car','3':'pedestrian','5':'cyclist'},
            'historical_split_reproduced':False,
            'limitations':['Hashes attest local files only; original conversion and split generation were not independently recovered.']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-root',default='datasets/kitti')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists(): parser.error('Output exists.')
    try: result=build_manifest(args.dataset_root)
    except (ValueError,OSError) as error: parser.error(str(error))
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')


if __name__=='__main__': main()
