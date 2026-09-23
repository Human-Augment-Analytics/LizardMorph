"""Small, reproducible landmark adaptation experiment with a frozen Anolis detector.

Uses 16 unique TPS-annotated scans for fitting, six separate scans for validation,
and the original demo scans (including 1910) as an additional diagnostic set.
No scan is selected by its resulting error. Existing weights are never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import xml.etree.ElementTree as ET

import cv2
import dlib
import numpy as np
from PIL import Image

# Local archival scans are routinely 200 megapixels; decode via JPEG draft.
Image.MAX_IMAGE_PIXELS = 300_000_000

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from ort_inference import OrtYoloDetector
from utils import _predict_toepad_crop


def rectify(image, corners):
    corners = np.asarray(corners, np.float32)
    by_y = corners[np.argsort(corners[:, 1])]
    top = by_y[:2][np.argsort(by_y[:2, 0])]
    bottom = by_y[2:][np.argsort(by_y[2:, 0])]
    box = np.array([top[0], top[1], bottom[1], bottom[0]], np.float32)
    w = int(round(np.linalg.norm(box[1] - box[0])))
    h = int(round(np.linalg.norm(box[2] - box[1])))
    if min(w, h) < 2:
        raise ValueError('Degenerate crop')
    dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)
    matrix = cv2.getPerspectiveTransform(box, dst)
    crop = cv2.warpPerspective(image, matrix, (w, h))
    scale = min(512 / h, 512 / w)
    nw, nh = int(w * scale), int(h * scale)
    px, py = (512-nw)//2, (512-nh)//2
    canvas = np.zeros((512, 512, 3), np.uint8)
    canvas[py:py+nh, px:px+nw] = cv2.resize(crop, (nw, nh))
    return canvas, matrix, scale, np.array([px, py])


def tps_points(path, width, height, new_width, new_height):
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip() and '=' not in line]
    if len(lines) != 11:
        raise ValueError(f'{path}: expected ruler + 9 landmarks, got {len(lines)}')
    points = np.array([list(map(float, line.split()[:2])) for line in lines[2:]])
    points[:, 1] = height - 1 - points[:, 1]
    points *= [new_width / width, new_height / height]
    if not np.isfinite(points).all() or (points < 0).any() or (points > [new_width, new_height]).any():
        raise ValueError(f'{path}: invalid landmarks')
    return points


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT/'artifacts/anolis-small-data')
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    scans = args.source/'data/miami_fall_24_jpgs'; tps = args.source/'data/tps_files'
    excluded = {'1004','1841','1866','1866b','1910'}
    stems = sorted(p.stem for p in scans.glob('*.jpg') if p.stem.isdigit() and p.stem not in excluded
                   and (tps/f'{p.stem}_toe.TPS').exists() and (tps/f'{p.stem}_finger.TPS').exists())
    random.Random(1910).shuffle(stems)
    train_ids, val_ids = stems[:16], stems[16:22]
    assert len(train_ids) == 16 and len(val_ids) == 6
    split = {'seed':1910,'train_ids':train_ids,'validation_ids':val_ids,
             'diagnostic_ids':sorted(excluded),'width':1600,
             'note':'Validation excludes new fitting data; historical pretrained-model overlap is unknown.'}
    (out/'split.json').write_text(json.dumps(split, indent=2))
    detector = OrtYoloDetector(str(ROOT/'models/lizard-toe-pad/yolo_obb_6class_h7.onnx'))
    records=[]
    for stem in train_ids + val_ids:
        cache=out/f'{stem}.npz'
        if cache.exists():
            pack=np.load(cache); image=pack['image']
            gt={c:pack['gt_'+c] for c in ['bot_toe','bot_finger']}
            detections={c:[{'corners':pack['corners_'+c]}] for c in gt if 'corners_'+c in pack}
        else:
            with Image.open(scans/f'{stem}.jpg') as source:
                width,height=source.size; nw=1600;nh=round(height*nw/width)
                source.draft('RGB',(nw,nh));image=cv2.cvtColor(np.asarray(source.convert('RGB').resize((nw,nh))),cv2.COLOR_RGB2BGR)
            gt={f'bot_{kind}':tps_points(tps/f'{stem}_{kind}.TPS',width,height,nw,nh) for kind in ['toe','finger']}
            detections=detector.detect(image)
            payload={'image':image,**{'gt_'+c:p for c,p in gt.items()}}
            payload.update({'corners_'+c:detections[c][0]['corners'] for c in gt if detections.get(c)})
            np.savez_compressed(cache,**payload)
        records.append({'id':stem,'image':image,'gt':gt,'detections':detections,'split':'train' if stem in train_ids else 'validation'})
        print('Prepared',stem,records[-1]['split'],flush=True)
    # Existing five-file demo has only four unique image contents.
    demo=ROOT/'sample_data/real_lizard_toepad_dataset'
    seen=set()
    for node in ET.parse(demo/'annotations.xml').findall('.//image'):
        path=demo/'images'/node.get('file');digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in seen:continue
        seen.add(digest)
        image=cv2.imread(str(path));gt={b.findtext('label'):np.array([[float(p.get('x')),float(p.get('y'))] for p in b.findall('part')]) for b in node.findall('box')}
        records.append({'id':path.stem,'image':image,'gt':gt,'detections':detector.detect(image),'split':'diagnostic'})
    models={}
    for kind in ['toe','finger']:
        model_path=out/f'shape_predictor_{kind}_rectify512.dat'; xml=out/f'train_{kind}.xml'
        root=ET.Element('dataset'); images=ET.SubElement(root,'images')
        for record in records:
            if record['split']!='train':continue
            c='bot_'+kind; boxes=record['detections'].get(c,[])
            if not boxes:raise ValueError(f'Training detection missing: {record["id"]} {c}')
            canvas,matrix,scale,padding=rectify(record['image'],boxes[0]['corners'])
            points=cv2.perspectiveTransform(record['gt'][c].astype(np.float32)[None],matrix)[0]*scale+padding
            if (points<0).any() or (points>511).any():raise ValueError(f'Landmarks outside crop: {record["id"]} {c}')
            crop_path=out/f'{record["id"]}_{kind}.png'
            # dlib loads files as RGB. Reverse before writing so training sees the
            # same BGR channel order supplied directly at inference.
            cv2.imwrite(str(crop_path),cv2.cvtColor(canvas,cv2.COLOR_BGR2RGB))
            elem=ET.SubElement(images,'image',file=str(crop_path));box=ET.SubElement(elem,'box',top='0',left='0',width='512',height='512')
            for k,(x,y) in enumerate(points):ET.SubElement(box,'part',name=str(k),x=str(round(float(x))),y=str(round(float(y))))
        ET.ElementTree(root).write(xml)
        if not model_path.exists():
            opts=dlib.shape_predictor_training_options();opts.num_threads=4;opts.tree_depth=4;opts.cascade_depth=15;opts.oversampling_amount=10;opts.feature_pool_size=400;opts.num_test_splits=20;opts.nu=.1;opts.random_seed='1910'
            print('Training',kind,'on',len(images),'unique scans',flush=True)
            dlib.train_shape_predictor(str(xml),str(model_path),opts)
        models[kind]=dlib.shape_predictor(str(model_path))
    pretrained=dlib.shape_predictor(str(ROOT/'models/lizard-toe-pad/ml_morph_best.dat'))
    rows=[]
    for record in records:
        if record['split']=='train':continue
        for c,gt in record['gt'].items():
            boxes=record['detections'].get(c,[])
            for candidate in ['pretrained','small_data']:
                row={'image':record['id'],'split':record['split'],'class':c,'candidate':candidate,'detected':bool(boxes)}
                if boxes:
                    source=cv2.flip(record['image'],0) if c.startswith('up_') else record['image']
                    predictor=pretrained if candidate=='pretrained' else models[c.split('_')[1]]
                    pred=_predict_toepad_crop(predictor,source,boxes[0]['corners'],'ml_morph_best.dat')
                    if c.startswith('up_'):pred[:,1]=record['image'].shape[0]-1-pred[:,1]
                    errors=np.linalg.norm(pred-gt,axis=1)
                    row.update(mean_error_px=float(errors.mean()),max_error_px=float(errors.max()),predicted=pred.tolist(),actual=gt.tolist())
                rows.append(row)
    summary={}
    for split_name in ['validation','diagnostic']:
        summary[split_name]={}
        for candidate in ['pretrained','small_data']:
            selected=[r for r in rows if r['split']==split_name and r['candidate']==candidate]
            summary[split_name][candidate]={'objects':len(selected),'detected':sum(r['detected'] for r in selected),
              'mean_error_px':float(np.mean([r['mean_error_px'] for r in selected if r['detected']])),
              'worst_object_mean_px':max(r['mean_error_px'] for r in selected if r['detected'])}
    (out/'evaluation.json').write_text(json.dumps({'split':split,'summary':summary,'rows':rows},indent=2))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
