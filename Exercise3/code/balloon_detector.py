import os
import glob
import pickle
import numpy as np
import joblib
from PIL import Image
import torch
from torchvision import models, transforms
from selective_search import selective_search
import skimage.io as skio
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_ROOT = '../data/balloon_dataset'
SPLITS = ['train', 'valid']
TRAIN_FEATS = 'train_data.npy'
TRAIN_LABELS = 'train_labels.npy'
VALID_FEATS = 'valid_data.npy'
VALID_LABELS = 'valid_labels.npy'
MODEL_OUTPUT = 'balloon_model.pkl'
TEST_IMAGE = '../data/balloon_dataset/test/7.jpg'
DETECTION_OUTPUT = 'result53.png'

# Thresholds and hyperparameters
_POS_IOU = 0.75
_NEG_IOU = 0.25
_HARD_COUNT = 500
_CONF_THRESH = 0.5
_NMS_IOU = 0.30
_SS_PARAMS = dict(scale=150, sigma=0.65, min_size=12)
_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# Utility: compute intersection-over-union for two boxes
def _iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[0] + a[2], b[0] + b[2])
    y2 = min(a[1] + a[3], b[1] + b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    box_a_area = a[2] * a[3]
    box_b_area = b[2] * b[3]
    union = box_a_area + box_b_area - inter
    return inter / union if union > 0 else 0


# 1) Extract region proposals
def extract_regions(root, splits):
    all_props = {}
    for split in splits:
        path = os.path.join(root, split)
        props = {}
        for img in sorted(glob.glob(os.path.join(path, '*.jpg'))):
            arr = skio.imread(img)
            _, regs = selective_search(arr, **_SS_PARAMS)
            props[os.path.basename(img)] = [r['rect'] for r in regs]
        all_props[split] = props
    return all_props


# 2) Pull CNN features via ResNet-50
_backbone = models.resnet50(pretrained=True).to(_DEVICE).eval()
_feat_net = torch.nn.Sequential(*list(_backbone.children())[:-1]).to(_DEVICE)
_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def extract_feat(img_path, box):
    x, y, w, h = box
    img = Image.open(img_path).convert('RGB')
    crop = img.crop((x, y, x + w, y + h))
    inp = _transform(crop).unsqueeze(0).to(_DEVICE)
    with torch.no_grad():
        feats = _feat_net(inp)
    return feats.squeeze().cpu().numpy()


# 3) Assemble dataset
def prepare_dataset():
    props = extract_regions(DATA_ROOT, SPLITS)
    for split in SPLITS:
        # load annotations
        annf = os.path.join(DATA_ROOT, split, '_annotations.coco.json')
        meta = pickle.load(open(annf.replace('.json', '.pkl'),
                           'rb')) if False else _load_json(annf)
        id2file = {i['id']: i['file_name'] for i in meta['images']}
        gt = {}
        for a in meta['annotations']:
            nm = id2file[a['image_id']]
            gt.setdefault(nm, []).append(tuple(a['bbox']))
        pos, neg = [], []
        for img, boxes in props[split].items():
            truths = gt.get(img, [])
            for b in boxes:
                score = max((_iou(b, t) for t in truths), default=0)
                if score >= _POS_IOU:
                    pos.append((img, b))
                elif score <= _NEG_IOU:
                    neg.append((img, b))
        total = len(pos) + len(neg)
        X = np.zeros((total, 2048))
        y = np.zeros((total,))
        idx = 0
        for lbl, coll in [(1, pos), (0, neg)]:
            for nm, b in coll:
                X[idx] = extract_feat(os.path.join(DATA_ROOT, split, nm), b)
                y[idx] = lbl
                idx += 1
        np.save(f'{split}_data.npy', X)
        np.save(f'{split}_labels.npy', y)
        pickle.dump({'pos': pos, 'neg': neg}, open(f'{split}_items.pkl', 'wb'))


# helper: load json annotation and returns as a python dictionary
def _load_json(fp): import json; return json.load(open(fp))


# 4) Train with hard-negative mining
def train_model():
    X_train = np.load(TRAIN_FEATS)
    y_train = np.load(TRAIN_LABELS)
    X_val = np.load(VALID_FEATS)
    y_val = np.load(VALID_LABELS)
    pos_i = np.where(y_train == 1)[0]
    neg_i = np.where(y_train == 0)[0]
    sel = np.hstack([pos_i, np.random.choice(
        neg_i, len(pos_i) * 5, replace=False)])
    pipe = make_pipeline(StandardScaler(),
                         SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42))
    pipe.fit(X_train[sel], y_train[sel])
    # mining
    scores = pipe.decision_function(X_train)
    fps = np.where((y_train == 0) & (scores > 0))[0]
    if len(fps):
        top = fps[np.argsort(-scores[fps])[:_HARD_COUNT]]
        pipe.fit(np.vstack([X_train[sel], X_train[top]]),
                 np.hstack([y_train[sel], np.zeros(len(top))]))
    preds = pipe.predict(X_val)
    print('Val Acc:', accuracy_score(y_val, preds))
    print(classification_report(y_val, preds, digits=4))
    joblib.dump(pipe, MODEL_OUTPUT)


# 5) Detect with NMS
def nonmax(rects, scores):
    ids = np.argsort(scores)[::-1]
    keep = []
    while len(ids):
        i = ids[0]
        keep.append(i)
        rem = [j for j in ids[1:] if _iou(rects[i], rects[j]) <= _NMS_IOU]
        ids = np.array(rem)
    return keep


def detect_image():
    clf = joblib.load(MODEL_OUTPUT)
    img = Image.open(TEST_IMAGE).convert('RGB')
    arr = np.array(img)
    _, regs = selective_search(arr, **_SS_PARAMS)
    cand = [r['rect'] for r in regs]
    feats = np.vstack([extract_feat(TEST_IMAGE, b) for b in cand])
    prob = clf.predict_proba(feats)[:, 1]
    idxs = np.where(prob >= _CONF_THRESH)[0]
    sel = [cand[i] for i in idxs]
    scr = prob[idxs]
    keep = nonmax(sel, scr)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(arr)
    ax.axis('off')
    for (x, y, w, h), s in zip([sel[i] for i in keep], [scr[i] for i in keep]):
        r = mpatches.Rectangle(
            (x, y), w, h, edgecolor='lime', facecolor='none', linewidth=2)
        ax.add_patch(r)
        ax.text(x, y - 4, f"{s:.2f}", fontsize=11,
                color='white', backgroundcolor='black')
    plt.tight_layout()
    plt.savefig(DETECTION_OUTPUT, dpi=150)
    print(f"Detection saved to {DETECTION_OUTPUT}")


# Entry point
if __name__ == '__main__':

    # prepare_dataset()
    # train_model()
    detect_image()
