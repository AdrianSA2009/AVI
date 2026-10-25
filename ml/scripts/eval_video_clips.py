"""Evaluasi model temporal per klip, termasuk J/Z.

Skor pada video yang sama dengan training adalah resubstitution dan bukan
validasi umum. Untuk klaim generalisasi, kumpulkan video signer/clip berbeda
serta split per klip sebelum training.

Pemakaian:
    python scripts\\eval_video_clips.py --checkpoint runs\\video_sibi_jz_temporal\\best.pt --source data\\video_train_sibi
"""
from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import torch
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import train_video as tv

def load_clip(path:Path):
    cap=cv2.VideoCapture(str(path)); frames=[]
    if not cap.isOpened(): raise RuntimeError(f'Video tidak terbuka: {path}')
    while True:
        ok,frame=cap.read()
        if not ok:break
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(frame,(tv.SIZE,tv.SIZE),interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:return None
    positions=np.linspace(0,len(frames)-1,tv.FRAMES,dtype=int)
    arr=np.stack([frames[int(i)] for i in positions])
    return torch.from_numpy(arr).permute(0,3,1,2).contiguous().float().div_(255).unsqueeze(0)

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--checkpoint',type=Path,default=Path('runs/video_sibi_jz_temporal/best.pt'))
    ap.add_argument('--source',type=Path,default=Path('data/video_train_sibi'))
    ap.add_argument('--out',type=Path,default=Path('outputs/video_clip_eval.json'))
    args=ap.parse_args(argv)
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    labels=ck['classes']; model=tv.ClipClassifier(len(labels),temporal=bool(ck.get('temporal',False)))
    model.load_state_dict(ck['state_dict']); model.eval()
    rows=[]; correct=0
    with torch.inference_mode():
        for cat in sorted(p for p in args.source.iterdir() if p.is_dir()):
            for cls in sorted(p for p in cat.iterdir() if p.is_dir()):
                truth=f'{cat.name}:{cls.name.lower()}'
                for path in sorted(p for p in cls.rglob('*') if p.is_file() and p.suffix.lower() in {'.mp4','.mov','.avi','.mkv','.webm'}):
                    clip=load_clip(path)
                    if clip is None:continue
                    pred=labels[int(model(clip).argmax(1).item())]
                    ok=pred==truth;correct+=ok
                    rows.append({'video':path.name,'truth':truth,'pred':pred,'correct':ok})
                    print(f"{path.name}: gt={truth} pred={pred} {'OK' if ok else 'WRONG'}")
    result={'videos':len(rows),'correct':correct,'resubstitution_accuracy':correct/len(rows) if rows else 0.0,'rows':rows,
            'warning':'In-sample clips only; resubstitution accuracy is not independent validation/generalization.'}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print('in-sample:',correct,'/',len(rows),'accuracy',result['resubstitution_accuracy'])
    return 0
if __name__=='__main__':raise SystemExit(main())
