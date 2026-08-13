"""Every candidate on the webcam clip. Proxy metrics only — no ground truth."""
import os, sys
import cv2, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,ROOT)
from gmm import GMM_CPU_NUMBA
from gmm.mog2_common import to_planar
from settings import MOG2_N_COMPONENTS
from utils.post_processing import fill_holes
EL=lambda k: cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k))
SIZE=(480,270); K=17   # 0.06 * 270 = 16.2 -> 17

def tin_largest(m,dr=18,er=6):
    d=cv2.dilate(m,EL(dr*2+1)); c,_=cv2.findContours(d,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    o=np.zeros_like(m)
    if c: cv2.fillPoly(o,[max(c,key=cv2.contourArea)],255)
    return cv2.erode(o,EL(er*2+1))

def variants(mask,bp):
    b=np.where(mask==255,np.uint8(255),np.uint8(0)); m=cv2.medianBlur(b,5)
    v={'mog2 +fill (SHIPPING)':fill_holes(m),
       f'mog2 +CLOSE{K}+fill':fill_holes(cv2.morphologyEx(m,cv2.MORPH_CLOSE,EL(K),iterations=1)),
       'Tin largest contour':tin_largest(m)}
    for t in (0.3,0.5,0.7):
        mm=cv2.medianBlur(np.where(bp<t,np.uint8(255),np.uint8(0)),5)
        v[f'bg_prob<{t} +fill']=fill_holes(mm)
        v[f'bg_prob<{t} +CLOSE{K}+fill']=fill_holes(cv2.morphologyEx(mm,cv2.MORPH_CLOSE,EL(K),iterations=1))
    return v

def run(cons):
    cap=cv2.VideoCapture(os.path.join(ROOT,"LTSSUD-Test.mp4"))
    cvt=lambda f: cv2.cvtColor(f,cv2.COLOR_BGR2YCrCb); model=None; st={}
    while True:
        ok,f=cap.read()
        if not ok: break
        f=cv2.resize(f,SIZE)
        if model is None: model=GMM_CPU_NUMBA(cvt(f),n_components=MOG2_N_COMPONENTS,conservative=cons)
        m,_=model.step(to_planar(cvt(f)))
        for name,out in variants(np.asarray(m),np.asarray(model.bg_prob)).items():
            fg=out==255; d=st.setdefault(name,{'c':[],'l':[],'b':[]})
            d['c'].append(fg.mean())
            n,_,s,_=cv2.connectedComponentsWithStats(fg.astype(np.uint8),8)
            d['b'].append(n-1); a=s[1:,cv2.CC_STAT_AREA]
            d['l'].append(a.max()/max(a.sum(),1) if len(a) else 0.0)
    cap.release()
    print(f"\nLTSSUD-Test.mp4 {SIZE[0]}x{SIZE[1]}, conservative={cons}  (subject truly occupies 25-30%)")
    print(f"  {'candidate':30s} {'cover':>6s} {'<5%':>4s} {'>60%':>5s} {'lcc':>5s} {'blobs':>6s}")
    for name,d in st.items():
        c=np.array(d['c'])
        print(f"  {name:30s} {c.mean()*100:5.1f}% {int((c<0.05).sum()):4d} "
              f"{int((c>0.60).sum()):5d} {np.mean(d['l']):5.2f} {np.mean(d['b']):6.1f}")
for cons in (False,True): run(cons)
