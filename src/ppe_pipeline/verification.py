"""Optional explicit hardhat/no-hardhat evidence on predicted-person head crops.

No detections, conflicting classes, very small heads and out-of-region boxes abstain.
This does not validate a camera, classify vests, or prove a crop is a person.
"""
from pathlib import Path
import numpy as np
from .association import intersection, area


def head_crop(image, box, min_width=24):
    x1,y1,x2,y2 = map(float, box)
    w,h=x2-x1,y2-y1
    if w < min_width or h * .32 < 20:
        return None
    height,width=image.shape[:2]
    left,top=max(0,int(x1-.25*w)),max(0,int(y1-.12*h))
    right,bottom=min(width,int(x2+.25*w)),min(height,int(y1+.45*h))
    crop=image[top:bottom,left:right]
    if crop.size == 0:
        return None
    return crop,(left,top),(x1,y1,x2,y1+.35*h)


class HeadVerifier:
    def __init__(self, path: Path, conf=.40, margin=.15):
        from ultralytics import YOLO
        if not Path(path).is_file():
            raise FileNotFoundError(path)
        self.model=YOLO(str(path))
        names={int(k):v.lower() for k,v in self.model.names.items()}
        if set(names.values()) != {"hardhat","no-hardhat"}:
            raise ValueError(f"expected Hardhat/NO-Hardhat classes, got {names}")
        self.classes={k:"on" if v=="hardhat" else "off" for k,v in names.items()}
        self.conf,self.margin=conf,margin

    def __call__(self,image,box):
        prepared=head_crop(image,box)
        if prepared is None:
            return {"state":"unknown","confidence":None,"reason":"head too small"}
        crop,(left,top),region=prepared
        result=self.model.predict(crop,imgsz=320,conf=self.conf,iou=.45,device="cpu",verbose=False)[0]
        scores={"on":0.0,"off":0.0}
        for row in result.boxes.data.cpu().numpy():
            ppe=(row[0]+left,row[1]+top,row[2]+left,row[3]+top)
            if area(ppe)<=0 or intersection(ppe,region)/area(ppe)<.6:
                continue
            cx,cy=(ppe[0]+ppe[2])/2,(ppe[1]+ppe[3])/2
            if not(region[0]<=cx<=region[2] and region[1]<=cy<=region[3]):
                continue
            kind=self.classes[int(row[5])]
            scores[kind]=max(scores[kind],float(row[4]))
        best=max(scores,key=scores.get)
        if scores[best]<self.conf or abs(scores["on"]-scores["off"])<self.margin:
            return {"state":"unknown","confidence":None,"reason":"missing or conflicting head evidence"}
        return {"state":best,"confidence":scores[best],"reason":"explicit head-crop detection"}
