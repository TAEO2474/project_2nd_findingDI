#%%writefile app.py
import os, json, sys, hashlib
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from PIL import Image
import streamlit as st
import pandas as pd

# =========================
# 기본 경로/전처리 설정
# =========================
# 배포 환경에서 안전한 기준 경로(파일 위치 기준)
BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts_3cls"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "resnet18_3cls_best.pth"   # 구글드라이브에서 받아옴
MAP_PATH   = ARTIFACTS_DIR / "class_to_idx.json"        # 레포에 있거나(권장), 필요 시 드라이브에서 받기

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif", ".jfif"}
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)
INFER_TF = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def _get_secret_or_env(key: str, default: str = "") -> str:
    # st.secrets 우선, 없으면 환경변수, 둘 다 없으면 default
    try:
        if key in st.secrets:
            return str(st.secrets[key]).strip()
    except Exception:
        pass
    return os.getenv(key, default).strip()

def _sha256sum(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# =========================
# 모델/라벨 준비 (다운로드 + 로드)
# =========================
def ensure_model_download():
    """
    모델(.pth) 파일이 없으면 Google Drive에서 자동 다운로드합니다.
    - FILE_ID는 st.secrets["MODEL_FILE_ID"] 또는 환경변수 MODEL_FILE_ID 로 주입 권_
