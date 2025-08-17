import os, json
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
BASE_DIR = Path(os.getcwd()) # Use current working directory instead
MODEL_PATH = BASE_DIR / "artifacts_3cls" / "resnet18_3cls_best.pth"     # 모델 파일은 런타임에 gdown으로 받음
MAP_PATH   = BASE_DIR / "artifacts_3cls" / "class_to_idx.json"          # 레포에 있으면 바로 사용

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff", ".gif", ".jfif"}
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)
INFER_TF = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# =========================
# 모델/라벨 준비 (다운로드 + 로드)
# =========================
def ensure_model_download():
    """
    Google Drive 공유 링크의 FILE_ID를 넣어두면, 모델(.pth)이 없을 때 자동으로 다운로드합니다.
    - 링크 공유 권한: '링크가 있는 모든 사용자 보기/다운로드' 필수
    """
    if MODEL_PATH.exists():
        return

    # TODO: 아래 FILE_ID를 실제 값으로 교체하세요.
    MODEL_FILE_ID = "FILE_ID"  # 예: 1AbCdEfGhIJ... (drive.google.com/file/d/여기부분/view)
    if MODEL_FILE_ID == "FILE_ID":
        st.error(
            "모델(.pth) 파일이 없습니다.\n\n"
            f"- 기대 경로: {MODEL_PATH}\n"
            "- Google Drive FILE_ID를 app.py 상단 ensure_model_download() 안에 설정해 주세요."
        )
        st.stop()

    try:
        import gdown
    except ImportError:
        st.error("gdown이 설치되어 있지 않습니다. requirements.txt에 gdown을 포함했는지 확인하세요.")
        st.stop()

    url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}&export=download"
    with st.spinner("모델 다운로드 중... (gdown)"):
        gdown.download(url, str(MODEL_PATH), quiet=False)

    if not MODEL_PATH.exists():
        st.error("모델 다운로드에 실패했습니다. 공유 설정과 FILE_ID를 확인하세요.")
        st.stop()

def ensure_label_map():
    """
    class_to_idx.json이 레포에 없으면(선택) Google Drive에서 받아옵니다.
    필요 없다면 아래 MAP_FILE_ID 부분을 'FILE_ID_JSON' 그대로 두고,
    레포에 class_to_idx.json 파일을 올려두세요.
    """
    if MAP_PATH.exists():
        return

    # (선택) JSON도 드라이브에서 받기 원하면 FILE_ID_JSON 설정하세요.
    MAP_FILE_ID = "FILE_ID_JSON"  # 예: 1XyZ...
    if MAP_FILE_ID == "FILE_ID_JSON":
        st.error(
            "`class_to_idx.json`이 없습니다.\n\n"
            f"- 기대 경로: {MAP_PATH}\n"
            "- 레포에 파일을 추가하거나, ensure_label_map()의 MAP_FILE_ID를 실제 값으로 설정하세요."
        )
        st.stop()

    try:
        import gdown
    except ImportError:
        st.error("gdown이 설치되어 있지 않습니다. requirements.txt에 gdown을 포함했는지 확인하세요.")
        st.stop()

    url = f"https://drive.google.com/uc?id={MAP_FILE_ID}&export=download"
    with st.spinner("라벨 맵 다운로드 중... (gdown)"):
        gdown.download(url, str(MAP_PATH), quiet=False)

    if not MAP_PATH.exists():
        st.error("라벨 맵 다운로드에 실패했습니다. 공유 설정과 FILE_ID_JSON을 확인하세요.")
        st.stop()


@st.cache_resource
def load_model_and_labels(arch: str = "resnet18"):
    """모델 가중치와 클래스 매핑을 로드하고, (model, idx_to_class, class_to_idx)를 반환합니다."""
    ensure_model_download()
    ensure_label_map()

    # 라벨 로드
    with open(MAP_PATH, "r", encoding="utf-8") as f:
        class_to_idx = json.load(f)
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)

    # 백본 구성 (학습 때와 동일해야 합니다)
    if arch == "resnet18":
        backbone = models.resnet18(weights=None)
    elif arch == "resnet50":
        backbone = models.resnet50(weights=None)
    else:
        raise ValueError("Unsupported arch (resnet18|resnet50)")

    in_features = backbone.fc.in_features

    # ⚠️ 학습 시 MLP 헤드를 썼다면 아래를 동일하게 맞춰야 합니다.
    # 예) Linear → ReLU → Dropout → Linear 였다면 같은 구조로 교체 필요.
    # 현재는 단일 Linear로 가정 (resnet18_3cls_best.pth 저장 시점과 일치)
    backbone.fc = nn.Linear(in_features, num_classes)

    # 가중치 로드
    state = torch.load(MODEL_PATH, map_location="cpu")
    backbone.load_state_dict(state, strict=True)
    backbone.eval()
    return backbone, idx_to_class, class_to_idx

def predict_one(img: Image.Image, model: torch.nn.Module, device, idx_to_class: dict):
    x = INFER_TF(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        prob = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
    items = [(idx_to_class[i], float(prob[i])) for i in range(len(prob))]
    items.sort(key=lambda t: t[1], reverse=True)
    return items, prob

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Knife/Awl/Scissor Classifier", page_icon="🔎", layout="centered")
st.title("🔎 3-Class Classifier (ResNet)")
st.caption("knife / awl / scissor — 확률 예측 데모")

with st.spinner("모델 로딩 중..."):
    model, idx_to_class, class_to_idx = load_model_and_labels(arch="resnet18")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

tab1, tab2 = st.tabs(["📄 단일 이미지", "📁 여러 이미지(배치)"])

with tab1:
    up = st.file_uploader("이미지 업로드", type=[ext.lstrip(".") for ext in IMG_EXTS])
    conf_th = st.slider("표시 임계값(threshold)", 0.0, 1.0, 0.0, 0.01)
    if up is not None:
        try:
            img = Image.open(up).convert("RGB")
            st.image(img, caption="입력 이미지", use_column_width=True)
            items, prob = predict_one(img, model, device, idx_to_class)

            df = pd.DataFrame([{"class": c, "prob": p} for c, p in items])
            st.subheader("확률")
            st.bar_chart(df.set_index("class"))

            top1 = items[0]
            st.markdown(f"**Top-1:** `{top1[0]}` — **{top1[1]*100:.2f}%**")
            label = top1[0] if top1[1] >= conf_th else "unknown"
            st.markdown(f"**최종 라벨(임계값 {conf_th:.2f}):** `{label}`")
        except Exception as e:
            st.error(f"이미지 처리 중 오류: {e}")

with tab2:
    ups = st.file_uploader("여러 이미지 업로드", type=[ext.lstrip(".") for ext in IMG_EXTS], accept_multiple_files=True)
    if ups:
        rows = []
        for f in ups:
            try:
                img = Image.open(f).convert("RGB")
                items, prob = predict_one(img, model, device, idx_to_class)
                top1 = items[0]
                row = {"filename": f.name, "pred": top1[0], "conf": top1[1]}
                # 고정 순서: class_to_idx의 인덱스 순
                for cls in sorted(class_to_idx, key=lambda k: class_to_idx[k]):
                    idx = class_to_idx[cls]
                    row[f"p_{cls}"] = float(prob[idx])
                rows.append(row)
            except Exception as e:
                rows.append({"filename": f.name, "pred": "ERROR", "conf": 0.0, "error": str(e)})

        if rows:
            out_df = pd.DataFrame(rows)
            st.dataframe(out_df, use_container_width=True)
            st.download_button(
                "CSV 다운로드",
                data=out_df.to_csv(index=False).encode("utf-8"),
                file_name="infer_results.csv",
                mime="text/csv",
            )
