# %%writefile app.py
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
# - 일반 실행: __file__ 기준
# - Colab/Jupyter: __file__ 미존재 → os.getcwd() 기준
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path(os.getcwd()).resolve()

ARTIFACTS_DIR = BASE_DIR / "artifacts_3cls"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "resnet18_3cls_best.pth"   # 구글드라이브에서 받아옴
MAP_PATH   = ARTIFACTS_DIR / "class_to_idx.json"        # 레포에 포함(권장) 또는 드라이브에서 받기

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
    - FILE_ID는 st.secrets['MODEL_FILE_ID'] 또는 환경변수 MODEL_FILE_ID 로 주입 권장
    - 드라이브 공유 설정: '링크가 있는 모든 사용자 보기/다운로드'
    - (선택) MODEL_SHA256 으로 무결성 검증 가능
    """
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        return

    MODEL_FILE_ID = _get_secret_or_env("MODEL_FILE_ID", "")
    if not MODEL_FILE_ID:
        st.error(
            "모델(.pth) 파일이 없습니다.\n\n"
            f"- 기대 경로: {MODEL_PATH}\n"
            "- Google Drive FILE_ID를 환경변수 또는 Secrets에 설정하세요.\n"
            "  · 환경변수: MODEL_FILE_ID\n"
            "  · Streamlit secrets: MODEL_FILE_ID"
        )
        st.stop()

    try:
        import gdown
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}&export=download"
    tmp_path = MODEL_PATH.with_suffix(".downloading")

    with st.spinner("모델 다운로드 중... (Google Drive → gdown)"):
        gdown.download(url, str(tmp_path), quiet=False)

    if not tmp_path.exists() or tmp_path.stat().st_size == 0:
        st.error("모델 다운로드 실패: 파일이 비어있거나 존재하지 않습니다. 공유 설정과 FILE_ID를 확인하세요.")
        st.stop()

    # (선택) 무결성 검증
    expected_hash = _get_secret_or_env("MODEL_SHA256", "")
    if expected_hash:
        got = _sha256sum(tmp_path)
        if got.lower() != expected_hash.lower():
            tmp_path.unlink(missing_ok=True)
            st.error("모델 해시 불일치: 업로드 파일/FILE_ID 또는 SHA256 값을 확인하세요.")
            st.stop()

    tmp_path.rename(MODEL_PATH)

def ensure_label_map():
    """
    class_to_idx.json 확보.
    1) 레포/이미지에 포함되어 있으면 그대로 사용 (권장: artifacts_3cls/class_to_idx.json 커밋)
    2) 없으면 Google Drive에서 받아오기 (MAP_FILE_ID 필요)
    """
    if MAP_PATH.exists() and MAP_PATH.stat().st_size > 0:
        return

    MAP_FILE_ID = _get_secret_or_env("MAP_FILE_ID", "")
    if not MAP_FILE_ID:
        st.error(
            "`class_to_idx.json`이 없습니다.\n\n"
            f"- 기대 경로: {MAP_PATH}\n"
            "- 파일을 리포에 포함시키거나, Google Drive FILE_ID를 secrets/env로 설정하세요.\n"
            "  · 환경변수: MAP_FILE_ID (옵션)\n"
            "  · Streamlit secrets: MAP_FILE_ID (옵션)"
        )
        st.stop()

    try:
        import gdown
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
        import gdown

    url = f"https://drive.google.com/uc?id={MAP_FILE_ID}&export=download"
    with st.spinner("라벨 맵 다운로드 중... (gdown)"):
        gdown.download(url, str(MAP_PATH), quiet=False)

    if not MAP_PATH.exists() or MAP_PATH.stat().st_size == 0:
        st.error("라벨 맵 다운로드 실패: 공유 설정과 MAP_FILE_ID를 확인하세요.")
        st.stop()

@st.cache_resource
def load_model_and_labels(arch: str = "resnet18"):
    """모델 가중치와 클래스 매핑을 로드하고, (model, idx_to_class, class_to_idx)를 반환합니다."""
    ensure_model_download()
    ensure_label_map()

    # 라벨 로드
    try:
        with open(MAP_PATH, "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)
    except Exception as e:
        st.error(f"class_to_idx.json 로드 실패: {e}")
        st.stop()

    # idx->class 매핑
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

    # ⚠️ 학습 시 MLP 헤드를 썼다면 동일 구조로 맞춰야 함.
    # 현재는 단일 Linear로 가정 (저장된 state_dict와 일치해야 함)
    backbone.fc = nn.Linear(in_features, num_classes)

    # 가중치 로드
    try:
        state = torch.load(MODEL_PATH, map_location="cpu")

        # 형태 판별: 순수 state_dict or {"state_dict": ...} or whole model
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            backbone.load_state_dict(state["state_dict"], strict=True)
        elif isinstance(state, dict) and all(
            isinstance(k, str) and (
                k.startswith(("conv", "bn", "layer", "fc")) or "num_batches_tracked" in k
            ) for k in state.keys()
        ):
            backbone.load_state_dict(state, strict=True)
        else:
            # 전체 모델 저장본일 가능성
            try:
                backbone = state
                if hasattr(backbone, "eval"):
                    backbone.eval()
                else:
                    raise RuntimeError("불지원 형식: 전체 모델 객체가 아님")
            except Exception as e:
                raise RuntimeError(f"지원하지 않는 모델 저장 형식입니다: {e}")

    except Exception as e:
        st.error(f"모델 가중치 로드 실패: {e}")
        st.stop()

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

# ---- 디버그 패널 (필요시 사용) ----
with st.sidebar.expander("🔍 Debug (secrets/env)", expanded=False):
    try:
        has_secret = "MODEL_FILE_ID" in st.secrets
    except Exception:
        has_secret = False
    st.write("has st.secrets['MODEL_FILE_ID']:", has_secret)
    st.write("env MODEL_FILE_ID set:", bool(os.getenv("MODEL_FILE_ID")))
    st.write("MODEL_PATH exists:", MODEL_PATH.exists())
    if MODEL_PATH.exists():
        st.write("MODEL_PATH size:", MODEL_PATH.stat().st_size)

# ---- (옵션) 임시 입력: Secrets 없이 테스트할 때 사용 후 삭제 권장 ----
if _get_secret_or_env("MODEL_FILE_ID", "") == "":
    with st.sidebar.expander("⚠️ Set MODEL_FILE_ID (temp)", expanded=False):
        tmp_id = st.text_input("Google Drive FILE_ID")
        if tmp_id:
            os.environ["MODEL_FILE_ID"] = tmp_id
            st.success("MODEL_FILE_ID set for this session. Rerun the app (⌘/Ctrl+R).")

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
