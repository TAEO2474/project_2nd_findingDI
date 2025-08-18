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
# 경로/전처리
# =========================
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path(os.getcwd()).resolve()

ARTIFACTS_DIR = BASE_DIR / "artifacts_3cls"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "resnet18_3cls_best.pth"
MAP_PATH   = ARTIFACTS_DIR / "class_to_idx.json"

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
# 내장 라벨맵(파일 없을 때 자동 생성용)
# 학습 시 사용한 매핑과 동일해야 합니다!
# =========================
EMBEDDED_CLASS_TO_IDX = {
    "awl": 0,
    "knife": 1,
    "scissor": 2,
}

# =========================
# 유틸
# =========================
def _get_secret_or_env(key: str, default: str = "") -> str:
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
# 다운로드/확보
# =========================
def ensure_model_download():
    """모델(.pth)이 없으면 Google Drive에서 gdown으로 다운로드 (Secrets/ENV: MODEL_FILE_ID)."""
    try:
        with st.sidebar.expander("🐞 DEBUG (env/paths)", expanded=False):
            try:
                has_secret = "MODEL_FILE_ID" in st.secrets
            except Exception:
                has_secret = False
            st.write({
                "cwd": os.getcwd(),
                "python": sys.version,
                "BASE_DIR": str(BASE_DIR),
                "MODEL_PATH": str(MODEL_PATH),
                "MAP_PATH": str(MAP_PATH),
                "has_secret_MODEL_FILE_ID": has_secret,
                "env_MODEL_FILE_ID": bool(os.getenv("MODEL_FILE_ID")),
            })

        if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
            st.sidebar.success(f"모델 존재: {MODEL_PATH.name} ({MODEL_PATH.stat().st_size} bytes)")
            return

        MODEL_FILE_ID = _get_secret_or_env("MODEL_FILE_ID", "")
        if not MODEL_FILE_ID:
            raise RuntimeError("MODEL_FILE_ID가 비어 있습니다. (Settings→Secrets 또는 환경변수로 설정)")

        try:
            import gdown
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown

        url = f"https://drive.google.com/uc?id={MODEL_FILE_ID}&export=download"
        tmp_path = MODEL_PATH.with_suffix(".downloading")
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

        with st.spinner("모델 다운로드 중... (gdown)"):
            st.sidebar.write("🐞 gdown url =", url)
            gdown.download(url, str(tmp_path), quiet=False)

        if not tmp_path.exists() or tmp_path.stat().st_size == 0:
            raise RuntimeError("모델 다운로드 실패: 파일이 비었거나 생성되지 않음 (공유권한/FILE_ID 확인)")

        expected_hash = _get_secret_or_env("MODEL_SHA256", "")
        if expected_hash:
            got = _sha256sum(tmp_path)
            if got.lower() != expected_hash.lower():
                tmp_path.unlink(missing_ok=True)
                raise RuntimeError(f"모델 해시 불일치: got={got} expected={expected_hash}")

        tmp_path.rename(MODEL_PATH)
        st.sidebar.success(f"모델 저장 완료: {MODEL_PATH} ({MODEL_PATH.stat().st_size} bytes)")

    except Exception as e:
        st.error(f"ensure_model_download() 실패: {type(e).__name__}: {e}")
        st.info("체크: Secrets의 MODEL_FILE_ID, 드라이브 공유(링크 있는 모든 사용자), requirements.txt의 gdown")
        st.stop()

def ensure_label_map():
    """
    class_to_idx.json 확보 우선순위:
    1) 파일이 이미 있으면 사용
    2) MAP_FILE_ID가 있으면 드라이브에서 다운로드
    3) 둘 다 아니면 EMBEDDED_CLASS_TO_IDX로 파일 생성
    """
    if MAP_PATH.exists() and MAP_PATH.stat().st_size > 0:
        return

    MAP_FILE_ID = _get_secret_or_env("MAP_FILE_ID", "")
    if MAP_FILE_ID:
        try:
            import gdown
        except ImportError:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
            import gdown

        url = f"https://drive.google.com/uc?id={MAP_FILE_ID}&export=download"
        with st.spinner("라벨 맵 다운로드 중... (gdown)"):
            gdown.download(url, str(MAP_PATH), quiet=False)

        if MAP_PATH.exists() and MAP_PATH.stat().st_size > 0:
            return
        else:
            st.warning("MAP_FILE_ID 다운로드 실패. 내장 라벨맵으로 생성합니다.")

    # 내장 라벨맵으로 파일 생성
    try:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(MAP_PATH, "w", encoding="utf-8") as f:
            json.dump(EMBEDDED_CLASS_TO_IDX, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"class_to_idx.json 생성 실패: {e}")
        st.stop()

# =========================
# 로드
# =========================
@st.cache_resource
def load_model_and_labels(arch: str = "resnet18"):
    """(model, idx_to_class, class_to_idx) 반환."""
    ensure_model_download()
    ensure_label_map()

    # 라벨 로드
    try:
        with open(MAP_PATH, "r", encoding="utf-8") as f:
            class_to_idx = json.load(f)
    except Exception as e:
        st.error(f"class_to_idx.json 로드 실패: {e}")
        st.stop()

    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(idx_to_class)

    # 백본 구성 (학습과 동일)
    if arch == "resnet18":
        backbone = models.resnet18(weights=None)
    elif arch == "resnet50":
        backbone = models.resnet50(weights=None)
    else:
        raise ValueError("Unsupported arch (resnet18|resnet50)")

    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)  # 헤드 구조가 학습과 동일해야 함

    # 가중치 로드
    try:
        state = torch.load(MODEL_PATH, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            backbone.load_state_dict(state["state_dict"], strict=True)
        elif isinstance(state, dict) and all(
            isinstance(k, str) and (
                k.startswith(("conv", "bn", "layer", "fc")) or "num_batches_tracked" in k
            ) for k in state.keys()
        ):
            backbone.load_state_dict(state, strict=True)
        else:
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
# UI
# =========================
st.set_page_config(page_title="Knife/Awl/Scissor Classifier", page_icon="🔎", layout="centered")
st.title("🔎 3-Class Classifier (ResNet)")
st.caption("knife / awl / scissor — 확률 예측 데모")

with st.sidebar.expander("🔍 Debug (secrets/env quick)", expanded=False):
    try:
        has_secret = "MODEL_FILE_ID" in st.secrets
    except Exception:
        has_secret = False
    st.write("has st.secrets['MODEL_FILE_ID']:", has_secret)
    st.write("env MODEL_FILE_ID set:", bool(os.getenv("MODEL_FILE_ID")))
    st.write("MODEL_PATH exists:", MODEL_PATH.exists())
    if MODEL_PATH.exists():
        st.write("MODEL_PATH size:", MODEL_PATH.stat().st_size)
with st.sidebar.expander("🔍 Label map check", expanded=False):
    st.write("MAP_PATH:", str(MAP_PATH))
    st.write("MAP exists:", MAP_PATH.exists())
    if MAP_PATH.exists():
        st.write("MAP size:", MAP_PATH.stat().st_size)
        try:
            with open(MAP_PATH, "r", encoding="utf-8") as f:
                jm = json.load(f)
            st.success(f"class_to_idx loaded. num_classes = {len(jm)} → keys: {list(jm.keys())}")
        except Exception as e:
            st.error(f"Failed to read class_to_idx.json: {e}")

# (옵션) Secrets 없이 테스트 시 임시 입력
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
            st.image(img, caption="입력 이미지", use_container_width=True)
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
