import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib, numpy as np, pandas as pd, cv2, streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['Calibri', 'DejaVu Sans']
from matplotlib.patches import Circle
import config
from src.masking import cell_mask
from src.extract_features import features

BG, PANEL, RULE, TXT, MUTED = "#120d1d", "#1a1329", "#33284a", "#ece6f7", "#9a8db3"
PINK, CYAN, AMBER = "#ff5c8a", "#4fc3d9", "#f0b429"

st.set_page_config(page_title="Malaria cell reader", layout="wide")
st.markdown(f"""<style>
.stApp {{ background:{BG}; color:{TXT}; font-family:Calibri,"Segoe UI",sans-serif; }}
[data-testid="stToolbar"], footer {{ visibility:hidden; }}
[data-testid="stSidebar"] {{ background:{PANEL}; border-right:1px solid {RULE}; }}
.block-container {{ padding-top:2.2rem; max-width:1250px; }}
.title {{ font-family:"Palatino Linotype","Book Antiqua",Palatino,Georgia,serif; font-weight:600; font-size:2rem; margin-bottom:0.2rem; }}
.lede {{ color:{MUTED}; max-width:68ch; margin-bottom:1.4rem; }}
.readouts {{ display:flex; gap:2.6rem; border-top:1px solid {RULE}; border-bottom:1px solid {RULE}; padding:0.9rem 0; margin-bottom:1.8rem; }}
.readouts span {{ display:block; color:{MUTED}; font-size:0.82rem; }}
.readouts b {{ font-family:"Palatino Linotype","Book Antiqua",Palatino,Georgia,serif; font-weight:600; font-size:1.25rem; }}
.num {{ font-family:Consolas,"Cascadia Mono",monospace; }}
.verdict {{ font-family:"Palatino Linotype","Book Antiqua",Palatino,Georgia,serif; font-weight:600; font-size:3.4rem; line-height:1.05; margin:0 0 0.4rem; }}
.sub {{ color:{MUTED}; max-width:60ch; }}
.gauge {{ position:relative; height:8px; background:linear-gradient(90deg,{CYAN} 50%,{PINK} 50%); margin:1.5rem 0 0.35rem; }}
.marker {{ position:absolute; top:-7px; width:3px; height:22px; background:{TXT}; transform:translateX(-1.5px); }}
.ends {{ display:flex; justify-content:space-between; color:{MUTED}; font-size:0.82rem; }}
.h {{ font-family:"Palatino Linotype","Book Antiqua",Palatino,Georgia,serif; font-weight:600; font-size:1.15rem; margin:1.9rem 0 0.5rem; }}
.cap {{ color:{MUTED}; font-size:0.85rem; max-width:62ch; }}
.cap b {{ color:{TXT}; font-weight:600; }}
</style>""", unsafe_allow_html=True)

CH = {"R": "Red", "G": "Green", "B": "Blue", "S": "Saturation", "V": "Brightness"}
STAT = {"mean": "average", "std": "spread", "p5": "low end", "p50": "median", "p95": "high end"}

def nice(c):
    a, _, b = c.partition("_")
    if a in CH and b in STAT:
        return f"{CH[a]} {STAT[b]}"
    if a == "H":
        return "Hue " + b.replace("_mean", "")
    if a == "glcm":
        return "Texture " + b
    return c.replace("_", " ").capitalize()

def square(a):
    s = max(a.shape[:2])
    out = np.zeros((s, s) + a.shape[2:], a.dtype)
    y, x = (s - a.shape[0]) // 2, (s - a.shape[1]) // 2
    out[y:y + a.shape[0], x:x + a.shape[1]] = a
    return out

@st.cache_resource
def load():
    art = joblib.load(config.MODELS / "all_lr.joblib")
    d = pd.read_parquet(config.PROCESSED / "features.parquet", columns=["path", "label", "slide_code", "split"])
    return art, d[d.split == "test"].reset_index(drop=True)

art, test = load()
pipe, cols = art["model"], art["cols"]
true_label, specimen = None, "Uploaded image"

st.sidebar.markdown("### Cell source")
source = st.sidebar.radio("Choose", ["Random held-out test cell", "Upload an image"], label_visibility="collapsed")
if source == "Random held-out test cell":
    want = st.sidebar.selectbox("Show", ["Any", "Parasitized", "Uninfected"])
    if "n" not in st.session_state:
        st.session_state.n = 0
    if st.sidebar.button("Show another cell"):
        st.session_state.n += 1
    pool = test if want == "Any" else test[test.label == want]
    row = pool.iloc[int(np.random.default_rng(st.session_state.n).integers(len(pool)))]
    path, true_label, specimen = row.path, row.label, f"Slide {row.slide_code}"
else:
    st.sidebar.warning("Use a single cell cut out on a black background, like the dataset images.")
    up = st.sidebar.file_uploader("Cell image", type=["png", "jpg", "jpeg"])
    if up is None:
        st.info("Upload a cell image, or switch to a random held-out test cell.")
        st.stop()
    arr = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_COLOR)
    if arr is None:
        st.error("That file could not be read as an image. Upload a PNG or JPG.")
        st.stop()
    cv2.imwrite(str(config.RESULTS / "_demo_upload.png"), arr)
    path = "results/_demo_upload.png"

bgr = cv2.imread(str(config.ROOT / path))
f = features(path)
x = pd.DataFrame([f])[cols]
p = float(pipe.predict_proba(x)[0, 1])
pred = "Parasitized" if p >= 0.5 else "Uninfected"
accent = PINK if pred == "Parasitized" else CYAN

rgb = square(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
m = square(cell_mask(bgr).astype(np.uint8)).astype(bool)
edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
ov = np.zeros(edge.shape + (4,), float)
ov[edge] = (0.91, 0.89, 0.96, 1.0)
fig1 = plt.figure(figsize=(4.6, 4.6))
fig1.patch.set_facecolor(BG)
ax1 = fig1.add_axes([0.02, 0.02, 0.96, 0.96])
clip = Circle((0.5, 0.5), 0.5, transform=ax1.transAxes)
for layer in (ax1.imshow(rgb), ax1.imshow(ov)):
    layer.set_clip_path(clip)
ax1.add_patch(Circle((0.5, 0.5), 0.5, transform=ax1.transAxes, fill=False, ec="#4a3a6b", lw=3))
ax1.axis("off")

sc, lr = pipe.named_steps["sc"], pipe.named_steps["m"]
contrib = lr.coef_[0] * sc.transform(x)[0]
top = np.argsort(-np.abs(contrib))[:8][::-1]
fig2, ax2 = plt.subplots(figsize=(6.2, 3.9))
fig2.patch.set_facecolor(BG); ax2.set_facecolor(BG)
ax2.barh([nice(cols[i]) for i in top], contrib[top], height=0.62,
         color=[PINK if contrib[i] > 0 else CYAN for i in top])
for s in ("top", "right", "left"):
    ax2.spines[s].set_visible(False)
ax2.spines["bottom"].set_color(RULE)
ax2.axvline(0, color=TXT, lw=0.8)
ax2.grid(axis="x", color=RULE, lw=0.8); ax2.set_axisbelow(True)
ax2.tick_params(length=0, labelsize=10, colors=TXT)
ax2.tick_params(axis="x", colors=MUTED)
ax2.set_xlabel("toward Uninfected   |   toward Parasitized", color=MUTED)
fig2.tight_layout()

st.markdown('<div class="title">Malaria cell reader</div>'
            '<div class="lede">Logistic regression on 38 shape, color and texture measurements. '
            'Cells come from the NIH malaria dataset, and test cells were never used for training.</div>',
            unsafe_allow_html=True)
st.markdown(f'<div class="readouts"><div><span>Specimen</span><b>{specimen}</b></div>'
            f'<div><span>Actual label</span><b>{true_label or "Not provided"}</b></div>'
            f'<div><span>Model score</span><b class="num">{p:.3f}</b></div></div>', unsafe_allow_html=True)

c1, c2 = st.columns([1, 1.15], gap="large")
with c1:
    st.pyplot(fig1)
    st.markdown('<div class="cap"><b>Fig. 1</b> Specimen field. The pale outline is the cell mask used for the shape measurements.</div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="verdict" style="color:{accent}">{pred}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub">A score above 0.5 is called Parasitized. 0 means certainly uninfected and 1 certainly parasitized.</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="gauge"><div class="marker" style="left:{p * 100:.1f}%"></div></div>'
                '<div class="ends"><span>Uninfected</span><span>Parasitized</span></div>', unsafe_allow_html=True)
    if true_label:
        if pred == true_label:
            st.markdown('<div class="sub" style="margin-top:0.9rem">Matches the actual label.</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="margin-top:0.9rem;color:{AMBER}">Differs from the actual label. This is one of the model errors, about 1 in 20 test cells.</div>', unsafe_allow_html=True)
    st.markdown('<div class="h">What decided this call</div>', unsafe_allow_html=True)
    st.pyplot(fig2)
    st.markdown('<div class="cap"><b>Fig. 2</b> The eight measurements that moved this cell\'s score most. Bars to the right point toward Parasitized, bars to the left toward Uninfected.</div>', unsafe_allow_html=True)

st.markdown(f'<div class="cap" style="margin-top:2.2rem">Research demo, not a diagnostic tool.</div>', unsafe_allow_html=True)

