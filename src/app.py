import json
import re
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
from src.features import extract_features, get_host  # noqa: E402


@st.cache_resource
def load():
    model = joblib.load(ROOT / "models" / "url_model.joblib")
    meta = json.loads((ROOT / "models" / "url_meta.json").read_text())
    return model, meta


model, meta = load()
LOW_MAX, HIGH_MIN = meta["low_max"], meta["high_min"]
URL_RE = re.compile(r"(?:https?://|www\.)[^\s<>\"')\]]+", re.IGNORECASE)
BARE_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*\.[a-z]{2,}(?:/\S*)?$", re.IGNORECASE)


def find_links(text: str) -> list:
    links = [u.rstrip(".,;:!?") for u in URL_RE.findall(text)]
    if not links:  # a bare domain such as example.com
        links = [t for t in text.split() if BARE_RE.match(t)]
    seen, out = set(), []
    for u in links:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def band(p: float) -> str:
    if p >= HIGH_MIN:
        return "HIGH"
    if p >= LOW_MAX:
        return "MEDIUM"
    return "LOW"


def reasons(f: dict) -> list:
    r = []
    if f["is_ip"]:
        r.append("The host is an IP address")
    if f["hosting_platform"]:
        r.append("Hosted on a free website platform")
    if f["suspicious_tld"]:
        r.append("Uses a TLD often abused for phishing")
    if f["bait_words"]:
        r.append("Host contains bait words (login, secure, verify...)")
    if f["n_hyphens"] >= 2:
        r.append(f"{f['n_hyphens']} hyphens in the host")
    if f["longest_digit_run"] >= 4:
        r.append("Long run of digits in the host")
    if f["n_labels"] >= 4:
        r.append(f"{f['n_labels']} dot-separated parts (many subdomains)")
    if f["host_len"] >= 30:
        r.append(f"Long host name ({f['host_len']} characters)")
    if f["has_punycode"]:
        r.append("Contains punycode (possible look-alike characters)")
    return r or ["No specific warning signs found in the host name"]


st.set_page_config(page_title="Phishing Link Checker", layout="wide")
st.title("Phishing Link Checker")
st.caption(
    "ICT606 Assessment 3. Paste a link or a whole email. The model scores the "
    "host name only (not the page, path or email wording)."
)

text = st.text_area("Paste a link or email text", height=220,
                    placeholder="Paste a URL or the full text of an email here...")

if st.button("Check", type="primary"):
    links = find_links(text)
    if not links:
        st.warning("No links found in the text.")
    else:
        rows = []
        for u in links:
            feats = extract_features(u)
            x = pd.DataFrame([feats])[meta["features"]]
            p = float(model.predict_proba(x)[0, 1])
            rows.append((u, p, band(p), feats))

        order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        rows.sort(key=lambda r: (order[r[2]], -r[1]))
        st.subheader(f"{len(rows)} link(s) found")
        for u, p, b, feats in rows:
            with st.container(border=True):
                c1, c2 = st.columns([3, 1])
                c1.markdown(f"**{u}**  \nHost checked: `{get_host(u)}`")
                c2.metric("Probability of phishing", f"{p:.1%}")
                msg = f"{b} RISK"
                if b == "HIGH":
                    st.error(msg)
                elif b == "MEDIUM":
                    st.warning(msg)
                else:
                    st.success(msg + " (this does not mean the link is safe)")
                for r in reasons(feats):
                    st.write("- " + r)

st.info(
    "Limitations: only the host name is analysed. Paths, query strings, the page "
    "content, the certificate and the email's wording are ignored. At the HIGH "
    f"threshold ({HIGH_MIN:.2f}) the model catches about 67% of phishing hosts and "
    "flags about 4% of legitimate ones, so a LOW result does not mean safe. It is a "
    "first-layer screening aid, not a verdict."
)