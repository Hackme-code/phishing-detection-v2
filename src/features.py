import math
import re
from urllib.parse import urlparse

SUSPICIOUS_TLDS = {"ga", "tk", "ml", "cf", "gq", "top", "xyz", "fun", "click", "icu", "buzz"}
HOSTING_PLATFORMS = {"firebaseapp.com", "web.app", "weeblysite.com", "godaddysites.com",
                     "square.site", "github.io", "wixsite.com", "blogspot.com",
                     "000webhostapp.com", "netlify.app", "pages.dev", "glitch.me"}
IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def get_host(url: str) -> str:
    url = url.strip()
    if not re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", url):
        url = "http://" + url
    host = (urlparse(url).hostname or "").lower()
    return re.sub(r"^www\.", "", host)


def entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = {c: s.count(c) for c in set(s)}
    return -sum(n / len(s) * math.log2(n / len(s)) for n in counts.values())


def extract_features(url: str) -> dict:
    host = get_host(url)
    labels = host.split(".") if host else []
    tld = labels[-1] if labels else ""
    sld = labels[-2] if len(labels) >= 2 else (labels[0] if labels else "")
    letters = sum(c.isalpha() for c in host)
    digits = sum(c.isdigit() for c in host)
    return {
        "host_len": len(host),
        "n_labels": len(labels),
        "n_dots": host.count("."),
        "n_hyphens": host.count("-"),
        "n_digits": digits,
        "digit_ratio": digits / len(host) if host else 0.0,
        "letter_ratio": letters / len(host) if host else 0.0,
        "longest_label": max((len(l) for l in labels), default=0),
        "sld_len": len(sld),
        "sld_entropy": entropy(sld),
        "tld_len": len(tld),
        "is_ip": int(bool(IP_RE.match(host))),
        "suspicious_tld": int(tld in SUSPICIOUS_TLDS),
        "hosting_platform": int(any(host == p or host.endswith("." + p) for p in HOSTING_PLATFORMS)),
        "has_punycode": int("xn--" in host),
    }