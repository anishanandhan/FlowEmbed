"""
LLM Alerter — Generate natural language security alerts using Ollama + Mistral.

When an anomalous or interesting flow is classified, this module generates
a plain-English explanation that a SOC analyst can actually act on.
"""

import logging
from typing import Optional, Dict
import hashlib
import json

from src.config import OLLAMA_MODEL, OLLAMA_HOST

logger = logging.getLogger(__name__)

# Cache for generated alerts (avoid redundant LLM calls)
_alert_cache: Dict[str, str] = {}


def _get_cache_key(traffic_class: str, confidence: float, features: dict) -> str:
    """Generate a cache key from classification result."""
    # Bucket confidence to increase cache hits
    conf_bucket = round(confidence, 1)
    # Hash features (rounded if numeric to reduce uniqueness)
    feat_hash = hashlib.md5(
        json.dumps({k: (round(v, 2) if isinstance(v, (int, float)) else str(v)) for k, v in features.items()}, sort_keys=True).encode()
    ).hexdigest()[:8]
    return f"{traffic_class}_{conf_bucket}_{feat_hash}"


def generate_alert(
    traffic_class: str,
    confidence: float,
    flow_features: dict,
    shap_top_features: Optional[list] = None,
    is_anomalous: bool = False,
    use_cache: bool = True,
) -> str:
    """
    Generate a natural language security alert using Ollama LLM.

    Args:
        traffic_class: Predicted traffic class (e.g., "malware_c2").
        confidence: Classification confidence (0-1).
        flow_features: Dict of flow features (src_ip, duration, pkt_size, etc.).
        shap_top_features: Top contributing features from SHAP.
        is_anomalous: Whether the flow was flagged as anomalous.
        use_cache: Whether to use cached responses.

    Returns:
        Natural language alert string.
    """
    # Check cache first
    if use_cache:
        cache_key = _get_cache_key(traffic_class, confidence, flow_features)
        if cache_key in _alert_cache:
            return _alert_cache[cache_key]

    # Build prompt
    prompt = _build_prompt(traffic_class, confidence, flow_features, shap_top_features, is_anomalous)

    try:
        alert = _call_ollama(prompt)
    except Exception as e:
        logger.warning(f"Ollama call failed: {e}. Using template fallback.")
        alert = _generate_template_alert(traffic_class, confidence, flow_features, is_anomalous)

    # Cache the result
    if use_cache:
        _alert_cache[cache_key] = alert

    return alert


def _build_prompt(
    traffic_class: str,
    confidence: float,
    flow_features: dict,
    shap_top_features: Optional[list],
    is_anomalous: bool,
) -> str:
    """Build the LLM prompt for alert generation."""
    severity = "HIGH" if is_anomalous or traffic_class in ("malware_c2", "vpn") else "MEDIUM"
    if confidence < 0.5:
        severity = "LOW"

    shap_str = ""
    if shap_top_features:
        shap_str = f"\nTop contributing features (SHAP): {', '.join(shap_top_features)}"

    return f"""You are an expert network security analyst at a Security Operations Center (SOC).
A network flow has been classified by our AI traffic classifier. Generate a concise, 
actionable security alert (2-3 sentences) explaining the classification and recommended action.

Classification Result:
- Traffic Class: {traffic_class}
- Confidence: {confidence:.1%}
- Severity: {severity}

Flow Characteristics:
- Source IP: {flow_features.get('src_ip', 'N/A')}
- Destination IP: {flow_features.get('dst_ip', 'N/A')}
- Protocol: {flow_features.get('protocol', 'N/A')}
- Duration: {flow_features.get('duration', 'N/A')}s
- Avg Packet Size: {flow_features.get('avg_pkt_size', 'N/A')} bytes
- Inter-Arrival Time: {flow_features.get('iat_mean', 'N/A')}ms
- Packets: {flow_features.get('total_packets', 'N/A')}
- Bytes: {flow_features.get('total_bytes', 'N/A')}
{shap_str}

{'⚠️ This flow was flagged as ANOMALOUS — it deviates significantly from known traffic patterns.' if is_anomalous else ''}

Generate a brief, professional alert with the classification reasoning and recommended SOC action.
Do NOT use markdown formatting. Keep it under 100 words."""


def _call_ollama(prompt: str) -> str:
    """Call Ollama API for text generation."""
    try:
        import ollama

        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 150},
        )

        return response["message"]["content"].strip()
    except ImportError:
        # Fallback to HTTP request if ollama package not available
        import urllib.request
        import json

        data = json.dumps({
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 150},
        }).encode()

        req = urllib.request.Request(
            f"{OLLAMA_HOST}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())
            return result["response"].strip()


def _generate_template_alert(
    traffic_class: str,
    confidence: float,
    flow_features: dict,
    is_anomalous: bool,
) -> str:
    """
    Generate a template-based alert when LLM is unavailable.
    Pre-written alerts for common scenarios ensure demo always works.
    """
    templates = {
        "streaming": (
            f"Flow from {flow_features.get('src_ip', 'unknown')} classified as "
            f"VIDEO STREAMING traffic ({confidence:.0%} confidence). "
            f"Pattern consistent with adaptive bitrate streaming — large packets "
            f"({flow_features.get('avg_pkt_size', 'N/A')} bytes avg) with regular inter-arrival times. "
            f"No action required — legitimate media consumption."
        ),
        "gaming": (
            f"Flow from {flow_features.get('src_ip', 'unknown')} classified as "
            f"GAMING traffic ({confidence:.0%} confidence). "
            f"Low-latency packet pattern detected — small, frequent packets with "
            f"consistent timing ({flow_features.get('iat_mean', 'N/A')}ms IAT). "
            f"Typical of real-time multiplayer protocols. No security concern."
        ),
        "voip": (
            f"Flow from {flow_features.get('src_ip', 'unknown')} classified as "
            f"VoIP/VIDEO CALL traffic ({confidence:.0%} confidence). "
            f"Bidirectional flow with codec-consistent packet sizes and regular timing. "
            f"Standard communication traffic — no action required."
        ),
        "malware_c2": (
            f"🚨 ALERT: Flow from {flow_features.get('src_ip', 'unknown')} to "
            f"{flow_features.get('dst_ip', 'unknown')} classified as "
            f"POTENTIAL C2 BEACON ({confidence:.0%} confidence). "
            f"Inter-arrival timing ({flow_features.get('iat_mean', 'N/A')}ms) matches known "
            f"APT beacon patterns. IMMEDIATE ACTION: Isolate source host, "
            f"check for lateral movement, escalate to incident response team."
        ),
        "vpn": (
            f"Flow from {flow_features.get('src_ip', 'unknown')} classified as "
            f"VPN TUNNEL traffic ({confidence:.0%} confidence). "
            f"Encrypted tunnel with uniform packet size distribution detected. "
            f"Verify against approved VPN policy. Unauthorized VPN usage may indicate "
            f"data exfiltration attempt."
        ),
        "social_media": (
            f"Flow from {flow_features.get('src_ip', 'unknown')} classified as "
            f"SOCIAL MEDIA traffic ({confidence:.0%} confidence). "
            f"Mixed content pattern: short API calls interspersed with media downloads. "
            f"Standard usage — monitor for excessive bandwidth consumption."
        ),
    }

    alert = templates.get(
        traffic_class,
        f"Flow from {flow_features.get('src_ip', 'unknown')} classified as "
        f"{traffic_class.upper()} ({confidence:.0%} confidence). "
        f"Review flow characteristics for policy compliance."
    )

    if is_anomalous:
        alert = f"⚠️ ANOMALOUS: {alert}"

    return alert


def preload_cache():
    """Pre-generate common alert templates into cache for instant responses."""
    common_scenarios = [
        ("streaming", 0.95, {"src_ip": "192.168.1.5", "dst_ip": "142.250.80.46", "avg_pkt_size": 1350, "iat_mean": 12.5, "total_packets": 15000, "duration": 120}),
        ("gaming", 0.92, {"src_ip": "192.168.1.5", "dst_ip": "52.36.110.2", "avg_pkt_size": 180, "iat_mean": 16.7, "total_packets": 8000, "duration": 300}),
        ("voip", 0.88, {"src_ip": "192.168.1.5", "dst_ip": "157.240.1.52", "avg_pkt_size": 200, "iat_mean": 20.0, "total_packets": 5000, "duration": 180}),
        ("malware_c2", 0.78, {"src_ip": "192.168.1.5", "dst_ip": "45.33.32.156", "avg_pkt_size": 64, "iat_mean": 60000, "total_packets": 120, "duration": 7200}),
        ("vpn", 0.85, {"src_ip": "192.168.1.5", "dst_ip": "104.16.249.0", "avg_pkt_size": 1400, "iat_mean": 5.2, "total_packets": 25000, "duration": 600}),
    ]

    for cls, conf, features in common_scenarios:
        cache_key = _get_cache_key(cls, conf, features)
        _alert_cache[cache_key] = _generate_template_alert(cls, conf, features, False)

    logger.info(f"Pre-loaded {len(common_scenarios)} alert templates into cache")
