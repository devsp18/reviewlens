"""Self-contained animated stat tiles.

Rendered via components.v1.html (an iframe) rather than injected CSS/markdown -
an iframe fully owns its own styling, so it can't lose a specificity fight
against Streamlit's own component CSS the way a <style> block injected into
the main DOM can.
"""

import streamlit.components.v1 as components

_CARD_TEMPLATE = """
<div id="root" style="font-family: Inter, sans-serif;">
  <style>
    :root {{ color-scheme: light dark; }}
    .tile {{
      border-radius: 16px;
      padding: 20px 24px;
      background: light-dark(#F8FAFC, #151B2B);
      border: 1px solid light-dark(#E2E8F0, #1E293B);
    }}
    .value {{
      font-family: "JetBrains Mono", monospace;
      font-size: 2.25rem;
      font-weight: 700;
      color: {accent};
      line-height: 1.1;
    }}
    .label {{
      font-size: 0.875rem;
      color: light-dark(#64748B, #94A3B8);
      margin-top: 4px;
    }}
  </style>
  <div class="tile">
    <div class="value" id="val">{start}{suffix}</div>
    <div class="label">{label}</div>
  </div>
</div>
<script>
  const target = {value};
  const suffix = {suffix_js};
  const el = document.getElementById("val");
  const duration = 900;
  const startTime = performance.now();
  const isInt = Number.isInteger(target);
  function frame(now) {{
    const t = Math.min(1, (now - startTime) / duration);
    const eased = 1 - Math.pow(1 - t, 3);
    const current = target * eased;
    el.textContent = (isInt ? Math.round(current).toLocaleString() : current.toFixed(1)) + suffix;
    if (t < 1) requestAnimationFrame(frame);
  }}
  requestAnimationFrame(frame);
</script>
"""


def animated_metric(label: str, value: float, suffix: str = "", accent: str = "#4F46E5", height: int = 110) -> None:
    import json
    html = _CARD_TEMPLATE.format(
        label=label, value=value, suffix="", start="0", accent=accent, suffix_js=json.dumps(suffix),
    )
    components.html(html, height=height)
