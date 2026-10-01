"""Common Alerting Protocol (CAP 1.2, OASIS) export.

CAP is the standard used by national warning authorities, Google Public Alerts, WMO's alert hub
and many SMS/cell-broadcast gateways. Each alert produces one <info> block per language.
"""

from __future__ import annotations

from datetime import UTC, datetime
from xml.sax.saxutils import escape

CAP_NS = "urn:oasis:names:tc:emergency:cap:1.2"

_SEVERITY = {"high": ("Severe", "Likely", "Immediate"), "medium": ("Moderate", "Possible", "Expected"),
             "low": ("Minor", "Unlikely", "Future")}


def _ts(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def alert_to_cap(alert, site, sender: str, base_url: str, references: str | None = None) -> str:
    severity, certainty, urgency = _SEVERITY.get(alert.level, _SEVERITY["low"])
    msg_type = "Cancel" if alert.status == "cancelled" else ("Update" if alert.kind in ("update", "all_clear")
                                                             else "Alert")
    sent = alert.dispatched_at or alert.approved_at or alert.created_at
    area_desc = escape(f"Downstream of {site.name} ({site.code})")
    circle = f"{site.lat:.4f},{site.lon:.4f} 5"
    infos = []
    for lang, cap_lang in (("ne", "ne-NP"), ("en", "en-US")):
        title = getattr(alert, f"title_{lang}")
        body = getattr(alert, f"body_{lang}")
        infos.append(f"""  <info>
    <language>{cap_lang}</language>
    <category>Geo</category>
    <category>Met</category>
    <event>{escape(title)}</event>
    <responseType>{"Evacuate" if alert.level == "high" else "Monitor"}</responseType>
    <urgency>{urgency}</urgency>
    <severity>{severity}</severity>
    <certainty>{certainty}</certainty>
    <effective>{_ts(alert.issued_at)}</effective>
    {f"<expires>{_ts(alert.expires_at)}</expires>" if alert.expires_at else ""}
    <senderName>{escape(sender)}</senderName>
    <headline>{escape(title[:160])}</headline>
    <description>{escape(body)}</description>
    <web>{escape(base_url.rstrip('/') + '/#site=' + site.code)}</web>
    <parameter><valueName>HimSatSiteCode</valueName><value>{escape(site.code)}</value></parameter>
    <parameter><valueName>HimSatRiskScore</valueName><value>{alert.facts.get('score', '')}</value></parameter>
    <area>
      <areaDesc>{area_desc}</areaDesc>
      <circle>{circle}</circle>
    </area>
  </info>""")
    refs = f"\n  <references>{escape(references)}</references>" if references else ""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<alert xmlns="{CAP_NS}">
  <identifier>urn:uuid:{alert.uid}</identifier>
  <sender>{escape(sender)}</sender>
  <sent>{_ts(sent)}</sent>
  <status>Actual</status>
  <msgType>{msg_type}</msgType>
  <scope>Public</scope>{refs}
{chr(10).join(infos)}
</alert>
"""
