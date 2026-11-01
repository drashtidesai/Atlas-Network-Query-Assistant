"""Narrow host-role checks from structured topology chunks; not general fact checking."""

import re


def ntp_host_issues(draft: str, records: list[dict]) -> list[str]:
    """Detect explicit placement of an NTP application on a modeled WAN router.

    Facts come from devices.csv and services.csv rows in the local export. This
    teaching-corpus rule covers one known wording pattern, not arbitrary prose.
    """
    routers = set()
    servers = []
    for record in records:
        source = record["metadata"].get("source", "")
        fields = dict(line.split(": ", 1) for line in record["page_content"].splitlines()
                      if ": " in line)
        if source == "corpus/topology/devices.csv" and fields.get("role") == "WAN router":
            if fields.get("hostname"):
                routers.add(fields["hostname"])
        if source == "corpus/topology/services.csv" and fields.get("protocol") == "UDP 123":
            servers.append(f"{fields.get('hostname', '')} ({fields.get('address', '')})")
    issues = []
    for hostname in sorted(routers):
        pattern = rf"\bNTP\s+application\s+on\s+(?:the\s+)?{re.escape(hostname)}\b"
        if re.search(pattern, draft, re.IGNORECASE):
            issues.append(
                f"NTP application attributed to {hostname}, modeled as a WAN router in devices.csv. "
                f"The modeled NTP service hosts in services.csv are: {', '.join(servers) or 'unavailable'}. "
                "Review the service host versus reporting-device distinction."
            )
    return issues
