"""Удалённый доступ: страница и MCP — по адресу этой машины в Tailscale.

Tailscale — частная сеть между своими устройствами: адреса 100.x.y.z видны только в
ней, трафик зашифрован, порты в интернет не открываются. web --remote слушает, кроме
127.0.0.1, ещё и этот адрес; в локальную сеть и интернет — ничего.

    python kbrag.py web --remote          страница :8000 и MCP по HTTP :8200
    KBRAG_REMOTE_HOST=…                   другой адрес вместо Tailscale (своя VPN)

Пускает только по токенам (.local/access.yaml): без людей с токенами — не запускается.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .machine import value

TAILSCALE_WINDOWS = Path(r"C:\Program Files\Tailscale\tailscale.exe")


@dataclass(frozen=True)
class Address:
    ip: str
    names: tuple[str, ...] = ()  # имя машины в Tailscale (MagicDNS): comp.tailnet.ts.net


def _tailscale() -> str | None:
    found = shutil.which("tailscale")
    if found:
        return found
    return str(TAILSCALE_WINDOWS) if TAILSCALE_WINDOWS.exists() else None


def find() -> tuple[Address | None, str]:
    """Адрес для удалённых машин и, если его нет, — почему (что сделать)."""
    own = value("KBRAG_REMOTE_HOST", "")
    if own:
        return Address(own), ""
    exe = _tailscale()
    if exe is None:
        return None, "Tailscale не установлен: tailscale.com/download, войти и включить"
    try:
        out = subprocess.run(
            [exe, "status", "--json"], capture_output=True, text=True, timeout=15
        ).stdout
        status = json.loads(out or "{}")
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        return None, f"Tailscale не ответил ({type(exc).__name__})"
    if status.get("BackendState") != "Running":
        state = status.get("BackendState") or "не запущен"
        return None, f"Tailscale: {state} — войдите в него и включите (tailscale up)"
    me = status.get("Self") or {}
    ips = [ip for ip in me.get("TailscaleIPs") or [] if "." in ip]  # IPv4: 100.x.y.z
    if not ips:
        return None, "у этой машины нет адреса в Tailscale"
    dns = str(me.get("DNSName") or "").rstrip(".")
    names = (dns, dns.split(".")[0]) if dns else ()
    return Address(ips[0], tuple(n for n in names if n)), ""
