#!/usr/bin/env bash
# Publish the policy assistant at https://bbl.mcp-digitalstudio.com
#
#   sudo bash deploy/install.sh
#
# Adds one ingress rule to the existing Cloudflare Tunnel and installs the app
# as a systemd service. The tunnel config is backed up first and validated
# before anything is restarted, because it also routes ssh, postgres and ~25
# other hostnames.

set -euo pipefail

HOSTNAME_FQDN="bbl.mcp-digitalstudio.com"
APP_PORT="8100"
TUNNEL_CONFIG="/etc/cloudflared/config.yml"
PROJECT_DIR="/home/monchai/policy-rag-agents"
UNIT_NAME="policy-assistant.service"

if [[ $EUID -ne 0 ]]; then
  echo "This script must run as root: sudo bash deploy/install.sh" >&2
  exit 1
fi

# --- 1. app service ---------------------------------------------------------
# A server started by hand during development would still be holding the port.
echo "==> stopping any hand-started server"
pkill -f "${PROJECT_DIR}/.venv/bin/python server.py" 2>/dev/null || true
sleep 2

echo "==> installing ${UNIT_NAME}"
install -m 644 "${PROJECT_DIR}/deploy/${UNIT_NAME}" "/etc/systemd/system/${UNIT_NAME}"
systemctl daemon-reload
systemctl enable --now "${UNIT_NAME}"

echo "==> waiting for the app to answer on 127.0.0.1:${APP_PORT}"
for _ in $(seq 1 60); do
  if curl -sf -o /dev/null "http://127.0.0.1:${APP_PORT}/api/meta"; then
    echo "    up"
    break
  fi
  sleep 2
done

# --- 2. tunnel ingress ------------------------------------------------------
BACKUP="${TUNNEL_CONFIG}.bak.bbl-$(date +%Y%m%d_%H%M%S)"
echo "==> backing up ${TUNNEL_CONFIG} -> ${BACKUP}"
cp -a "${TUNNEL_CONFIG}" "${BACKUP}"

echo "==> adding ingress rule for ${HOSTNAME_FQDN}"
python3 - "$TUNNEL_CONFIG" "$HOSTNAME_FQDN" "$APP_PORT" <<'PYEOF'
import re
import sys

config_path, fqdn, port = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(config_path, encoding="utf-8").read().splitlines(keepends=True)

if any(fqdn in line for line in lines):
    print(f"    {fqdn} already present, leaving the config untouched")
    raise SystemExit(0)

# The catch-all wildcard must stay last, so insert immediately above it.
pattern = re.compile(r'^(\s*)-\s*hostname:\s*"?\*\.')
for index, line in enumerate(lines):
    match = pattern.match(line)
    if match:
        indent = match.group(1)
        lines[index:index] = [
            f"{indent}- hostname: {fqdn}\n",
            f"{indent}  service: http://localhost:{port}\n",
        ]
        break
else:
    raise SystemExit("could not find the wildcard ingress rule; aborting")

open(config_path, "w", encoding="utf-8").writelines(lines)
print(f"    inserted {fqdn} -> http://localhost:{port}")
PYEOF

echo "==> validating the tunnel config"
if ! cloudflared tunnel --config "${TUNNEL_CONFIG}" ingress validate; then
  echo "!! validation failed — restoring ${BACKUP} and leaving the tunnel alone" >&2
  cp -a "${BACKUP}" "${TUNNEL_CONFIG}"
  exit 1
fi

echo "==> restarting cloudflared (all tunnel hostnames blip for a few seconds)"
systemctl restart cloudflared

# --- 3. report --------------------------------------------------------------
sleep 5
echo
systemctl --no-pager --lines=0 status "${UNIT_NAME}" cloudflared | grep -E 'Active:|●' || true
echo
echo "Done. https://${HOSTNAME_FQDN}"
echo "Rollback: cp -a ${BACKUP} ${TUNNEL_CONFIG} && systemctl restart cloudflared"
