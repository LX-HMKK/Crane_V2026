#!/usr/bin/env bash
# Install the early-start competition service on an RDK X5. Run as root on the board.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVICE_SRC="$REPO_DIR/tools/crane-vision.service"
SERVICE_DST="/etc/systemd/system/crane-vision.service"
AUTOSTART="/home/sunrise/.config/autostart/Crane.desktop"
AUTOSTART_BACKUP="${AUTOSTART}.pre-headless"

if [[ ${EUID} -ne 0 ]]; then
    echo "Run this installer as root." >&2
    exit 1
fi

install -m 0644 "$SERVICE_SRC" "$SERVICE_DST"

# Deployments commonly copy the workspace as root. Repair its ownership here,
# outside the critical boot path, so the service user can read updates.
chown -R sunrise:sunrise "$REPO_DIR"
if [[ -f "$REPO_DIR/scripts/watchdog.sh" ]]; then
    chmod 0755 "$REPO_DIR/scripts/watchdog.sh"
fi

# The old XFCE entry starts a second GUI too late for power-on competition mode
# and would contend with the service-owned cameras and serial port.
if [[ -f "$AUTOSTART" ]]; then
    if [[ ! -f "$AUTOSTART_BACKUP" ]]; then
        cp -p "$AUTOSTART" "$AUTOSTART_BACKUP"
    fi
    if grep -q '^Hidden=' "$AUTOSTART"; then
        sed -i 's/^Hidden=.*/Hidden=true/' "$AUTOSTART"
    else
        printf '\nHidden=true\n' >> "$AUTOSTART"
    fi
fi

systemctl daemon-reload
systemctl enable crane-vision.service

echo "crane-vision.service is enabled for the next boot; it opens the UI when X11 is ready."
echo "For calibration, stop it first: systemctl stop crane-vision.service"
