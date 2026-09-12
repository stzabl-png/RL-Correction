#!/usr/bin/env bash
# One-shot Azure Kinect Sensor SDK 1.4.2 install for Ubuntu 22.04/24.04 (x86_64).
#
# Microsoft only publishes the SDK for Ubuntu 18.04/20.04; the 18.04 .deb files work on
# newer releases when combined with the jammy libsoundio1 package.  This script:
#   1. downloads libk4a1.4 / libk4a1.4-dev / k4a-tools (1.4.2) and libsoundio1 (1.1.0-1)
#   2. installs them with the EULA accepted non-interactively (ACCEPT_EULA=Y)
#   3. installs /etc/udev/rules.d/99-k4a.rules so non-root users can open the device
#
# Usage:  sudo scripts/install_kinect_sdk.sh            (from the repo root or anywhere)
#         sudo scripts/install_kinect_sdk.sh --uninstall
set -euo pipefail

K4A_VERSION="1.4.2"
MS_POOL="https://packages.microsoft.com/ubuntu/18.04/prod/pool/main"
SOUNDIO_DEB="libsoundio1_1.1.0-1_amd64.deb"
SOUNDIO_URL="http://archive.ubuntu.com/ubuntu/pool/universe/libs/libsoundio/${SOUNDIO_DEB}"
RULES_URL="https://raw.githubusercontent.com/microsoft/Azure-Kinect-Sensor-SDK/develop/scripts/99-k4a.rules"
CACHE_DIR="${K4A_CACHE_DIR:-${SUDO_USER:+/home/$SUDO_USER}/.cache/dexmate-calib/k4a}"
CACHE_DIR="${CACHE_DIR:-/tmp/dexmate-calib-k4a}"

log() { printf '\033[1;34m[k4a]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[k4a] %s\033[0m\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "run with sudo: sudo $0"
[[ "$(uname -m)" == "x86_64" ]] || die "only x86_64 is supported by the Microsoft packages"
. /etc/os-release
case "${VERSION_ID:-}" in
  18.04|20.04) log "Ubuntu ${VERSION_ID}: you can also use Microsoft's apt repository; continuing with .deb files";;
  22.04|24.04) ;;
  *) log "warning: untested Ubuntu ${VERSION_ID:-?}; continuing";;
esac

if [[ "${1:-}" == "--uninstall" ]]; then
  apt-get remove -y k4a-tools libk4a1.4-dev libk4a1.4 libsoundio1 || true
  rm -f /etc/udev/rules.d/99-k4a.rules
  udevadm control --reload-rules
  log "removed"
  exit 0
fi

mkdir -p "$CACHE_DIR"
cd "$CACHE_DIR"
fetch() {  # url  file
  if [[ -s "$2" ]]; then log "cached $2"; else log "downloading $2"; curl -fsSL -o "$2" "$1"; fi
}
fetch "${MS_POOL}/libk/libk4a${K4A_VERSION%.*}/libk4a${K4A_VERSION%.*}_${K4A_VERSION}_amd64.deb"        "libk4a1.4_${K4A_VERSION}_amd64.deb"
fetch "${MS_POOL}/libk/libk4a${K4A_VERSION%.*}-dev/libk4a${K4A_VERSION%.*}-dev_${K4A_VERSION}_amd64.deb" "libk4a1.4-dev_${K4A_VERSION}_amd64.deb"
fetch "${MS_POOL}/k/k4a-tools/k4a-tools_${K4A_VERSION}_amd64.deb"                                       "k4a-tools_${K4A_VERSION}_amd64.deb"
fetch "$SOUNDIO_URL" "$SOUNDIO_DEB"
fetch "$RULES_URL" 99-k4a.rules
grep -q '045e' 99-k4a.rules || die "99-k4a.rules looks wrong"

export DEBIAN_FRONTEND=noninteractive
export ACCEPT_EULA=Y   # honoured by libk4a1.4's preinst; replaces the debconf EULA dialog
apt-get update -qq
# libk4a1.4 first via dpkg so a half-installed state from a previous attempt cannot block apt.
dpkg -i "libk4a1.4_${K4A_VERSION}_amd64.deb"
apt-get install -f -y -qq
apt-get install -y -qq "./${SOUNDIO_DEB}" "./libk4a1.4-dev_${K4A_VERSION}_amd64.deb" "./k4a-tools_${K4A_VERSION}_amd64.deb"
dpkg --configure -a

install -m 644 99-k4a.rules /etc/udev/rules.d/99-k4a.rules
udevadm control --reload-rules
udevadm trigger

log "installed packages:"
dpkg -l | awk '/k4a|libsoundio1/ {print "  " $1, $2, $3}'
ls -1 /usr/lib/x86_64-linux-gnu/libk4a.so.${K4A_VERSION} /usr/lib/x86_64-linux-gnu/libk4a1.4/libdepthengine.so.2.0 >/dev/null \
  || die "libk4a / libdepthengine missing after install"
log "OK. Re-plug the Kinect if it was connected, then check:  lsusb | grep 045e   (needs 045e:097a SuperSpeed hub)"
