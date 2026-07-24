#!/usr/bin/env bash

set -euo pipefail

for cpu_online in /sys/devices/system/cpu/cpu*/online; do
  [[ -f "${cpu_online}" ]] || continue
  printf '1\n' | sudo tee "${cpu_online}" >/dev/null
done

grep -H . /sys/devices/system/cpu/cpu*/online 2>/dev/null
