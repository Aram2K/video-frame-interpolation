#!/bin/bash
# Refresh the small "viewing" package with every run finished so far (previews, comparisons, logs).
# Usage: scripts/pack_previews.sh [SHOT]
R=${MVFI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)}
ONLY=previews DEL=$R/delivery_previews "$R/scripts/package_delivery.sh" "${1:-test01}"
