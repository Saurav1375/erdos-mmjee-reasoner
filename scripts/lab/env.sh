# Shared settings for the lab helper scripts (sourced, not executed).
LAB_HOST="${LAB_HOST:-ml-lab}"
LAB_ROOT="${LAB_ROOT:-~/saurav_mmjee}"            # everything of this project lives here
LAB_PROJECT="$LAB_ROOT/mmjee-reasoner"
LOCAL_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
