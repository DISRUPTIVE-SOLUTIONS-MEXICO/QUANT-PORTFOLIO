import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_core.capability_manifest import validate_manifest, write_manifest  # noqa: E402


def main() -> int:
    valid, errors = validate_manifest()
    if not valid:
        raise RuntimeError("; ".join(errors))
    target = ROOT / "FEATURE_PRESERVATION_MANIFEST.json"
    write_manifest(target)
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
