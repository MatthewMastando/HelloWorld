"""Fail if any installed dependency carries a license outside the allow-list."""

from __future__ import annotations

import json
import subprocess
import sys

ALLOWED_TOKENS = (
    "mit",
    "bsd",
    "apache",
    "isc",
    "psf",
    "python software foundation",
    "mozilla public license 2.0",
    "mpl-2.0",
    "mpl 2.0",
)

IGNORE_PACKAGES = {"trw", "helloworld-workspace"}


def allowed(license_text: str) -> bool:
    text = license_text.lower()
    return any(tok in text for tok in ALLOWED_TOKENS)


def main() -> int:
    out = subprocess.run(
        [sys.executable, "-m", "piplicenses", "--format=json", "--with-system"],
        check=True,
        capture_output=True,
        text=True,
    )
    packages = json.loads(out.stdout)
    offenders = []
    for pkg in packages:
        name = (pkg.get("Name") or "").lower()
        if name in IGNORE_PACKAGES:
            continue
        licenses = pkg.get("License") or pkg.get("License-Expression") or ""
        license_text = licenses if isinstance(licenses, str) else "; ".join(str(x) for x in licenses)
        if not license_text.strip() or license_text.strip().upper() == "UNKNOWN":
            offenders.append((pkg.get("Name"), license_text or "UNKNOWN"))
        elif not allowed(license_text):
            offenders.append((pkg.get("Name"), license_text))
    if offenders:
        print("Disallowed or unknown licenses found:")
        for name, lic in offenders:
            print(f"  {name}: {lic}")
        return 1
    print(f"All {len(packages)} installed packages use allowed licenses.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
