"""Package existing files only; never execute experiments or regenerate results."""
from pathlib import Path
import hashlib
import json
import zipfile


def package_results(root=".", destination=None):
    """Create a submission ZIP from the files already saved in root.

    Save the main notebook in root before calling this function. This does not
    capture an open browser notebook or overwrite the original execution record.
    """
    root = Path(root).resolve()
    if destination is None:
        destination = root.parent / "AI624_PA2_Submission.zip"
    destination = Path(destination).resolve()
    for name in ["AI624_PA2.ipynb", "REPORT.md", "README.md"]:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Required submission file is missing: {name}")

    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.resolve() == destination:
            continue
        relative = path.relative_to(root)
        if relative.parts[0] == "data" or path.name == "weights.zip":
            continue
        if ".git" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.name.endswith("_resume.pt") or path.name == "SHA256SUMS.json":
            continue
        files.append(path)

    manifest = {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }
    manifest_path = root / "SHA256SUMS.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    files.append(manifest_path)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(destination) as archive:
        bad_file = archive.testzip()
        if bad_file is not None:
            raise ValueError(f"ZIP integrity check failed for {bad_file}")
    print(f"Saved existing submission files to {destination}")
    return destination
