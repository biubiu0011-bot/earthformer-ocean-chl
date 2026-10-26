"""Release-readiness checks for the public repository layout."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_required_public_repository_files_exist():
    required = [
        "README.md", "LICENSE", "environment.yml", ".gitignore",
        "data/README.md", "results/README.md", "src/earthformer/__init__.py",
        "src/earthformer_ocean_chl/__init__.py", "src/earthformer_ocean_chl/paths.py",
        "configs/nemo_pretrain.yaml", "configs/argo_transfer.yaml",
        "scripts/train_nemo_pretrain.py", "scripts/train_argo_transfer.py",
        "notebooks/argo", "analysis", "assets",
    ]
    missing = [path for path in required if not (ROOT / path).exists()]
    assert not missing, f"missing public repository paths: {missing}"


def test_scripts_do_not_contain_the_original_server_root():
    server_root = "/data/Lixiaoling"
    files = [
        *ROOT.joinpath("scripts").rglob("*.py"),
        *ROOT.joinpath("baselines").rglob("*.py"),
        *ROOT.joinpath("src", "earthformer_ocean_chl").rglob("*.py"),
    ]
    offenders = [str(path.relative_to(ROOT)) for path in files
                if server_root in path.read_text(encoding="utf-8", errors="ignore")]
    assert not offenders, f"hard-coded server paths remain: {offenders}"


def test_notebooks_are_kept_out_of_scripts():
    assert not list(ROOT.joinpath("scripts").rglob("*.ipynb"))


def test_legacy_notebooks_are_ignored_by_git():
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "notebooks/**/*.ipynb" in gitignore
