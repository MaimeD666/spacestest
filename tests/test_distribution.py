from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_runtime_install_does_not_require_git_executable() -> None:
    requirements = (PROJECT_ROOT / "requirements.txt").read_text("utf-8")
    assert "git+" not in requirements
    assert "/CLIP/archive/68dce32140994dfcb645a1320c4ebdc034fc19fd.zip" in requirements


def test_cross_platform_entrypoints_are_present() -> None:
    assert (PROJECT_ROOT / "install.py").is_file()
    assert (PROJECT_ROOT / "start.sh").is_file()
    assert (PROJECT_ROOT / "start.ps1").is_file()
