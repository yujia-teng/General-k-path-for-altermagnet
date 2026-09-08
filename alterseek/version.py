"""Use project metadata for source checkouts and distribution metadata for installs."""

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib


def _get_version():
    # A source checkout can shadow an older installed distribution on PYTHONPATH.
    project_file = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if project_file.is_file():
        with project_file.open("rb") as stream:
            project = tomllib.load(stream).get("project", {})
        if project.get("name") == "alterseek-path":
            return project["version"]
    try:
        return version("alterseek-path")
    except PackageNotFoundError:
        return "unknown"


__version__ = _get_version()
