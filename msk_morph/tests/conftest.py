"""
conftest.py
=================
Shared pytest configuration, CLI options, and session-scoped fixtures
for all deformetrica_registration tests.

"""

import importlib.util
import shutil
from pathlib import Path

import numpy as np
import pytest


from utils.mesh_loader import load_with_fallbacks

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE           = Path(__file__).resolve().parent
TESTS_DIR       = _HERE / "tests" if (_HERE / "tests").exists() else _HERE
FIXTURES_DIR    = TESTS_DIR / "fixtures"
FIXTURES_MODULE = FIXTURES_DIR / "patella_mesh_fixtures.py"
REPO_ROOT       = TESTS_DIR.parent

# Stable, in-repo dirs for real-test I/O (used instead of pytest tmp_path)
# so inputs/outputs persist for inspection and --update-golden
REAL_INPUT_DIR  = FIXTURES_DIR / "registration_input"
REAL_OUTPUT_DIR = FIXTURES_DIR / "registration_output"


# ---------------------------------------------------------------------------
# CLI options
# ---------------------------------------------------------------------------

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real: marks tests that require Deformetrica installed "
        "(deselect with -m 'not real')"
    )


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help=(
            "Overwrite EXPECTED_* arrays in patella_mesh_fixtures.py with the "
            "output from the current registration run, then skip the comparison. "
            "Review the diff and commit the result."
        ),
    )
    parser.addoption(
        "--template-mesh",
        default=None,
        metavar="PATH",
        help=(
            "Path to a VTK/STL file to use as the template mesh instead of the "
            "inline arrays in patella_mesh_fixtures.py.  "
            "Use tests/fixtures/update_fixtures.py to permanently bake a new mesh in."
        ),
    )
    parser.addoption(
        "--target-mesh",
        default=None,
        metavar="PATH",
        help=(
            "Path to a VTK/STL file to use as the target mesh instead of the "
            "inline arrays in patella_mesh_fixtures.py."
        ),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _import_fixtures():
    """Dynamically import patella_mesh_fixtures so its path is always explicit."""
    spec = importlib.util.spec_from_file_location(
        "patella_mesh_fixtures", FIXTURES_MODULE
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _read_mesh_arrays(path: Path):
    """
    Return (points ndarray, cells ndarray) from a VTK/STL file using
    the project's own mesh_loader utility with its Open3D/VTK/meshio fallbacks.
    """
    mesh = load_with_fallbacks(str(path), verbose=False)

    # open3d TriangleMesh
    try:
        import open3d as o3d
        if isinstance(mesh, o3d.geometry.TriangleMesh):
            import numpy as np
            return (
                np.asarray(mesh.vertices, dtype="float32"),
                np.asarray(mesh.triangles, dtype="int64"),
            )
    except ImportError:
        pass

    # meshio fallback (returned as meshio.Mesh when open3d absent)
    if hasattr(mesh, "points") and hasattr(mesh, "cells"):
        return mesh.points, mesh.cells[0].data

    # numpy array (points only — no connectivity)
    import numpy as np
    if isinstance(mesh, np.ndarray):
        raise ValueError(
            f"load_with_fallbacks returned only a point array for {path}; "
            "cell/connectivity data is required to write a VTK mesh."
        )

    raise ValueError(f"Unrecognised mesh type returned by load_with_fallbacks: {type(mesh)}")


def _write_vtk(points: np.ndarray, cells: np.ndarray, path: Path) -> None:
    """
    Write a triangle-mesh VTK (ASCII PolyData) with the real vertex
    coordinates and connectivity written directly. This mirrors the writer
    in test_registration.py so fixtures and pipeline share one format.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_pts   = len(points)
    n_cells = len(cells)

    with open(str(path), "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("mesh\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n_pts} float\n")
        for pt in points:                          # write real points directly
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
        f.write(f"POLYGONS {n_cells} {n_cells * 4}\n")
        for tri in cells:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")


def _read_vtk_arrays(path: Path):
    """
    Read a VTK file back into (points, cells) arrays using the project's
    mesh_loader, mirroring exactly what the registration pipeline does.
    """
    return _read_mesh_arrays(path)


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fixture_data():
    """Load inline mesh arrays from patella_mesh_fixtures.py once per session."""
    return _import_fixtures()


@pytest.fixture(scope="session")
def template_vtk(request, fixture_data):
    """
    Write the template mesh to a VTK file in the fixtures input dir.

    Source priority:
      1. --template-mesh CLI path  (lets you swap meshes without editing any file)
      2. Inline arrays in patella_mesh_fixtures.py  (default, reproducible)
    """
    cli_path = request.config.getoption("--template-mesh")

    if cli_path:
        src = Path(cli_path)
        if not src.exists():
            pytest.exit(f"--template-mesh path not found: {src}", returncode=1)
        # If already a VTK, use it directly — no need to round-trip through arrays
        if src.suffix.lower() == ".vtk":
            print(f"\n[fixture] template VTK used directly from CLI: {src}")
            return src
        points, cells = _read_mesh_arrays(src)
        print(f"\n[fixture] template mesh loaded from CLI: {src}  "
              f"({len(points)} pts, {len(cells)} faces)")
    else:
        points = fixture_data.TEMPLATE_POINTS
        cells  = fixture_data.TEMPLATE_CELLS

    path = REAL_INPUT_DIR / "patella_r_template_aligned.vtk"
    REAL_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_vtk(points, cells, path)
    return path


@pytest.fixture(scope="session")
def target_vtk(request, fixture_data):
    """
    Write the target mesh to a VTK file in the fixtures input dir.

    Source priority:
      1. --target-mesh CLI path
      2. Inline arrays in patella_mesh_fixtures.py
    """
    cli_path = request.config.getoption("--target-mesh")

    if cli_path:
        src = Path(cli_path)
        if not src.exists():
            pytest.exit(f"--target-mesh path not found: {src}", returncode=1)
        if src.suffix.lower() == ".vtk":
            print(f"\n[fixture] target VTK used directly from CLI: {src}")
            return src
        points, cells = _read_mesh_arrays(src)
        print(f"\n[fixture] target mesh loaded from CLI: {src}  "
              f"({len(points)} pts, {len(cells)} faces)")
    else:
        points = fixture_data.TARGET_POINTS
        cells  = fixture_data.TARGET_CELLS

    path = REAL_INPUT_DIR / "patella_r_target_original.vtk"
    REAL_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    _write_vtk(points, cells, path)
    return path


@pytest.fixture(scope="session")
def aligned_meshes_dir(template_vtk, target_vtk):
    """
    Directory that mirrors what the alignment step produces.
    find_mesh_pairs() expects:
        <dir>/<name>_template_aligned.vtk
        <dir>/<name>_target_original.vtk
    """
    REAL_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Files are already written there by template_vtk / target_vtk fixtures.
    # If CLI paths were used and they're already VTKs in that location, copy them.
    dst_tmpl = REAL_INPUT_DIR / "patella_r_template_aligned.vtk"
    dst_tgt  = REAL_INPUT_DIR / "patella_r_target_original.vtk"
    if template_vtk != dst_tmpl:
        shutil.copy(template_vtk, dst_tmpl)
    if target_vtk != dst_tgt:
        shutil.copy(target_vtk, dst_tgt)
    return REAL_INPUT_DIR