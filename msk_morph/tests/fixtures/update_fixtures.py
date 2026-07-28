"""
msk_morph/tests/fixtures/update_fixtures.py
===========================================
Bake new template and/or target MESH data permanently into
msk_morph/tests/fixtures/patella_mesh_fixtures.py.

Scope
-----
This script updates only the mesh fixtures (TEMPLATE_*, TARGET_*, and the
EXPECTED_* mesh golden) in patella_mesh_fixtures.py. It reads the meshes
passed on the command line and writes their point/cell arrays into that file.

It does NOT update the marker golden (marker_fixtures.py / EXPECTED_MARKERS).
Markers are an output of running the full pipeline, not an input file, so they
are regenerated separately:
    pytest -m real --update-golden      (see test_pipeline_e2e.py)

Usage (run from msk_morph directory)
-----
# Replace both meshes:
python tests/fixtures/update_fixtures.py \\
    --template path/to/new_template.stl \\
    --target   path/to/new_target.vtk

# Replace only the template:
python tests/fixtures/update_fixtures.py --template path/to/new_template.stl

# Replace only the target:
python tests/fixtures/update_fixtures.py --target path/to/new_target.vtk

After running, review the diff and commit:
    git diff tests/fixtures/patella_mesh_fixtures.py
    git add tests/fixtures/patella_mesh_fixtures.py
    git commit -m "fixtures: update template/target mesh"

Notes
-----
- EXPECTED_POINTS is reset to zeros (placeholder) whenever the template
  or target mesh changes, because the golden output will be different.
  Re-run  pytest -m real --update-golden  to regenerate it.
- The script accepts meshio-readable format: .stl, .vtk, .obj, etc.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

# This script is always run from the repo root, so use cwd to locate utils.
_REPO_ROOT = Path.cwd()
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.mesh_loader import load_with_fallbacks

REPO_ROOT       = _REPO_ROOT
FIXTURES_MODULE = Path(__file__).resolve().parent / "patella_mesh_fixtures.py"


def read_mesh(path: Path):
    """
    Return (points_list, cells_list) from a VTK or STL mesh file.

    For VTK files: uses the vtk library or meshio directly, since Open3D
    does not support the .vtk extension.
    For other formats (STL, OBJ, PLY …): uses load_with_fallbacks which
    goes through Open3D first.
    """
    suffix = path.suffix.lower()

    if suffix == ".vtk":
        # Open3D does not handle .vtk - go straight to vtk library or meshio.
        try:
            import vtk
            from vtk.util.numpy_support import vtk_to_numpy

            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(str(path))
            reader.Update()
            pd = reader.GetOutput()

            if pd.GetNumberOfPoints() == 0:
                raise ValueError("vtkPolyDataReader returned 0 points")

            points = vtk_to_numpy(pd.GetPoints().GetData()).astype("float32").tolist()
            cells  = vtk_to_numpy(pd.GetPolys().GetData()).reshape(-1, 4)[:, 1:].astype("int64").tolist()
            return points, cells

        except ImportError:
            pass  # vtk not installed, fall through to meshio

        try:
            import meshio
            m = meshio.read(str(path))
            for cell_block in m.cells:
                if cell_block.type == "triangle":
                    return m.points.tolist(), cell_block.data.tolist()
            raise ValueError(f"No triangle cells found in {path}")
        except ImportError:
            raise ImportError(
                "Neither the 'vtk' nor 'meshio' library is installed. "
                "Install one of them to read VTK files:\n"
                "  pip install vtk   or   pip install meshio"
            )

    else:
        # Non-VTK formats: use load_with_fallbacks (Open3D handles STL, PLY, OBJ)
        mesh = load_with_fallbacks(str(path), verbose=False)

        try:
            import open3d as o3d
            if isinstance(mesh, o3d.geometry.TriangleMesh):
                return (
                    np.asarray(mesh.vertices,  dtype="float32").tolist(),
                    np.asarray(mesh.triangles, dtype="int64").tolist(),
                )
        except ImportError:
            pass

        # meshio fallback (when open3d is absent)
        if hasattr(mesh, "points") and hasattr(mesh, "cells"):
            return mesh.points.tolist(), mesh.cells[0].data.tolist()

        raise ValueError(
            f"Could not extract triangle mesh from {path} (type: {type(mesh)}). "
            "Connectivity data is required."
        )


def update_block(text: str, var_points: str, var_cells: str,
                 new_points: list, new_cells: list,
                 dtype_points: str = "float32",
                 dtype_cells:  str = "int64") -> str:
    """Replace a POINTS/CELLS variable pair in the fixture source text."""
    text = re.sub(
        rf"{var_points}\s*=.*?(?=\n\n|\n#)",
        f'{var_points} = np.array({new_points!r}, dtype="{dtype_points}")',
        text, flags=re.DOTALL,
    )
    text = re.sub(
        rf"{var_cells}\s*=.*?(?=\n\n|\n#)",
        f'{var_cells} = np.array({new_cells!r}, dtype="{dtype_cells}")',
        text, flags=re.DOTALL,
    )
    return text


def reset_expected(text: str, n_points: int, cells: list) -> str:
    """Reset EXPECTED_POINTS to zeros and EXPECTED_CELLS to the new topology.
    """
    text, n1 = re.subn(
        r"EXPECTED_POINTS\s*=.*?(?=\n\n|\n#)",
        f'EXPECTED_POINTS = np.zeros(({n_points}, 3), dtype="float32")  # regenerate!',
        text, flags=re.DOTALL,
    )
    text, n2 = re.subn(
        r"EXPECTED_CELLS\s*=.*?# topology is unchanged by registration",
        f'EXPECTED_CELLS = np.array({cells!r}, dtype="int64")'
        f'  # topology is unchanged by registration',
        text, flags=re.DOTALL,
    )
    if (n1, n2) != (1, 1):
        raise RuntimeError(
            f"reset_expected matched EXPECTED_POINTS={n1}, EXPECTED_CELLS={n2} "
            f"(expected 1 each). Fixture layout changed — update the regexes."
        )
    return text


def generate_fixture_file(template_pts, template_cells,
                          target_pts,   target_cells) -> str:
    """
    Produce the full source text for patella_mesh_fixtures.py from scratch.
    EXPECTED_* are left as zeros/placeholder - run pytest -m real --update-golden
    to populate them after the first successful registration.
    """
    n_pts = len(template_pts)
    lines = [
        '"""',
        'Mesh fixture data for deformetrica_registration tests.',
        '',
        'Contains:',
        '  TEMPLATE_POINTS / TEMPLATE_CELLS  - template mesh',
        '  TARGET_POINTS   / TARGET_CELLS    - target mesh',
        '  EXPECTED_POINTS / EXPECTED_CELLS  - expected registered output (golden file)',
        '',
        'How to regenerate the EXPECTED_* arrays after an intentional algorithm change:',
        '  pytest -m real --update-golden',
        '  Then inspect the diff and commit.',
        '"""',
        'import numpy as np',
        '',
        '',
        '# ---------------------------------------------------------------------------',
        '# Template mesh',
        '# ---------------------------------------------------------------------------',
        f'TEMPLATE_POINTS = np.array({template_pts!r}, dtype="float32")',
        '',
        f'TEMPLATE_CELLS = np.array({template_cells!r}, dtype="int64")',
        '',
        '',
        '# ---------------------------------------------------------------------------',
        '# Target mesh',
        '# ---------------------------------------------------------------------------',
        f'TARGET_POINTS = np.array({target_pts!r}, dtype="float32")',
        '',
        f'TARGET_CELLS = np.array({target_cells!r}, dtype="int64")',
        '',
        '',
        '# ---------------------------------------------------------------------------',
        '# Expected registered output mesh (golden file)',
        '# PLACEHOLDER - run:  pytest -m real --update-golden  to populate.',
        '# ---------------------------------------------------------------------------',
        f'EXPECTED_POINTS = np.zeros(({n_pts}, 3), dtype="float32")  # regenerate!',
        '',
        f'EXPECTED_CELLS = np.array({template_cells!r}, dtype="int64")'
        f'  # topology is unchanged by registration',
    ]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", metavar="PATH",
                        help="Template mesh file (.stl, .vtk, …)")
    parser.add_argument("--target",   metavar="PATH",
                        help="Target mesh file (.stl, .vtk, …)")
    args = parser.parse_args()

    if not args.template and not args.target:
        parser.error("Provide at least --template or --target (or both).")

    # -----------------------------------------------------------------------
    # Case 1: fixture file does not exist yet - generate it from scratch.
    # Both --template and --target are required in this case.
    # -----------------------------------------------------------------------
    if not FIXTURES_MODULE.exists():
        if not args.template or not args.target:
            sys.exit(
                "patella_mesh_fixtures.py does not exist yet.\n"
                "Provide both --template and --target to create it from scratch."
            )

        tmpl_src = Path(args.template)
        tgt_src  = Path(args.target)
        if not tmpl_src.exists():
            sys.exit(f"Template file not found: {tmpl_src}")
        if not tgt_src.exists():
            sys.exit(f"Target file not found: {tgt_src}")

        tmpl_pts, tmpl_cells = read_mesh(tmpl_src)
        tgt_pts,  tgt_cells  = read_mesh(tgt_src)

        FIXTURES_MODULE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURES_MODULE.write_text(
            generate_fixture_file(tmpl_pts, tmpl_cells, tgt_pts, tgt_cells)
        )
        print(f"Created {FIXTURES_MODULE}")
        print(f"  Template: {tmpl_src}  ({len(tmpl_pts)} pts, {len(tmpl_cells)} faces)")
        print(f"  Target:   {tgt_src}   ({len(tgt_pts)} pts, {len(tgt_cells)} faces)")
        print("  EXPECTED_POINTS set to zeros - run  pytest -m real --update-golden  to populate.")
        print(f"\nDone. Commit the new file:\n  git add {FIXTURES_MODULE.relative_to(REPO_ROOT)}")
        return

    # -----------------------------------------------------------------------
    # Case 2: fixture file already exists - update only what was passed.
    # -----------------------------------------------------------------------
    text = FIXTURES_MODULE.read_text()
    pts, cells = None, None   # track last-updated mesh for golden reset

    if args.template:
        src = Path(args.template)
        if not src.exists():
            sys.exit(f"Template file not found: {src}")
        pts, cells = read_mesh(src)
        text = update_block(text, "TEMPLATE_POINTS", "TEMPLATE_CELLS", pts, cells)
        print(f"Updated TEMPLATE mesh from: {src}  ({len(pts)} pts, {len(cells)} faces)")

    if args.target:
        src = Path(args.target)
        if not src.exists():
            sys.exit(f"Target file not found: {src}")
        pts, cells = read_mesh(src)
        text = update_block(text, "TARGET_POINTS", "TARGET_CELLS", pts, cells)
        print(f"Updated TARGET mesh from: {src}  ({len(pts)} pts, {len(cells)} faces)")

    # Reset golden output - it's invalid whenever inputs change
    text = reset_expected(text, len(pts), cells)
    print("Reset EXPECTED_POINTS to zeros - run  pytest -m real --update-golden  to regenerate.")

    FIXTURES_MODULE.write_text(text)
    print(f"\nDone. Review the diff:\n  git diff {FIXTURES_MODULE.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()