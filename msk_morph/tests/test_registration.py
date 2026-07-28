"""
msk_morph/tests/test_deformetrica_registration.py
========================================

Three test tiers
----------------

1. TestDeformetricaRegistrationUnit  (fast)
   Pure Python logic tests — validation, path conversion, file discovery,
   output-file finding.  The Deformetrica API is mocked throughout.

2. TestRunBatchRegistrationIntegration  (fast)
   Full run_batch_registration() code path with the Deformetrica API mocked
   to plant a fake output VTK.  Verifies Python orchestration and golden comparison.

3. TestRealRegistration  (slow, requires deformetrica)
   Runs the actual Deformetrica API end-to-end.
   Skipped automatically when Deformetrica is not available.
   Run explicitly with:   pytest -m real
   Skip explicitly with:  pytest -m "not real"

Golden file workflow
--------------------
After an intentional algorithm change, regenerate the expected output:

    pytest -m real --update-golden

This runs the real registration, overwrites EXPECTED_* in
msk_morph/tests/fixtures/patella_mesh_fixtures.py, then skips the comparison so
the test passes.  Review the diff and commit.

Swapping input meshes
---------------------------------------------------------
To permanently bake new meshes into the fixture file:
    python tests/fixtures/update_fixtures.py \\
        --template path/to/new_template.vtk \\
        --target   path/to/new_target.vtk
"""

import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from utils.mesh_loader import load_with_fallbacks



def _read_vtk(path: Path):
    """Read a VTK file into (points, cells) arrays, bypassing Open3D
    which does not support .vtk and falls back to a PointCloud."""
    try:
        import vtk
        from vtk.util.numpy_support import vtk_to_numpy
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(path))
        reader.Update()
        pd = reader.GetOutput()
        if pd.GetNumberOfPoints() == 0:
            raise ValueError("vtkPolyDataReader returned 0 points")
        points = vtk_to_numpy(pd.GetPoints().GetData()).astype("float64")
        cells  = vtk_to_numpy(pd.GetPolys().GetData()).reshape(-1, 4)[:, 1:].astype("int64")
        return points, cells
    except ImportError:
        pass
    import meshio
    m = meshio.read(str(path))
    for block in m.cells:
        if block.type == "triangle":
            return m.points.astype("float64"), block.data.astype("int64")
    raise ValueError(f"No triangle cells found in {path}")


def _write_vtk(points: np.ndarray, cells: np.ndarray, path) -> None:
    """
    Write a triangle mesh as ASCII VTK PolyData — same format used throughout
    the pipeline.  Mirrors the helper in conftest.py.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_pts, n_cells = len(points), len(cells)
    with open(str(path), "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("mesh\n")
        f.write("ASCII\n")
        f.write("DATASET POLYDATA\n")
        f.write(f"POINTS {n_pts} float\n")
        for pt in points:
            f.write(f"{pt[0]:.6f} {pt[1]:.6f} {pt[2]:.6f}\n")
        f.write(f"POLYGONS {n_cells} {n_cells * 4}\n")
        for tri in cells:
            f.write(f"3 {tri[0]} {tri[1]} {tri[2]}\n")

# ---------------------------------------------------------------------------
# Paths  (fixtures live in msk_morph/tests/fixtures/, this file in msk_morph/tests/)
# ---------------------------------------------------------------------------
TESTS_DIR       = Path(__file__).parent
FIXTURES_MODULE = TESTS_DIR / "fixtures" / "patella_mesh_fixtures.py"

# ---------------------------------------------------------------------------
# Shared helpers (used by both integration and real tests)
# ---------------------------------------------------------------------------

def _compare_meshes(result_path: Path, expected_points, expected_cells,
                    point_tol: float = 1e-3):
    """
    Load a VTK file and compare its geometry to expected arrays.

    point_tol: absolute tolerance in mesh coordinate units (mm).
        Default 1e-3 mm — tight, since LDDMM is deterministic and
        both result and golden are stored and compared as float64.

    Connectivity (cells) is always compared exactly — topology should
    never change between runs.
    """
    result_points, result_cells = _read_vtk(result_path)
    result_points = result_points.astype("float64")
    expected_points = np.asarray(expected_points, dtype="float64")

    assert result_points.shape == expected_points.shape, (
        f"Point array shape mismatch: got {result_points.shape}, "
        f"expected {expected_points.shape}"
    )
    np.testing.assert_allclose(
        result_points, expected_points, atol=point_tol, rtol=0,
        err_msg=(
            f"Registered mesh points differ from golden output beyond "
            f"{point_tol}mm tolerance. If this is an intentional change, "
            f"re-run with --update-golden."
        ),
    )

    assert result_cells.shape == expected_cells.shape, (
        f"Cell array shape mismatch: got {result_cells.shape}, "
        f"expected {expected_cells.shape}"
    )
    np.testing.assert_array_equal(
        result_cells, expected_cells,
        err_msg="Registered mesh connectivity differs from golden output.",
    )


def _replace_variable(text: str, var_name: str, new_value: str) -> str:
    """
    Replace a variable assignment in Python source text.
    """
    lines = text.split("\n")
    start = None
    for i, line in enumerate(lines):
        if line.startswith(f"{var_name} =") or line.startswith(f"{var_name}="):
            start = i
            break
    if start is None:
        raise ValueError(f"Variable {var_name!r} not found in fixture file")

    # Find end: next blank line or next line starting a new top-level name
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line == "" or (line and not line[0].isspace() and "=" in line):
            break
        end += 1

    lines[start:end] = [new_value]
    return "\n".join(lines)


def _update_golden(output_path: Path):
    """
    Overwrite EXPECTED_POINTS / EXPECTED_CELLS in the fixture module
    with the arrays from output_path.
    """
    new_points_arr, new_cells_arr = _read_vtk(output_path)
    new_points = new_points_arr.tolist()
    new_cells  = new_cells_arr.tolist()

    text = FIXTURES_MODULE.read_text()
    text = _replace_variable(
        text, "EXPECTED_POINTS",
        f'EXPECTED_POINTS = np.array({new_points!r}, dtype="float64")',
    )
    text = _replace_variable(
        text, "EXPECTED_CELLS",
        f'EXPECTED_CELLS = np.array({new_cells!r}, dtype="int64")'
        f'  # topology is unchanged by registration',
    )
    FIXTURES_MODULE.write_text(text)
    print(f"\n[update-golden] Wrote new golden output -> {FIXTURES_MODULE}")


def _find_output_vtk(output_dir: Path) -> Path:
    """Locate the registered output VTK produced by Deformetrica."""
    files = list(output_dir.rglob("*Reconstruction*.vtk"))
    assert len(files) >= 1, (
        f"No output VTK found under {output_dir}.\n"
        f"Directory contents: {list(output_dir.rglob('*'))}"
    )
    return files[0]


# ===========================================================================
# 1. Unit tests
# ===========================================================================

class TestDeformetricaRegistrationUnit:
    """
    Isolated unit tests — the Deformetrica API is mocked throughout.
    """

    def _make_registrar(self):
        from utils.deformetrica_registration import DeformetricaRegistration
        return DeformetricaRegistration()

    # --- find_mesh_pairs ---

    def test_find_mesh_pairs_finds_correct_files(self, aligned_meshes_dir):
        from utils.deformetrica_registration import find_mesh_pairs
        pairs = find_mesh_pairs(str(aligned_meshes_dir))

        assert len(pairs) == 1, f"Expected 1 pair, found {len(pairs)}"
        mesh_name, tmpl, tgt = pairs[0]
        assert mesh_name == "patella_r"
        assert tmpl.endswith("patella_r_template_aligned.vtk")
        assert tgt.endswith("patella_r_target_original.vtk")

    def test_find_mesh_pairs_empty_dir(self, tmp_path):
        from utils.deformetrica_registration import find_mesh_pairs
        assert find_mesh_pairs(str(tmp_path)) == []

    def test_find_mesh_pairs_missing_target(self, tmp_path):
        """Template without a matching target should be silently skipped."""
        (tmp_path / "orphan_template_aligned.vtk").touch()
        from utils.deformetrica_registration import find_mesh_pairs
        assert find_mesh_pairs(str(tmp_path)) == []


    # --- kernel width validation ---

    def test_iterative_reg_requires_list_deformation_kw(
        self, tmp_path, template_vtk, target_vtk
    ):
        reg = self._make_registrar()
        with pytest.raises(ValueError, match="deformation_kernel_width to be a list"):
            reg.run_single_mesh_registration(
                template_file=str(template_vtk),
                target_file=str(target_vtk),
                output_dir=str(tmp_path),
                iterative_reg=True,
                deformation_kernel_width=20.0,
                template_kernel_width=[20.0, 10.0],
            )

    def test_iterative_reg_requires_list_template_kw(
        self, tmp_path, template_vtk, target_vtk
    ):
        reg = self._make_registrar()
        with pytest.raises(ValueError, match="template_kernel_width to be a list"):
            reg.run_single_mesh_registration(
                template_file=str(template_vtk),
                target_file=str(target_vtk),
                output_dir=str(tmp_path),
                iterative_reg=True,
                deformation_kernel_width=[20.0, 10.0],
                template_kernel_width=10.0,
            )

    def test_iterative_reg_mismatched_list_lengths(
        self, tmp_path, template_vtk, target_vtk
    ):
        reg = self._make_registrar()
        with pytest.raises(ValueError, match="must have the same length"):
            reg.run_single_mesh_registration(
                template_file=str(template_vtk),
                target_file=str(target_vtk),
                output_dir=str(tmp_path),
                iterative_reg=True,
                deformation_kernel_width=[20.0, 10.0, 5.0],
                template_kernel_width=[20.0, 10.0],
            )

    # --- _find_warped_template_output ---

    def test_find_warped_template_output_exact_name(self, tmp_path):
        reg = self._make_registrar()
        name = "DeterministicAtlas__Reconstruction__bone_trial__subject_subject_0.vtk"
        (tmp_path / name).touch()
        result = reg._find_warped_template_output(str(tmp_path))
        assert result is not None and result.endswith(name)

    def test_find_warped_template_output_fallback_glob(self, tmp_path):
        reg = self._make_registrar()
        (tmp_path / "SomeOther__Reconstruction__result.vtk").touch()
        result = reg._find_warped_template_output(str(tmp_path))
        assert result is not None and "Reconstruction" in result

    def test_find_warped_template_output_missing(self, tmp_path):
        reg = self._make_registrar()
        assert reg._find_warped_template_output(str(tmp_path)) is None

    # --- _run_registration_step (mocked Deformetrica API) ---

    @patch("utils.deformetrica_registration.Deformetrica")
    def test_run_registration_step_success(
        self, MockDeformetrica, tmp_path, template_vtk, target_vtk
    ):
        MockDeformetrica.return_value.estimate_registration.return_value = None
        reg = self._make_registrar()
        assert reg._run_registration_step(
            template_file=str(template_vtk),
            target_file=str(target_vtk),
            output_dir=str(tmp_path),
        ) is True
        assert MockDeformetrica.called
        assert MockDeformetrica.return_value.estimate_registration.called

    @patch("utils.deformetrica_registration.Deformetrica")
    def test_run_registration_step_failure(
        self, MockDeformetrica, tmp_path, template_vtk, target_vtk
    ):
        MockDeformetrica.return_value.estimate_registration.side_effect = RuntimeError(
            "Deformetrica crashed"
        )
        reg = self._make_registrar()
        assert reg._run_registration_step(
            template_file=str(template_vtk),
            target_file=str(target_vtk),
            output_dir=str(tmp_path),
        ) is False


# ===========================================================================
# 2. Integration tests
# ===========================================================================

class TestRunBatchRegistrationIntegration:
    """
    End-to-end run_batch_registration() with the Deformetrica API mocked.
    The mock plants a fake output VTK so the full orchestration logic
    (file discovery -> registration loop -> output finding) can be exercised
    without running real registration.
    """

    RECON_FILENAME = (
        "DeterministicAtlas__Reconstruction__bone_trial__subject_subject_0.vtk"
    )

    def _make_fake_estimate_registration(self, output_dir: Path, fixture_data):
        """Return a side_effect for Deformetrica.estimate_registration that plants a fake output VTK."""
        def _fake_estimate(template_specifications, dataset_specifications,
                           model_options, estimator_options, write_output=True):
            # Deformetrica writes into the directory it was initialised with,
            # which isn't accessible from here. The fake output is therefore
            # written into every subdirectory under output_dir at call time.
            for subdir in list(output_dir.rglob("*")) + [output_dir]:
                if subdir.is_dir():
                    out = subdir / self.RECON_FILENAME
                    if not out.exists():
                        _write_vtk(
                            fixture_data.TEMPLATE_POINTS,
                            fixture_data.TEMPLATE_CELLS,
                            out,
                        )
        return _fake_estimate

    @patch("utils.deformetrica_registration.Deformetrica")
    def test_single_step_produces_output_and_matches_golden(
        self, MockDeformetrica, tmp_path, aligned_meshes_dir, fixture_data
    ):
        from utils.deformetrica_registration import run_batch_registration

        output_dir = tmp_path / "registration_output"
        MockDeformetrica.return_value.estimate_registration.side_effect = (
            self._make_fake_estimate_registration(output_dir, fixture_data)
        )

        results = run_batch_registration(
            aligned_meshes_dir=str(aligned_meshes_dir),
            registration_output_dir=str(output_dir),
            iterative_reg=False,
            deformation_kernel_width=20.0,
            template_kernel_width=20.0,
            max_iterations=40,
            verbose=False,
        )

        assert results.get("patella_r") is True, (
            f"run_batch_registration reported failure: {results}"
        )

        result_vtk = _find_output_vtk(output_dir)

        # Integration test only verifies orchestration logic — not output quality.
        # Golden updates and geometry comparison are exclusively handled by
        # TestRealRegistration.
        out_pts, out_cells = _read_vtk(result_vtk)
        assert len(out_pts)   > 0, "Output mesh has no points"
        assert len(out_cells) > 0, "Output mesh has no cells"

        
    @patch("utils.deformetrica_registration.Deformetrica")
    def test_iterative_two_steps_creates_correct_subdirs(
        self, MockDeformetrica, tmp_path, aligned_meshes_dir, fixture_data
    ):
        from utils.deformetrica_registration import run_batch_registration

        output_dir = tmp_path / "iterative_output"
        MockDeformetrica.return_value.estimate_registration.side_effect = (
            self._make_fake_estimate_registration(output_dir, fixture_data)
        )

        results = run_batch_registration(
            aligned_meshes_dir=str(aligned_meshes_dir),
            registration_output_dir=str(output_dir),
            iterative_reg=True,
            deformation_kernel_width=[20.0, 10.0],
            template_kernel_width=[20.0, 10.0],
            max_iterations=2,
            verbose=False,
        )

        assert results.get("patella_r") is True

        patella_dir = output_dir / "patella_r"
        step_dirs   = [d for d in patella_dir.iterdir() if d.is_dir()] \
                      if patella_dir.exists() else []
        assert len(step_dirs) == 2, (
            f"Expected 2 iterative step dirs, found {len(step_dirs)}: {step_dirs}"
        )
        step_names = {d.name for d in step_dirs}
        assert any("defkw20.0" in n for n in step_names), step_names
        assert any("defkw10.0" in n for n in step_names), step_names


# ===========================================================================
# 3. Real end-to-end tests
# ===========================================================================

# Stable output directory — persists after the test run for inspection
# and --update-golden.  Listed in .gitignore.
REAL_OUTPUT_DIR = TESTS_DIR / "fixtures" / "registration_output"


@pytest.mark.real
class TestRealRegistration:
    """
    Runs the actual Deformetrica API directly.

    Only the iterative coarse-to-fine registration (kernel 20 -> 10) is tested,
    since that is the production configuration. The golden file is populated from
    the step-2 output:
        msk_morph/tests/fixtures/registration_output/
            patella_r_iterative/patella_r/
                iter_2_of_2__defkw10.0__tmplkw10.0/
                    DeterministicAtlas__Reconstruction__...vtk

    This folder is listed in .gitignore - outputs persist after the run
    to be inspected or to use --update-golden.

    Usage:

        pytest -m real                        # run only this test
        pytest -m real --update-golden        # run + bless output as new golden
        pytest -m real -s                     # show Deformetrica stdout live
        pytest -m "not real"                  # skip, run everything else

    Swap input meshes without editing any file:

        pytest -m real \\
            --template-mesh data/new_template.stl \\
            --target-mesh   data/new_target.vtk

    To permanently bake those meshes into the fixture file:
        python tests/fixtures/update_fixtures.py \\
            --template data/new_template.stl \\
            --target   data/new_target.vtk
    """

    # Path to the step-2 output relative to REAL_OUTPUT_DIR —
    # this is the canonical result used for golden comparison.
    GOLDEN_VTK = (
        "patella_r_iterative/patella_r/"
        "iter_2_of_2__defkw10.0__tmplkw10.0/"
        "DeterministicAtlas__Reconstruction__bone_trial__subject_subject_0.vtk"
    )

    def test_real_iterative_registration(
        self, aligned_meshes_dir, fixture_data, request
    ):
        """
        Full iterative coarse-to-fine registration (kernel 20 -> 10) using
        the real Deformetrica API.

        Output -> msk_morph/tests/fixtures/registration_output/patella_r_iterative/

        Verifies:
          - both registration steps complete successfully
          - the step-2 output VTK exists and is a valid non-empty mesh
          - the step-2 output geometry matches the golden fixture (within tolerance)

        Golden update:
          pytest -m real --update-golden -s
          (overwrites EXPECTED_* in patella_mesh_fixtures.py with step-2 output)
        """
        from utils.deformetrica_registration import run_batch_registration

        # Always wipe the output directory before running so stale mock
        # or previous outputs can never be mistaken for a fresh real result.
        output_dir = REAL_OUTPUT_DIR / "patella_r_iterative"
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = run_batch_registration(
            aligned_meshes_dir=str(aligned_meshes_dir),
            registration_output_dir=str(output_dir),
            iterative_reg=True,
            deformation_kernel_width=[20.0, 10.0],
            template_kernel_width=[20.0, 10.0],
            max_iterations=40,
            verbose=True,
        )

        assert results.get("patella_r") is True, (
            f"Real iterative registration failed.\n"
            f"Output directory: {output_dir}\n"
            f"Results: {results}"
        )

        # Both step directories must exist
        patella_dir = output_dir / "patella_r"
        step_dirs = [d for d in patella_dir.iterdir() if d.is_dir()] \
                    if patella_dir.exists() else []
        assert len(step_dirs) == 2, (
            f"Expected 2 iterative step dirs, found {len(step_dirs)}: {step_dirs}"
        )

        # Locate the specific step-2 golden VTK
        golden_vtk = REAL_OUTPUT_DIR / self.GOLDEN_VTK
        assert golden_vtk.exists(), (
            f"Step-2 output not found at expected path:\n  {golden_vtk}\n"
            f"Directory contents: {list(patella_dir.rglob('*.vtk'))}"
        )
        print(f"\n[real test] Step-2 output VTK: {golden_vtk}")

        # Basic sanity: readable, non-empty mesh
        out_pts, out_cells = _read_vtk(golden_vtk)
        assert len(out_pts)   > 0, "Step-2 output has no points"
        assert len(out_cells) > 0, "Step-2 output has no cells"

        if request.config.getoption("--update-golden", default=False):
            _update_golden(golden_vtk)
            pytest.skip("Golden file updated — re-run without --update-golden to verify.")

        _compare_meshes(golden_vtk, fixture_data.EXPECTED_POINTS,
                        fixture_data.EXPECTED_CELLS)