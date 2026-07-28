"""
msk_morph/tests/test_pipeline_e2e.py
====================================

End-to-end test of the full pipeline on the shipped "test" participant.

The pipeline is run ONCE (module-scoped `pipeline_run` fixture, exactly as a
user runs it: `python msk_morph.py`), then three checks are made against its
outputs:

  1. markers.xml and markers.trc contain the same landmarks and values.
  2. The generated warping settings XML equals the template with the
     participant id substituted in.
  3. The markers.xml coordinates match a saved golden fixture
     (marker_fixtures.EXPECTED_MARKERS), regenerated with --update-golden.

Requires the full stack, so the whole module is marked `real`:

    pytest -m real                     # run these + the real registration test
    pytest -m real --update-golden     # (re)generate the marker golden, then skip
    pytest -m "not real"               # fast tiers, skips this

"""

import importlib.util
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.real

# ---------------------------------------------------------------------------
# Layout
#   This file lives in    <repo>/msk_morph/tests/
#   msk_morph.py lives in <repo>/msk_morph/
#   participant data in   <repo>/participant_data/   (i.e. ../ from the script)
# Adjust SCRIPT_DIR if your layout differs.
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent.parent       # <repo>/msk_morph
SCRIPT       = SCRIPT_DIR / "msk_morph.py"
PARTICIPANT  = "test"
PARTICIPANT_DIR = SCRIPT_DIR.parent / "participant_data" / PARTICIPANT

WARPING_XML_NAME = "SettingsModelWarper_StationDefinedTemplateModel_HipJoints.xml"
WARPING_TEMPLATE = SCRIPT_DIR / "template_model_and_settings" / WARPING_XML_NAME

MARKER_FIXTURES_MODULE = Path(__file__).resolve().parent / "fixtures" / "marker_fixtures.py"

# Wall-clock ceiling for the whole run. Tune to your mesh size / iteration count.
TIMEOUT_SECONDS = 4 * 60 * 60

# Absolute tolerance (metres) for the marker golden comparison.
MARKER_TOL = 1e-4

# Tight tolerance for xml-vs-trc: both are written from the same in-memory values
# formatted to 6 decimals.
XML_TRC_TOL = 1e-6


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def _parse_markerset(path: Path) -> dict:
    """markers.xml -> {marker_name: (x, y, z)} in metres."""
    root = ET.parse(str(path)).getroot()
    markers = {}
    for m in root.iter("Marker"):
        name = m.get("name")
        loc = m.find("location").text.split()
        markers[name] = tuple(float(v) for v in loc)
    return markers


def _parse_trc(path: Path) -> dict:
    """markers.trc -> {marker_name: (x, y, z)} from the first data frame."""
    lines = path.read_text().splitlines()

    # Line index 3 (4th line) holds marker names; first two columns are Frame#/Time.
    name_cells = lines[3].split("\t")
    names = [c.strip() for c in name_cells[2:] if c.strip()]

    # First data row = first line whose first tab-field is an integer frame number.
    data_row = None
    for ln in lines[4:]:
        if ln.split("\t")[0].strip().isdigit():
            data_row = ln
            break
    assert data_row is not None, f"No data frame found in {path}"

    values = [float(v) for v in data_row.split("\t")[2:] if v.strip()]
    assert len(values) == 3 * len(names), (
        f"TRC value count {len(values)} does not match 3 x {len(names)} markers"
    )
    coords = [tuple(values[i:i + 3]) for i in range(0, len(values), 3)]
    return dict(zip(names, coords))


# ---------------------------------------------------------------------------
# Marker golden I/O
# ---------------------------------------------------------------------------

def _load_marker_golden() -> dict:
    if not MARKER_FIXTURES_MODULE.exists():
        return {}
    spec = importlib.util.spec_from_file_location(
        "marker_fixtures", MARKER_FIXTURES_MODULE
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return dict(getattr(mod, "EXPECTED_MARKERS", {}))


def _write_marker_golden(markers: dict) -> None:
    lines = [
        '"""',
        "marker_fixtures.py",
        "==================",
        'Golden landmark coordinates for test_pipeline_e2e.py (participant "test").',
        "EXPECTED_MARKERS maps each marker name to its [x, y, z] location (metres).",
        "",
        "Regenerate after an intentional change:  pytest -m real --update-golden",
        '"""',
        "",
        "EXPECTED_MARKERS = {",
    ]
    for name, (x, y, z) in markers.items():
        lines.append(f"    {name!r}: [{x!r}, {y!r}, {z!r}],")
    lines.append("}")
    MARKER_FIXTURES_MODULE.parent.mkdir(parents=True, exist_ok=True)
    MARKER_FIXTURES_MODULE.write_text("\n".join(lines) + "\n")
    print(f"\n[update-golden] Wrote {len(markers)} markers -> {MARKER_FIXTURES_MODULE}")


# ---------------------------------------------------------------------------
# Run the whole pipeline once
# ---------------------------------------------------------------------------

def _clean_previous_run():
    """
    Remove generated outputs so no stage is served from a prior run and no
    stale file satisfies a check. CI is a fresh checkout (no-op); this matters
    on a developer machine. Only generated artifacts are removed, never input
    geometry.
    """
    temp_dir = SCRIPT_DIR / "temp"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    for p in (PARTICIPANT_DIR / "markers.xml", PARTICIPANT_DIR / "markers.trc"):
        if p.is_file():
            p.unlink()
    warping_dir = PARTICIPANT_DIR / "warping_files"
    if warping_dir.exists():
        shutil.rmtree(warping_dir)


@pytest.fixture(scope="module")
def pipeline_run():
    """Run `python msk_morph.py` once; assert it exits cleanly; yield the output dir."""
    assert SCRIPT.exists(), f"Cannot find pipeline script at {SCRIPT}"

    _clean_previous_run()

    result = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=str(SCRIPT_DIR),          # relative paths in msk_morph.py resolve from here
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
    )

    print("\n===== msk_morph.py STDOUT =====\n" + result.stdout)
    if result.stderr:
        print("\n===== msk_morph.py STDERR =====\n" + result.stderr)

    assert result.returncode == 0, (
        f"Pipeline exited with code {result.returncode}. See captured output above."
    )
    return PARTICIPANT_DIR


# ---------------------------------------------------------------------------
# 1. markers.xml and markers.trc agree
# ---------------------------------------------------------------------------

def test_markers_xml_trc_consistent(pipeline_run):
    xml = _parse_markerset(pipeline_run / "markers.xml")
    trc = _parse_trc(pipeline_run / "markers.trc")

    assert set(xml) == set(trc), (
        "markers.xml and markers.trc contain different landmarks:\n"
        f"  only in xml: {sorted(set(xml) - set(trc))}\n"
        f"  only in trc: {sorted(set(trc) - set(xml))}"
    )
    for name in xml:
        np.testing.assert_allclose(
            xml[name], trc[name], atol=XML_TRC_TOL, rtol=0,
            err_msg=f"markers.xml and markers.trc disagree for marker {name!r}",
        )


# ---------------------------------------------------------------------------
# 2. warping settings == template with participant id substituted
# ---------------------------------------------------------------------------

def test_warping_settings_matches_template(pipeline_run):
    generated = pipeline_run / "warping_files" / WARPING_XML_NAME
    assert WARPING_TEMPLATE.exists(), f"Template not found: {WARPING_TEMPLATE}"
    assert generated.exists(), f"Generated settings not found: {generated}"

    t_elems = list(ET.parse(str(WARPING_TEMPLATE)).getroot().iter())
    g_elems = list(ET.parse(str(generated)).getroot().iter())

    assert len(t_elems) == len(g_elems), (
        f"Element count differs: template has {len(t_elems)}, "
        f"generated has {len(g_elems)} - structure was not preserved."
    )

    substituted = False
    for t, g in zip(t_elems, g_elems):
        assert t.tag == g.tag, f"Tag mismatch: {t.tag!r} vs {g.tag!r}"
        assert set(t.attrib) == set(g.attrib), (
            f"Attribute keys differ on <{t.tag}>: {set(t.attrib)} vs {set(g.attrib)}"
        )

        t_text, g_text = (t.text or "").strip(), (g.text or "").strip()
        if t_text != g_text:
            assert PARTICIPANT in g_text, (
                f"<{t.tag}> text changed but not by inserting participant id "
                f"{PARTICIPANT!r}: template={t_text!r} generated={g_text!r}"
            )
            substituted = True

        for k in t.attrib:
            if t.attrib[k] != g.attrib[k]:
                assert PARTICIPANT in g.attrib[k], (
                    f"Attribute {k!r} on <{t.tag}> changed but not by inserting "
                    f"participant id {PARTICIPANT!r}: {t.attrib[k]!r} -> {g.attrib[k]!r}"
                )
                substituted = True

    assert substituted, (
        f"Generated settings are identical to the template - participant id "
        f"{PARTICIPANT!r} was never substituted in."
    )


# ---------------------------------------------------------------------------
# 3. marker coordinates match the golden fixture
# ---------------------------------------------------------------------------

def test_marker_values_match_golden(pipeline_run, request):
    markers = _parse_markerset(pipeline_run / "markers.xml")

    if request.config.getoption("--update-golden"):
        _write_marker_golden(markers)
        pytest.skip("Marker golden updated - re-run without --update-golden to verify.")

    expected = _load_marker_golden()
    if not expected:
        pytest.skip(
            "No marker golden yet - run:  pytest -m real --update-golden  to create it."
        )

    assert set(markers) == set(expected), (
        "Marker set differs from golden:\n"
        f"  new / unexpected: {sorted(set(markers) - set(expected))}\n"
        f"  missing:          {sorted(set(expected) - set(markers))}"
    )
    for name in expected:
        np.testing.assert_allclose(
            markers[name], expected[name], atol=MARKER_TOL, rtol=0,
            err_msg=(
                f"Marker {name!r} differs from golden beyond {MARKER_TOL} m. "
                f"If this is an intentional change, re-run with --update-golden."
            ),
        )