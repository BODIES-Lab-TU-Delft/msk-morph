"""
Standalone script to pose morphed LD-MSK models onto the original pose from the medical imaging data
It uses OpenSim's API to re-pose the model using the stations

Usage (through Opensim's Python API):
    python posing_morphed_model_onto_CSscanner.py

Configuration:
    Edit the PARTICIPANT_IDS variable below

Input (per participant, in .\<participant_id>\):
    - StationDefinedMorphedModel_HipJoints_rel-path.osim
    - markers.xml
    - markers.trc

Output (per participant, in .\<participant_id>\):
    - StationDefinedMorphedModel_with_markers_posed.osim
"""

import opensim as osim
import os

PARTICIPANT_IDS = ["test"]  # List of participant IDs to process in batch mode ["001","002","003"]

def process_participant(participant_id):
    print(f"\n{'='*50}")
    print(f"Processing participant: {participant_id}")
    print(f"{'='*50}")

    # Build paths relative to participant folder
    base_dir = os.path.join(".", participant_id)

    model_file = os.path.join(base_dir, "StationDefinedMorphedModel_HipJoints_rel-path.osim")
    markers_xml_file = os.path.join(base_dir, "markers.xml")
    markers_trc_file = os.path.join(base_dir, "markers.trc")
    output_file = os.path.join(base_dir, "StationDefinedMorphedModel_with_markers_posed.osim")

    # Validate input files exist before processing
    for f in [model_file, markers_xml_file, markers_trc_file]:
        if not os.path.exists(f):
            print(f"  [ERROR] Missing file: {f} — skipping participant.")
            return False

    # Load model
    model = osim.Model(model_file)

    # Load MarkerSet from file and append to model
    marker_set = osim.MarkerSet(markers_xml_file)
    model_marker_set = model.updMarkerSet()
    for i in range(marker_set.getSize()):
        model_marker_set.cloneAndAppend(marker_set.get(i))

    # Finalize and initialize
    model.finalizeConnections()
    model.initSystem()

    # Load experimental measurements into a `MarkersReference`
    markers_reference = osim.MarkersReference(markers_trc_file, osim.SetMarkerWeights())
    ik_solver = osim.InverseKinematicsSolver(model, markers_reference, osim.SimTKArrayCoordinateReference())
    ik_solver.setAccuracy(1e-10)
    state = model.updWorkingState()
    ik_solver.assemble(state)
    ik_solver.track(state)

    errors = osim.SimTKArrayDouble()
    ik_solver.computeCurrentSquaredMarkerErrors(errors)
    for i in range(errors.size()):
        print(f"  {i} -> {ik_solver.getMarkerNameForIndex(i)} -> {errors.getElt(i)}")

    for coordinate in model.getCoordinateSet():
        coordinate.set_default_value(coordinate.getValue(state))

    model.printToXML(output_file)
    print(f"  [OK] Output saved to: {output_file}")
    return True


# --- Batch runner ---
if __name__ == "__main__":
    results = {"success": [], "failed": []}

    for pid in PARTICIPANT_IDS:
        try:
            ok = process_participant(pid)
            (results["success"] if ok else results["failed"]).append(pid)
        except Exception as e:
            print(f"  [ERROR] Participant {pid} raised an exception: {e}")
            results["failed"].append(pid)

    print(f"\n{'='*50}")
    print(f"Batch complete.")
    print(f"  Succeeded: {results['success']}")
    print(f"  Failed:    {results['failed']}")