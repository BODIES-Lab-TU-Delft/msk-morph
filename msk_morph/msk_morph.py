#!/usr/bin/env python3
"""
Framework to morph landmark-defined musculoskeletal models into subject-specific bone geometries.
Complete pipeline for processing 3D bone meshes into OpenSim-ready landmark-defined musculoskeletal models.
Supports single participant or batch processing of multiple participants.

Author: Judit Cueto Fernandez
"""

import sys
import os
from pathlib import Path

from utils.mesh_converter import convert_stl_to_vtk_batch

from utils.mesh_processing_utils import(
    check_vtk_files, 
    check_stl_files,
    check_obj_files,
    get_mesh_files
)

from utils.mesh_alignment import (
    anatomical_mesh_alignment_workflow,
    create_axes_definition
)

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Participant configuration - MODIFY THIS FOR BATCH PROCESSING
PARTICIPANT_IDS = ["test"]  # List of participant IDs to process in batch mode ["001","002","003"]
# Run on "test" to test pipeline on test data

# Directory configuration
TEMPLATE_GEOM_DIR = "./template_model_and_settings/template_geometry" # Path to the template bone geometries of the LD-MSK model
TEMPLATE_REGISTRATION_DIR = "./template_mesh_registration" # Directory where the generated vtk versions of the template LD-MSK geometries are saved (expected to be empty on first run of a new LD-MSK model; populated automatically during execution)
TEMPLATE_WARPING_DIR = "./template_warping_files" # Directory where the generated landmark files (CSV) are saved (expected to be empty on first run of a new LD-MSK model; populated automatically during execution)

# MSK-Morph settings file path
SETTINGS_FILE_PATH = "./template_model_and_settings/MSK-Morph_settings.yaml" # Path to the MSK-Morph settings file defining the morphing and landmark identification rules

# ModelWarper settings file path
TEMPLATE_WARPING_SET_FILENAME = "./template_model_and_settings/SettingsModelWarper_StationDefinedTemplateModel_HipJoints.xml" # Path to the template LD-MSK model morphing settings file (for OpenSim Creator's ModelWarper)

PARTICIPANT_BASE_DIR = "../participant_data" # Path to the base directory containing your participant dataset
TEMP_BASE_DIR = "./temp" # Path to the directory where temporary files are saved (must be deleted manually for the files to be overwritten)

# Mesh processing parameters - Adjust according to template and target bone geometries
TEMPLATE_SCALE_FACTOR = 1000.0  # m to mm for template meshes
TARGET_SCALE_FACTOR = 1.0       # Target meshes already in mm, no scaling needed
POSTPROCESS_SCALE_FACTOR = 0.001  # mm to m for final output

REGISTRATION_PARAMS = {          # Settings for the mesh registration step to be fed to Deformetrica
    'downsample_meshes': False,             # downsample_meshes: set to True to downsample the target meshes to speed the processing
                                            # Default value is 50% downsampling (0.5)
    #'downsample_reduction_factor': 0.9,    # To select a different value, add the argument 'downsample_reduction_factor' and set your desired downsampling factor
    'max_iterations': 40,
    'conda_env': 'deformetrica',
    'verbose': True,
    'iterative_reg': True,                      #   iterative_reg: Set to True to enable coarse-to-fine iterative registration
                                                #   When True, deformation_kernel_width and template_kernel_width must be equal-length lists
                                                #   Each element defines one registration stage
                                                #   When False, both kernel width parameters must be a single scalar value
    'deformation_kernel_width': [20.0, 10.0],
    'template_kernel_width': [20.0, 10.0],
}

# Anatomical axes definitions - Adjust according to template and target bone geometries
TEMPLATE_AXES = create_axes_definition(   # Manually identify the closest matching anatomical directions in the template geometries
    anterior='x',
    superior='y',
    right='z'
)

TARGET_AXES = create_axes_definition(     # Manually identify the closest matching anatomical directions in the target geometries
    anterior='-y',
    superior='z',
    right='-x'
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_participant_directories(participant_id):
    """Get all directory paths for a specific participant."""
    target_geom_dir = f"{PARTICIPANT_BASE_DIR}/{participant_id}/target_geometry"
    target_vtk_dir = f"{target_geom_dir}/vtk_files"
    warping_files_dir = f"{PARTICIPANT_BASE_DIR}/{participant_id}/warping_files"
    opensim_output_dir = f"{PARTICIPANT_BASE_DIR}/{participant_id}"
    
    alignment_output_dir = f"{TEMP_BASE_DIR}/{participant_id}_alignment_output"
    registration_output_dir = f"{TEMP_BASE_DIR}/{participant_id}_registration_output"
    postprocessing_output_dir = f"{TEMP_BASE_DIR}/{participant_id}_postprocessing_output"
    
    return {
        'target_geom_dir': target_geom_dir,
        'target_vtk_dir': target_vtk_dir,
        'warping_files_dir': warping_files_dir,
        'opensim_output_dir': opensim_output_dir,
        'alignment_output_dir': alignment_output_dir,
        'registration_output_dir': registration_output_dir,
        'postprocessing_output_dir': postprocessing_output_dir
    }

# =============================================================================
# SINGLE PARTICIPANT PIPELINE FUNCTIONS
# =============================================================================

def process_template_meshes():
    """Process template meshes: convert STL/OBJ to VTK and scale if needed."""
    print("=== Template mesh pre-processing ===")
    print(f"Checking for VTK files in: {TEMPLATE_REGISTRATION_DIR}")

    # Check if VTK files already exist
    if check_vtk_files(TEMPLATE_REGISTRATION_DIR):
        print("Template VTK files found. No conversion needed.")
        return True
    
    print("No VTK files found. Checking for mesh files to convert...")
    print(f"Looking for mesh files in: {TEMPLATE_GEOM_DIR}")

    # Check for STL files first, then OBJ files
    has_stl = check_stl_files(TEMPLATE_GEOM_DIR)
    has_obj = check_obj_files(TEMPLATE_GEOM_DIR)
    
    if not has_stl and not has_obj:
        print("❌ No STL or OBJ files found for conversion.")
        print("Please ensure STL or OBJ files are present in the template_geometry directory.")
        return False
    
    # Determine which format to convert
    if has_stl and has_obj:
        print("⚠️  Both STL and OBJ files found. Prioritizing STL files.")
        conversion_format = 'stl'
    elif has_stl:
        conversion_format = 'stl'
    else:
        conversion_format = 'obj'
    
    print(f"\n✅ Converting {conversion_format.upper()} files to VTK format...")
    print(f"Input directory: {TEMPLATE_GEOM_DIR}")
    print(f"Output directory: {TEMPLATE_REGISTRATION_DIR}")
    print(f"Scale factor: {TEMPLATE_SCALE_FACTOR}")

    try:
        # Convert mesh files to VTK based on format
        if conversion_format == 'stl':
            from utils.mesh_converter import convert_stl_to_vtk_batch
            successful, total = convert_stl_to_vtk_batch(
                input_dir=TEMPLATE_GEOM_DIR,
                output_dir=TEMPLATE_REGISTRATION_DIR,
                scale_factor=TEMPLATE_SCALE_FACTOR,
                verbose=True
            )
        else:  # obj
            from utils.mesh_converter import convert_obj_to_vtk_batch
            successful, total = convert_obj_to_vtk_batch(
                input_dir=TEMPLATE_GEOM_DIR,
                output_dir=TEMPLATE_REGISTRATION_DIR,
                scale_factor=TEMPLATE_SCALE_FACTOR,
                verbose=True
            )
    
        if successful > 0:
            print(f"\n✅ Conversion completed successfully!")
            print(f"Converted {successful}/{total} files")
            return True
        else:
            print(f"\n❌ Conversion failed! No files were converted.")
            return False
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {str(e)}")
        return False

def create_template_landmark_files():
    """Create template landmark CSV files from settings."""
    print("\n=== Template landmark CSV file creation ===")
    print(f"Creating template landmark CSV files in: {TEMPLATE_WARPING_DIR}")

    from utils.landmark_processing_utils import create_template_landmark_csv_files

    print(f"Settings file: {SETTINGS_FILE_PATH}")
    print(f"Template geometry directory: {TEMPLATE_GEOM_DIR}")
    print(f"Output directory: {TEMPLATE_WARPING_DIR}")

    try:
        # Create template CSV files
        results = create_template_landmark_csv_files(
            settings_file_path=SETTINGS_FILE_PATH,
            template_geom_dir=TEMPLATE_GEOM_DIR,  # ← ADD THIS PARAMETER
            output_directory=TEMPLATE_WARPING_DIR,
            verbose=True
        )

        # Check results
        successful = sum(results.values())
        total = len(results)
        
        if successful > 0:
            print(f"\n✅ Template landmark CSV creation completed")
            print(f"Successfully created {successful}/{total} template CSV files")
            print(f"Files saved to: {TEMPLATE_WARPING_DIR}")
            return True
        else:
            print(f"\n❌ All template CSV file creation failed!")
            return False
            
    except Exception as e:
        print(f"\n❌ Error creating template landmark CSV files: {str(e)}")
        return False

def process_target_meshes(participant_id, directories):
    """Process target meshes: convert STL/OBJ to VTK and scale as needed."""
    print(f"\n=== Target mesh pre-processing for {participant_id} ===")
    print(f"Checking for VTK files in: {directories['target_vtk_dir']}")

    # Check if target VTK files already exist
    if check_vtk_files(directories['target_vtk_dir']):
        print("Target VTK files found. No conversion needed.")
        
        # Verify we have the same number of template and target files
        template_vtk_files = get_mesh_files(TEMPLATE_REGISTRATION_DIR, 'vtk')
        target_vtk_files = get_mesh_files(directories['target_vtk_dir'], 'vtk')
        
        if len(template_vtk_files) != len(target_vtk_files):
            print(f"⚠️  Number of template VTK files ({len(template_vtk_files)}) does not match target VTK files ({len(target_vtk_files)})")
        else:
            print(f"✅ Found matching number of template and target VTK files: {len(template_vtk_files)}")
        
        return True
        
    print(f"\nNo target VTK files found. Checking for mesh files to convert...")
    print(f"Looking for mesh files in: {directories['target_geom_dir']}")

    # Check for STL files first, then OBJ files
    has_stl = check_stl_files(directories['target_geom_dir'])
    has_obj = check_obj_files(directories['target_geom_dir'])
    
    if not has_stl and not has_obj:
        print("❌ No STL or OBJ files found for conversion.")
        print("Please ensure STL or OBJ files are present in the target_geometry directory.")
        return False
    
    # Determine which format to convert
    if has_stl and has_obj:
        print("⚠️  Both STL and OBJ files found. Prioritizing STL files.")
        conversion_format = 'stl'
    elif has_stl:
        conversion_format = 'stl'
    else:
        conversion_format = 'obj'

    print(f"\n✅ Converting target {conversion_format.upper()} files to VTK format...")
    print(f"Input directory: {directories['target_geom_dir']}")
    print(f"Output directory: {directories['target_vtk_dir']}")
    print(f"Scale factor: {TARGET_SCALE_FACTOR}")

    try:
        # Convert mesh files to VTK based on format
        if conversion_format == 'stl':
            from utils.mesh_converter import convert_stl_to_vtk_batch
            successful, total = convert_stl_to_vtk_batch(
                input_dir=directories['target_geom_dir'],
                output_dir=directories['target_vtk_dir'],
                scale_factor=TARGET_SCALE_FACTOR,
                verbose=True
            )
        else:  # obj
            from utils.mesh_converter import convert_obj_to_vtk_batch
            successful, total = convert_obj_to_vtk_batch(
                input_dir=directories['target_geom_dir'],
                output_dir=directories['target_vtk_dir'],
                scale_factor=TARGET_SCALE_FACTOR,
                verbose=True
            )
    
        if successful > 0:
            print(f"\n✅ Target conversion completed successfully!")
            print(f"Converted {successful}/{total} files")
            
            # Verify we have the same number of template and target files
            template_vtk_files = get_mesh_files(TEMPLATE_REGISTRATION_DIR, 'vtk')
            target_vtk_files = get_mesh_files(directories['target_vtk_dir'], 'vtk')
            
            if len(template_vtk_files) != len(target_vtk_files):
                print(f"⚠️  Number of template VTK files ({len(template_vtk_files)}) does not match target VTK files ({len(target_vtk_files)})")
            else:
                print(f"✅ Found matching number of template and target VTK files: {len(template_vtk_files)}")
            
            return True
        else:
            print(f"\n❌ Target conversion failed! No files were converted.")
            return False
        
    except Exception as e:
        print(f"\n❌ Error during target conversion: {str(e)}")
        return False

def perform_mesh_alignment(participant_id, directories):
    """Perform template-target mesh alignment."""
    print(f"\n=== Template-Target mesh alignment for {participant_id} ===")
    
    # Check if alignment results already exist
    alignment_output_dir = directories['alignment_output_dir']
    if os.path.exists(alignment_output_dir):
        # Check if the directory contains VTK files (alignment results)
        vtk_files = [f for f in os.listdir(alignment_output_dir) 
                    if f.endswith('.vtk')]
        
        if vtk_files:
            print(f"✅ Alignment results found in: {alignment_output_dir}")
            print(f"   Found {len(vtk_files)} aligned mesh(es)")
            print("   Skipping alignment (results already exist)")
            
            # Create a mock results dictionary indicating success for found meshes
            alignment_results = {os.path.splitext(f)[0]: True for f in vtk_files}
            
            # Verify results exist
            print("\nExisting alignment results:")
            for mesh_name in vtk_files:
                print(f"  ✅ {mesh_name}")
            
            return True, alignment_results
    
    print("No existing alignment results found. Starting mesh alignment using anatomical axes...")
    print(f"Template VTK directory: {TEMPLATE_REGISTRATION_DIR}")
    print(f"Target VTK directory: {directories['target_vtk_dir']}")
    print(f"Output directory: {alignment_output_dir}")

    print(f"Template anatomical axes: {TEMPLATE_AXES}")
    print(f"Target anatomical axes: {TARGET_AXES}")

    try:
        # Run the anatomical mesh alignment workflow
        alignment_results = anatomical_mesh_alignment_workflow(
            template_folder=TEMPLATE_REGISTRATION_DIR,
            target_folder=directories['target_vtk_dir'],
            file_extension="*.vtk",
            template_axes=TEMPLATE_AXES,
            target_axes=TARGET_AXES,
            output_folder=alignment_output_dir,
            use_icp_refinement=True,
            use_bounding_box_scaling=True,
            icp_max_iterations=50,
            icp_tolerance=1e-6,
            verbose=True
        )
        
        print(f"\nMesh alignment completed successfully")
        print(f"Processed {len(alignment_results)} mesh pairs")
        print(f"Aligned meshes saved to: {alignment_output_dir}")
        
        return True, alignment_results
        
    except Exception as e:
        print(f"\nError during mesh alignment: {str(e)}")
        return False, {}

def perform_mesh_registration(participant_id, directories):
    """Perform Deformetrica registration on aligned meshes."""
    print(f"\n=== Mesh registration for {participant_id} ===")
    
    # Check if registration results already exist
    registration_output_dir = directories['registration_output_dir']
    if os.path.exists(registration_output_dir):
        # Check if the directory contains subdirectories (registration results)
        subdirs = [d for d in os.listdir(registration_output_dir) 
                   if os.path.isdir(os.path.join(registration_output_dir, d))]
        
        if subdirs:
            print(f"✅ Registration results found in: {registration_output_dir}")
            print(f"   Found {len(subdirs)} registered mesh result(s)")
            print("   Skipping registration (results already exist)")
            
            # Create a mock results dictionary indicating success for found meshes
            registration_results = {mesh_name: True for mesh_name in subdirs}
            
            # Verify results exist
            print("\nExisting registration results:")
            for mesh_name in subdirs:
                print(f"  ✅ {mesh_name}")
            
            return True, registration_results
    
    print("No existing registration results found. Starting Deformetrica registration...")
    print(f"Aligned template meshes directory: {directories['alignment_output_dir']}")
    print(f"Registration output directory: {registration_output_dir}")

    from utils.deformetrica_registration import run_batch_registration

    # Log registration mode
    if REGISTRATION_PARAMS.get('iterative_reg', False):
        n_steps = len(REGISTRATION_PARAMS['deformation_kernel_width'])
        print(f"Registration mode: Iterative coarse-to-fine ({n_steps} steps)")
        print(f"  deformation_kernel_width: {REGISTRATION_PARAMS['deformation_kernel_width']}")
        print(f"  template_kernel_width:    {REGISTRATION_PARAMS['template_kernel_width']}")
    else:
        print(f"Registration mode: Single step")
        print(f"  deformation_kernel_width: {REGISTRATION_PARAMS.get('deformation_kernel_width', 20.0)}")
        print(f"  template_kernel_width:    {REGISTRATION_PARAMS.get('template_kernel_width', 20.0)}")

    print(f"Registration parameters:")
    for key, value in REGISTRATION_PARAMS.items():
        print(f"  {key}: {value}")

    try:
        # Run batch registration for all aligned mesh pairs
        registration_results = run_batch_registration(
            aligned_meshes_dir=directories['alignment_output_dir'],
            registration_output_dir=registration_output_dir,
            **REGISTRATION_PARAMS
        )
        
        # Check if any registrations were successful
        successful_registrations = sum(registration_results.values())
        total_registrations = len(registration_results)
        
        if successful_registrations > 0:
            print(f"\n✅ Mesh registration completed")
            print(f"Successfully registered {successful_registrations}/{total_registrations} mesh pairs")
            print(f"Registration results saved to: {registration_output_dir}")
            
            # Print summary of successful registrations
            print("\nSuccessful registrations:")
            for mesh_name, success in registration_results.items():
                if success:
                    print(f"  ✅ {mesh_name}")
            
            if successful_registrations < total_registrations:
                print("\nFailed registrations:")
                for mesh_name, success in registration_results.items():
                    if not success:
                        print(f"  ❌ {mesh_name}")
            
            return True, registration_results
        else:
            print(f"\n❌ All mesh registrations failed!")
            print("Please check the log files and Deformetrica setup.")
            return False, {}
            
    except Exception as e:
        print(f"\n❌ Error during mesh registration: {str(e)}")
        print("Please check:")
        print("1. Deformetrica is properly installed in the specified conda environment")
        print("2. WSL is configured and accessible")
        print("3. File paths are correct")
        return False, {}

def perform_postprocessing(participant_id, directories):
    """Post-process registered meshes: convert to STL and scale if needed."""
    print(f"\n=== Post-processing of registered meshes for {participant_id} ===")
    
    # Check if postprocessing results already exist
    postprocessing_output_dir = directories['postprocessing_output_dir']
    if os.path.exists(postprocessing_output_dir):
        # Check if the directory contains STL files (postprocessing results)
        stl_files = [f for f in os.listdir(postprocessing_output_dir) 
                    if f.endswith('.stl')]
        
        if stl_files:
            print(f"✅ Postprocessing results found in: {postprocessing_output_dir}")
            print(f"   Found {len(stl_files)} postprocessed mesh(es)")
            print("   Skipping postprocessing (results already exist)")
            
            # Create a mock results dictionary indicating success for found meshes
            postprocessing_results = {os.path.splitext(f)[0]: True for f in stl_files}
            
            # Verify results exist
            print("\nExisting postprocessing results:")
            for mesh_name in stl_files:
                print(f"  ✅ {mesh_name}")
            
            return True, postprocessing_results

    print("No existing postprocessing results found. Starting postprocessing...")

    from utils.postprocessing_utils import run_postprocessing_step1, verify_postprocessing_outputs

    # Determine the correct source subdirectory when iterative registration was used.
    # In iterative mode each mesh's output folder contains per-step subdirectories;
    # the final registered mesh lives inside the last iteration subdirectory.
    registration_output_dir = directories['registration_output_dir']
    iterative_reg = REGISTRATION_PARAMS.get('iterative_reg', False)

    if iterative_reg:
        print("Iterative registration was used; locating final-step output subdirectories...")
    
    print("Converting registered VTK meshes to STL format and scaling...")
    print(f"Input directory: {registration_output_dir}")
    print(f"Output directory: {postprocessing_output_dir}")
    print("Target file: DeterministicAtlas__Reconstruction__bone_trial__subject_subject_0.vtk")
    print(f"Scale factor: {POSTPROCESS_SCALE_FACTOR}")

    try:
        # run_postprocessing_step1 traverses mesh subdirectories; when iterative
        # registration is active it must look inside the deepest iter_N subdirectory.
        # The target_filename is fixed regardless of registration mode.
        postprocessing_results = run_postprocessing_step1(
            registration_output_dir=registration_output_dir,
            postprocessing_output_dir=postprocessing_output_dir,
            target_filename="DeterministicAtlas__Reconstruction__bone_trial__subject_subject_0.vtk",
            scale_factor=POSTPROCESS_SCALE_FACTOR,
            output_format="stl",
            verbose=True
        )
        
        # Verify outputs
        found_files, missing_files = verify_postprocessing_outputs(
            postprocessing_output_dir=postprocessing_output_dir,
            output_format="stl",
            verbose=True
        )
        
        # Check if any conversions were successful
        successful_postprocessing = sum(postprocessing_results.values())
        total_postprocessing = len(postprocessing_results)
        
        if successful_postprocessing > 0:
            print(f"\nPost-processing completed")
            print(f"Successfully converted {successful_postprocessing}/{total_postprocessing} registered meshes")
            print(f"STL files saved to: {postprocessing_output_dir}")
            
            # Print summary of successful conversions
            print("\nSuccessful conversions:")
            for mesh_name, success in postprocessing_results.items():
                if success:
                    print(f"  {mesh_name}.stl")
            
            if successful_postprocessing < total_postprocessing:
                print("\nFailed conversions:")
                for mesh_name, success in postprocessing_results.items():
                    if not success:
                        print(f"  {mesh_name}")
            
            return True, postprocessing_results
        else:
            print(f"\nAll post-processing conversions failed!")
            print("Please check the registration output files and file paths.")
            return False, {}
            
    except Exception as e:
        print(f"\nError during post-processing: {str(e)}")
        print("Please check:")
        print("1. Registration output directory contains the expected subdirectories")
        print("2. Target VTK files exist in each subdirectory")
        print("3. File paths are correct")
        return False, {}

def process_landmarks(participant_id, directories):
    """Process anatomical landmarks and generate CSV files using settings file."""
    print(f"\n=== Landmark processing and CSV generation for {participant_id} ===")

    from utils.landmark_processing_utils import run_landmark_processing, verify_landmark_csv_files

    print("Processing anatomical landmarks and generating CSV files from settings...")
    print(f"Settings file: {SETTINGS_FILE_PATH}")
    print(f"Template geometry directory: {TEMPLATE_GEOM_DIR}")
    print(f"Target meshes directory: {directories['postprocessing_output_dir']}")
    print(f"Output directory: {directories['warping_files_dir']}")

    try:
        # Check if settings file exists
        if not os.path.exists(SETTINGS_FILE_PATH):
            print(f"❌ Settings file not found at {SETTINGS_FILE_PATH}")
            print("Please ensure the MSK-Morph_settings.yaml file is in the correct location.")
            print("Skipping landmark processing...")
            return False, {}
        
        # Run landmark processing using settings file
        processed_landmarks, landmark_csv_results = run_landmark_processing(
            settings_file_path=SETTINGS_FILE_PATH,
            template_geom_dir=TEMPLATE_GEOM_DIR,
            mesh_directory=directories['postprocessing_output_dir'],
            output_directory=directories['warping_files_dir'],
            verbose=True
        )
        
        # Verify outputs
        found_csv_files, missing_csv_files = verify_landmark_csv_files(
            output_directory=directories['warping_files_dir'],
            settings_file_path=SETTINGS_FILE_PATH,
            verbose=True
        )
        
        # Check results
        successful_csv = sum(landmark_csv_results.values())
        total_csv = len(landmark_csv_results)
        
        if successful_csv > 0:
            print(f"\n✅ Landmark processing completed")
            print(f"Successfully processed {successful_csv}/{total_csv} CSV files")
            print(f"CSV files saved to: {directories['warping_files_dir']}")
            
            # Print summary of successful processing
            print("\nSuccessful CSV file creation:")
            for csv_name, success in landmark_csv_results.items():
                if success:
                    print(f"  ✅ {csv_name}")
            
            if successful_csv < total_csv:
                print("\nFailed CSV file creation:")
                for csv_name, success in landmark_csv_results.items():
                    if not success:
                        print(f"  ❌ {csv_name}")
            
            # Print landmark summary
            print(f"\nTotal landmarks processed: {len(processed_landmarks)}")
            
            return True, processed_landmarks
        else:
            print(f"\n❌ All landmark processing failed!")
            print("Please check:")
            print("1. Settings file format and content")
            print("2. Mesh files in postprocessing output directory")
            print("3. PyMeshLab installation")
            return False, {}
            
    except Exception as e:
        print(f"\n❌ Error during landmark processing: {str(e)}")
        print("Please check:")
        print("1. Settings file exists and is readable")
        print("2. Mesh files are accessible")
        print("3. PyMeshLab is properly installed")
        print("4. Output directory permissions")
        return False, {}
      
def export_opensim_landmarks(participant_id, directories):
    """Export landmarks to OpenSim XML and TRC formats."""
    print(f"\n=== OpenSim landmark export for {participant_id} ===")

    from utils.landmark_export_utils import export_landmarks_to_opensim_formats, verify_opensim_exports

    print("Exporting landmarks to OpenSim XML and TRC formats...")
    print(f"Settings file: {SETTINGS_FILE_PATH}")
    print(f"Input CSV directory: {directories['warping_files_dir']}")
    print(f"Output directory: {directories['opensim_output_dir']}")
    print(f"Participant ID: {participant_id}")

    try:
        # Export landmarks to OpenSim formats using settings file
        opensim_export_results = export_landmarks_to_opensim_formats(
            settings_file_path=SETTINGS_FILE_PATH,
            csv_directory=directories['warping_files_dir'],
            output_directory=directories['opensim_output_dir'],
            participant_id=participant_id,
            verbose=True
        )
        
        # Verify exports
        xml_exists, trc_exists = verify_opensim_exports(
            output_directory=directories['opensim_output_dir'],
            verbose=True
        )
        
        # Check results
        if opensim_export_results['xml'] and opensim_export_results['trc'] and xml_exists and trc_exists:
            print(f"\n✅ OpenSim landmark export completed")
            print(f"Successfully exported landmarks to OpenSim formats")
            print(f"Files created:")
            print(f"  ✅ markers.xml")
            print(f"  ✅ markers.trc")
            print(f"Files saved to: {directories['opensim_output_dir']}")
            
            return True
        else:
            print(f"\n❌ OpenSim landmark export failed!")
            print("Please check:")
            print("1. CSV landmark files exist and are readable")
            print("2. Output directory permissions")
            print("3. Settings file configuration")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during OpenSim landmark export: {str(e)}")
        print("Please check:")
        print("1. CSV landmark files exist and contain required landmarks")
        print("2. Output directory is writable")
        print("3. Settings file is correct")
        return False

def create_warping_settings(participant_id, directories):
    """Create participant-specific warping settings file."""
    print(f"\n=== Warping settings file creation for {participant_id} ===")

    from utils.warping_settings_utils import create_warping_settings_file, validate_warping_settings_file

    # Extract directory and filename from the template path
    template_dir = os.path.dirname(TEMPLATE_WARPING_SET_FILENAME)
    template_filename = os.path.basename(TEMPLATE_WARPING_SET_FILENAME)

    print("Creating participant-specific warping settings file...")
    print(f"Participant ID: {participant_id}")
    print(f"Template file: {TEMPLATE_WARPING_SET_FILENAME}")
    print(f"Output directory: {directories['warping_files_dir']}")

    try:
        # Create the participant-specific settings file
        settings_creation_success = create_warping_settings_file(
            participant_id=participant_id,
            template_directory=template_dir,
            template_filename=template_filename,
            output_directory=directories['warping_files_dir'],
            verbose=True
        )
        
        if settings_creation_success:
            print(f"\nWarping settings file created successfully")
            
            # Validate the created file - construct path from output directory and template filename
            settings_file_path = os.path.join(directories['warping_files_dir'], template_filename)
            
            print("Validating created settings file...")
            validation_success = validate_warping_settings_file(
                settings_file_path=settings_file_path,
                participant_id=participant_id,
                verbose=True
            )
            
            if validation_success:
                print(f"Warping settings file validated successfully")
                print(f"Settings file ready for use: {settings_file_path}")
                return True
            else:
                print(f"Settings file validation failed")
                print("Please check the file content manually")
                return False
        else:
            print(f"\nFailed to create warping settings file!")
            print("Please check:")
            print(f"1. Template file exists at: {TEMPLATE_WARPING_SET_FILENAME}")
            print("2. Output directory is writable")
            print("3. Template file format is correct")
            return False
            
    except Exception as e:
        print(f"\nError during warping settings file creation: {str(e)}")
        print("Please check:")
        print(f"1. Template file path is correct: {TEMPLATE_WARPING_SET_FILENAME}")
        print("2. Template file is readable")
        print("3. Output directory permissions")
        return False

def print_participant_summary(participant_id, results):
    """Print final pipeline summary for a participant."""
    
    print(f"\n=== PIPELINE SUMMARY FOR {participant_id} ===")
    
    target_success = results.get('target_processing', False)
    alignment_success = results.get('alignment', False)
    registration_success = results.get('registration', False)
    postprocessing_success = results.get('postprocessing', False)
    landmark_success = results.get('landmark_processing', False)
    opensim_success = results.get('opensim_export', False)
    settings_success = results.get('warping_settings', False)
    
    print(f"Target mesh preprocessing: {'Completed' if target_success else 'Failed'}")
    print(f"Mesh alignment: {'Completed' if alignment_success else 'Failed'}")
    print(f"Mesh registration: {'Completed' if registration_success else 'Failed'}")
    print(f"Post-processing: {'Completed' if postprocessing_success else 'Failed'}")
    print(f"Landmark processing: {'Completed' if landmark_success else 'Had issues' if landmark_success is None else 'Failed'}")
    print(f"OpenSim export: {'Completed' if opensim_success else 'Had issues' if opensim_success is None else 'Failed'}")
    print(f"Warping settings: {'Completed' if settings_success else 'Had issues' if settings_success is None else 'Failed'}")

    # Overall success check
    core_success = all([target_success, alignment_success, registration_success, postprocessing_success])
    all_success = core_success and landmark_success and opensim_success and settings_success
    
    # Get the output filename from the template path
    template_filename = os.path.basename(TEMPLATE_WARPING_SET_FILENAME)
    
    if all_success:
        print(f"\nPipeline completed successfully for {participant_id}! All files are ready for OpenSim ModelWarper.")
        directories = get_participant_directories(participant_id)
        print(f"\nNext steps:")
        print(f"1. Use OpenSim ModelWarper with the settings file:")
        print(f"   {directories['warping_files_dir']}/{template_filename}")
        print(f"2. The warping will use the landmark files in:")
        print(f"   {directories['warping_files_dir']}/")
        return True
    elif core_success:
        print(f"\nCore pipeline completed successfully for {participant_id} with some optional warnings.")
        print(f"Check logs for details on landmark processing, OpenSim export, or settings creation.")
        return True
    else:
        print(f"\nPipeline failed for {participant_id}. Check logs for details.")
        return False
    
def process_single_participant(participant_id):
    """Process a single participant through the entire pipeline."""
    print(f"\n{'='*50}")
    print(f"PROCESSING PARTICIPANT: {participant_id}")
    print(f"{'='*50}")
    
    # Get participant directories
    directories = get_participant_directories(participant_id)
    
    print(f"Processing directories:")
    print(f"  Target geometry: {directories['target_geom_dir']}")
    print(f"  Output base: {directories['opensim_output_dir']}")
    
    # Track results
    results = {}
    
    # Step 1: Target mesh preprocessing
    results['target_processing'] = process_target_meshes(participant_id, directories)
    if not results['target_processing']:
        print(f"Target preprocessing failed for {participant_id}. Stopping pipeline.")
        return False
    
    # Step 2: Mesh alignment
    results['alignment'], alignment_data = perform_mesh_alignment(participant_id, directories)
    if not results['alignment']:
        print(f"Mesh alignment failed for {participant_id}. Stopping pipeline.")
        return False
    
    # Step 3: Mesh registration
    results['registration'], registration_data = perform_mesh_registration(participant_id, directories)
    if not results['registration']:
        print(f"Mesh registration failed for {participant_id}. Stopping pipeline.")
        return False
    
    # Step 4: Post-processing
    results['postprocessing'], postprocessing_data = perform_postprocessing(participant_id, directories)
    if not results['postprocessing']:
        print(f"Post-processing failed for {participant_id}. Stopping pipeline.")
        return False
    
    # Step 5: Landmark processing
    results['landmark_processing'], landmark_data = process_landmarks(participant_id, directories)
    
    # Step 6: OpenSim export (only if landmarks succeeded)
    if results['landmark_processing']:
        results['opensim_export'] = export_opensim_landmarks(participant_id, directories)
    else:
        results['opensim_export'] = None
        print("Skipping OpenSim export due to landmark processing issues...")
    
    # Step 7: Warping settings (only if landmarks succeeded)
    if results['landmark_processing']:
        results['warping_settings'] = create_warping_settings(participant_id, directories)
    else:
        results['warping_settings'] = None
        print("Skipping warping settings creation due to landmark processing issues...")
    
    # Print participant summary
    return print_participant_summary(participant_id, results)

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main pipeline execution."""
    print(f"=== MSK Model Morphing Pipeline ===")
    print(f"Settings file: {SETTINGS_FILE_PATH}")
    print(f"Participant IDs: {PARTICIPANT_IDS}")
    print(f"Processing {len(PARTICIPANT_IDS)} participant(s)")
    
    # Check if settings file exists
    if not os.path.exists(SETTINGS_FILE_PATH):
        print(f"\n❌ ERROR: Settings file not found: {SETTINGS_FILE_PATH}")
        print("Please ensure MSK-Morph_settings.yaml is in the correct location.")
        sys.exit(1)
    else:
        print(f"✅ Settings file found: {SETTINGS_FILE_PATH}")
    
    # Step 1: Template mesh preprocessing (shared across all participants)
    print("\n=== SHARED TEMPLATE PREPROCESSING ===")
    template_success = process_template_meshes()
    if not template_success:
        print("❌ Template preprocessing failed. Stopping pipeline.")
        sys.exit(1)
    
    # Step 1.5: Create template landmark CSV files
    template_csv_success = create_template_landmark_files()
    if not template_csv_success:
        print("⚠️ Template landmark CSV creation had issues, but continuing...")
    
    # Step 2: Process each participant
    successful_participants = 0
    failed_participants = []
    
    for participant_id in PARTICIPANT_IDS:
        participant_success = process_single_participant(participant_id)
        if participant_success:
            successful_participants += 1
        else:
            failed_participants.append(participant_id)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"Total participants: {len(PARTICIPANT_IDS)}")
    print(f"Successful: {successful_participants}")
    print(f"Failed: {len(failed_participants)}")
    
    if successful_participants == len(PARTICIPANT_IDS):
        print(f"\nAll participants processed successfully!")
    elif successful_participants > 0:
        print(f"\nPartially successful batch processing.")
        if failed_participants:
            print(f"Failed participants: {failed_participants}")
    else:
        print(f"\nBatch processing failed for all participants.")
        if failed_participants:
            print(f"Failed participants: {failed_participants}")
        sys.exit(1)

if __name__ == "__main__":
    main()