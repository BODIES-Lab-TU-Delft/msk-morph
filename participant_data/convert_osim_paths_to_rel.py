"""
Standalone OpenSim Model Path Converter
Converts absolute paths to relative paths in OpenSim .osim model files.

Usage:
    python convert_osim_paths.py

Configuration:
    Edit the PARTICIPANT_IDS and BASE_DIRECTORY variables below
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple

# =============================================================================
# CONFIGURATION - EDIT THESE PARAMETERS
# =============================================================================

# List of participant IDs to process
PARTICIPANT_IDS = ["test"]  # List of participant IDs to process in batch mode ["001","002","003"]

# Base directory containing participant folders
BASE_DIRECTORY = ""

# Suffix to add to output filenames (e.g., "model.osim" --> "model_rel-path.osim")
OUTPUT_SUFFIX = "_rel-path"

# Target subfolder name (for display/reference only - actual relative paths are calculated)
TARGET_SUBFOLDER = "morphed_geometry"

# Enable verbose output
VERBOSE = True

# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

def find_osim_files(participant_ids: List[str], base_directory: str) -> Dict[str, List[str]]:
    """Find all .osim files for specified participants."""
    osim_files = {}
    
    for participant_id in participant_ids:
        participant_dir = os.path.join(base_directory, participant_id)
        
        if not os.path.exists(participant_dir):
            print(f"⚠️ Participant directory not found: {participant_dir}")
            continue
        
        # Find all .osim files in participant directory
        files = []
        for root, dirs, filenames in os.walk(participant_dir):
            for filename in filenames:
                if filename.endswith('.osim'):
                    full_path = os.path.join(root, filename)
                    files.append(full_path)
        
        if files:
            osim_files[participant_id] = files
            if VERBOSE:
                print(f"✅ Found {len(files)} .osim file(s) for participant {participant_id}")
                for f in files:
                    print(f"   - {os.path.basename(f)}")
        else:
            print(f"⚠️ No .osim files found for participant {participant_id}")
    
    return osim_files


def extract_mesh_paths(osim_file_path: str) -> List[Tuple[str, str]]:
    """Extract all mesh file paths from an OpenSim model file."""
    mesh_paths = []
    
    try:
        with open(osim_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Find all <mesh_file> tags using regex
        pattern = r'<mesh_file>(.*?)</mesh_file>'
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            mesh_path = match.group(1).strip()
            if mesh_path:
                mesh_paths.append((match.group(0), mesh_path))
        
        return mesh_paths
        
    except Exception as e:
        print(f"❌ Error reading {osim_file_path}: {e}")
        return []


def convert_to_relative_path(absolute_path: str, osim_file_dir: str) -> str:
    """
    Convert absolute path to relative path.
    
    Calculates the actual relative path from the .osim file location
    to the mesh file location.
    
    Args:
        absolute_path: Absolute path to the mesh file
        osim_file_dir: Directory containing the .osim file
        
    Returns:
        Relative path from osim_file_dir to mesh file (with forward slashes)
    """
    # Normalize path separators to OS standard
    absolute_path = absolute_path.replace('\\', os.sep).replace('/', os.sep)
    osim_file_dir = osim_file_dir.replace('\\', os.sep).replace('/', os.sep)
    
    # Convert to absolute paths for proper calculation
    absolute_path = os.path.abspath(absolute_path)
    osim_file_dir = os.path.abspath(osim_file_dir)
    
    # Calculate relative path
    try:
        relative_path = os.path.relpath(absolute_path, osim_file_dir)
    except ValueError:
        # On Windows, relpath can fail if paths are on different drives
        # In this case, just use the basename in current directory
        print(f"     ⚠️ Warning: Cannot create relative path (different drives?)")
        print(f"        Using filename only: {os.path.basename(absolute_path)}")
        relative_path = os.path.basename(absolute_path)
    
    # Convert to forward slashes for OpenSim compatibility (cross-platform)
    relative_path = relative_path.replace(os.sep, '/')
    
    return relative_path


def is_absolute_path(path: str) -> bool:
    """Check if a path is absolute."""
    # Check for Windows drive letters or absolute Unix paths
    return (os.path.isabs(path) or 
            path.startswith(('C:', 'c:', 'D:', 'd:', 'E:', 'e:')))


def process_osim_file(osim_file_path: str, output_suffix: str, 
                     target_subfolder: str) -> Tuple[bool, str]:
    """Process a single .osim file to convert paths."""
    try:
        # Read the file
        with open(osim_file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Get the directory containing the .osim file
        osim_file_dir = os.path.dirname(os.path.abspath(osim_file_path))
        
        # Extract mesh paths
        mesh_references = extract_mesh_paths(osim_file_path)
        
        if not mesh_references:
            if VERBOSE:
                print(f"  ⚠️ No mesh file references found")
            return False, ""
        
        if VERBOSE:
            print(f"  ✅ Found {len(mesh_references)} mesh file reference(s)")
            print(f"  .osim file directory: {osim_file_dir}")
        
        # Track replacements
        replacements_made = 0
        
        # Replace each absolute path with relative path
        for full_tag, absolute_path in mesh_references:
            # Check if path is already relative
            if not is_absolute_path(absolute_path):
                if VERBOSE:
                    print(f"     ⚠️ Already relative: {absolute_path}")
                continue
            
            # Convert to relative path (calculate from osim file location)
            relative_path = convert_to_relative_path(absolute_path, osim_file_dir)
            
            # Create new tag
            new_tag = f"<mesh_file>{relative_path}</mesh_file>"
            
            # Replace in content
            content = content.replace(full_tag, new_tag)
            replacements_made += 1
            
            if VERBOSE:
                print(f"     ✅ Converted: {os.path.basename(absolute_path)}")
                print(f"        From: {absolute_path}")
                print(f"        To:   {relative_path}")
        
        if replacements_made == 0:
            if VERBOSE:
                print(f"  ⚠️ No absolute paths found to convert")
            return False, ""
        
        # Create output filename
        file_path = Path(osim_file_path)
        output_filename = f"{file_path.stem}{output_suffix}{file_path.suffix}"
        output_file_path = file_path.parent / output_filename
        
        # Write modified content
        with open(output_file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        if VERBOSE:
            print(f"  ✅ Created: {output_filename} ({replacements_made} path(s) converted)")
        
        return True, str(output_file_path)
        
    except Exception as e:
        print(f"  ❌ Error processing file: {e}")
        import traceback
        traceback.print_exc()
        return False, ""


def main():
    """Main execution function."""
    print("=" * 70)
    print("OpenSim Model Path Converter")
    print("=" * 70)
    print()
    
    # Display configuration
    print("Configuration:")
    print(f"  Participant IDs: {PARTICIPANT_IDS}")
    print(f"  Base directory: {BASE_DIRECTORY}")
    print(f"  Output suffix: {OUTPUT_SUFFIX}")
    print(f"  Target subfolder: {TARGET_SUBFOLDER}")
    print()
    
    # Find all .osim files
    print("=" * 70)
    print("Finding .osim files...")
    print("=" * 70)
    print()
    
    osim_files = find_osim_files(PARTICIPANT_IDS, BASE_DIRECTORY)
    
    if not osim_files:
        print("❌ No .osim files found for any participant")
        print("\nPlease check:")
        print("  1. Participant IDs are correct")
        print("  2. Base directory path is correct")
        print("  3. .osim files exist in participant directories")
        return
    
    # Process each file
    print()
    print("=" * 70)
    print("Processing .osim files...")
    print("=" * 70)
    print()
    
    total_processed = 0
    total_successful = 0
    results = {}
    
    for participant_id, file_list in osim_files.items():
        print(f"\n{'='*70}")
        print(f"Participant: {participant_id}")
        print(f"{'='*70}")
        
        participant_results = {}
        
        for osim_file_path in file_list:
            print(f"\nProcessing: {os.path.basename(osim_file_path)}")
            
            success, output_path = process_osim_file(
                osim_file_path, OUTPUT_SUFFIX, TARGET_SUBFOLDER
            )
            
            participant_results[os.path.basename(osim_file_path)] = success
            total_processed += 1
            
            if success:
                total_successful += 1
        
        results[participant_id] = participant_results
    
    # Print summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    print(f"Total files processed: {total_processed}")
    print(f"Successfully converted: {total_successful}/{total_processed}")
    print()
    
    if total_successful < total_processed:
        print("Failed conversions:")
        for participant_id, file_results in results.items():
            for filename, success in file_results.items():
                if not success:
                    print(f"  ❌ {participant_id}/{filename}")
        print()
    
    # Success summary by participant
    print("Results by participant:")
    for participant_id, file_results in results.items():
        successful = sum(file_results.values())
        total = len(file_results)
        status = "✅" if successful == total else "⚠️"
        print(f"  {status} {participant_id}: {successful}/{total} files converted")
    
    print()
    print("=" * 70)
    
    if total_successful == total_processed:
        print("✅ All conversions completed successfully!")
    elif total_successful > 0:
        print("⚠️ Partial conversion completed")
    else:
        print("❌ All conversions failed")
    
    print("=" * 70)


# =============================================================================
# SCRIPT ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Script interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()