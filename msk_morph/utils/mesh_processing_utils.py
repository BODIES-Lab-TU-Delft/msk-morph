#!/usr/bin/env python3
"""
Mesh Processing Utilities Module
Utility functions for checking and managing mesh files.
"""

from pathlib import Path
from typing import List, Tuple

def check_vtk_files(directory: str) -> bool:
    """
    Check if there are any VTK files in the specified directory.
    
    Args:
        directory: Path to directory to check
        
    Returns:
        bool: True if VTK files found, False otherwise
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"❌ Directory does not exist: {directory}")
        return False
    
    # Look for VTK files (both .vtk and .VTK extensions) and remove duplicates
    vtk_files = list(set(list(dir_path.glob("*.vtk")) + list(dir_path.glob("*.VTK"))))
    
    if vtk_files:
        print(f"✅ Found {len(vtk_files)} VTK file(s) in {directory}")
        for vtk_file in sorted(vtk_files):
            print(f"  ✅ {vtk_file.name}")
        return True
    else:
        print(f"❌ No VTK files found in {directory}")
        return False

def check_stl_files(directory: str) -> bool:
    """
    Check if there are any STL files in the specified directory.
    
    Args:
        directory: Path to directory to check
        
    Returns:
        bool: True if STL files found, False otherwise
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"❌ Directory does not exist: {directory}")
        return False
    
    # Look for STL files (both .stl and .STL extensions) and remove duplicates
    stl_files = list(set(list(dir_path.glob("*.stl")) + list(dir_path.glob("*.STL"))))
    
    if stl_files:
        print(f"✅ Found {len(stl_files)} STL file(s) in {directory}")
        for stl_file in sorted(stl_files):
            print(f"  ✅ {stl_file.name}")
        return True
    else:
        print(f"❌ No STL files found in {directory}")
        return False

def check_obj_files(directory: str) -> bool:
    """
    Check if there are any OBJ files in the specified directory.
    
    Args:
        directory: Path to directory to check
        
    Returns:
        bool: True if OBJ files found, False otherwise
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"❌ Directory does not exist: {directory}")
        return False
    
    # Look for OBJ files (both .obj and .OBJ extensions) and remove duplicates
    obj_files = list(set(list(dir_path.glob("*.obj")) + list(dir_path.glob("*.OBJ"))))
    
    if obj_files:
        print(f"✅ Found {len(obj_files)} OBJ file(s) in {directory}")
        for obj_file in sorted(obj_files):
            print(f"  ✅ {obj_file.name}")
        return True
    else:
        print(f"❌ No OBJ files found in {directory}")
        return False

def get_mesh_files(directory: str, file_type: str) -> List[Path]:
    """
    Get list of mesh files of specified type in directory.
    
    Args:
        directory: Path to directory to search
        file_type: Type of files to search for ('vtk', 'stl', or 'obj')
        
    Returns:
        List[Path]: List of found mesh files
    """
    dir_path = Path(directory)
    
    if not dir_path.exists():
        print(f"❌ Directory does not exist: {directory}")
        return []
    
    if file_type.lower() == 'vtk':
        files = list(set(list(dir_path.glob("*.vtk")) + list(dir_path.glob("*.VTK"))))
    elif file_type.lower() == 'stl':
        files = list(set(list(dir_path.glob("*.stl")) + list(dir_path.glob("*.STL"))))
    elif file_type.lower() == 'obj':
        files = list(set(list(dir_path.glob("*.obj")) + list(dir_path.glob("*.OBJ"))))
    else:
        print(f"❌ Unsupported file type: {file_type}")
        return []
    
    if files:
        print(f"✅ Found {len(files)} {file_type.upper()} file(s) in {directory}")
    else:
        print(f"⚠️  No {file_type.upper()} files found in {directory}")
    
    return sorted(files)

def verify_mesh_file_counts(template_dir: str, target_dir: str, file_type: str = 'vtk') -> Tuple[bool, int, int]:
    """
    Verify that template and target directories have matching numbers of mesh files.
    
    Args:
        template_dir: Path to template mesh directory
        target_dir: Path to target mesh directory
        file_type: Type of files to check ('vtk', 'stl', or 'obj')
        
    Returns:
        Tuple[bool, int, int]: (files_match, template_count, target_count)
    """
    template_files = get_mesh_files(template_dir, file_type)
    target_files = get_mesh_files(target_dir, file_type)
    
    template_count = len(template_files)
    target_count = len(target_files)
    
    if template_count == target_count:
        print(f"✅ Matching number of {file_type.upper()} files: {template_count} template, {target_count} target")
        return True, template_count, target_count
    else:
        print(f"⚠️  Mismatched {file_type.upper()} file counts: {template_count} template, {target_count} target")
        return False, template_count, target_count

def check_mesh_file_names(template_dir: str, target_dir: str, file_type: str = 'vtk') -> Tuple[List[str], List[str], List[str]]:
    """
    Check which mesh files exist in both directories and identify missing files.
    
    Args:
        template_dir: Path to template mesh directory
        target_dir: Path to target mesh directory
        file_type: Type of files to check ('vtk', 'stl', or 'obj')
        
    Returns:
        Tuple[List[str], List[str], List[str]]: (common_files, template_only, target_only)
    """
    template_files = get_mesh_files(template_dir, file_type)
    target_files = get_mesh_files(target_dir, file_type)
    
    # Extract just the filenames (without extensions) for comparison
    template_names = {f.stem for f in template_files}
    target_names = {f.stem for f in target_files}
    
    common_files = sorted(template_names.intersection(target_names))
    template_only = sorted(template_names - target_names)
    target_only = sorted(target_names - template_names)
    
    print(f"\n=== {file_type.upper()} File Name Analysis ===")
    
    if common_files:
        print(f"✅ Common files ({len(common_files)}):")
        for name in common_files:
            print(f"  ✅ {name}.{file_type}")
    
    if template_only:
        print(f"⚠️  Template-only files ({len(template_only)}):")
        for name in template_only:
            print(f"  ⚠️  {name}.{file_type}")
    
    if target_only:
        print(f"⚠️  Target-only files ({len(target_only)}):")
        for name in target_only:
            print(f"  ⚠️  {name}.{file_type}")
    
    if not common_files:
        print("❌ No common files found between template and target directories!")
    
    return common_files, template_only, target_only

def validate_mesh_directories(template_dir: str, target_dir: str, file_type: str = 'vtk') -> bool:
    """
    Comprehensive validation of mesh directories for pipeline processing.
    
    Args:
        template_dir: Path to template mesh directory
        target_dir: Path to target mesh directory
        file_type: Type of files to check ('vtk', 'stl', or 'obj')
        
    Returns:
        bool: True if directories are valid, False otherwise
    """
    print(f"\n=== Validating Mesh Directories ({file_type.upper()}) ===")
    
    # Check counts
    files_match, template_count, target_count = verify_mesh_file_counts(
        template_dir, target_dir, file_type
    )
    
    if template_count == 0:
        print(f"❌ No {file_type.upper()} files found in template directory")
        return False
    
    if target_count == 0:
        print(f"❌ No {file_type.upper()} files found in target directory")
        return False
    
    # Check file names
    common_files, template_only, target_only = check_mesh_file_names(
        template_dir, target_dir, file_type
    )
    
    if not common_files:
        print("❌ Validation failed: No matching files between directories")
        return False
    
    if not files_match:
        print("⚠️  Warning: File counts don't match, but some common files exist")
        return True  # Still allow processing if there are common files
    
    print("✅ Validation successful")
    return True

def create_directory_structure(base_dir: str, participant_id: str) -> bool:
    """
    Create the complete directory structure for a participant.
    
    Args:
        base_dir: Base directory for participant data
        participant_id: Participant identifier
        
    Returns:
        bool: True if successful, False otherwise
    """
    from pathlib import Path
    
    participant_dir = Path(base_dir) / participant_id
    
    directories = [
        participant_dir / "target_geometry",
        participant_dir / "target_geometry" / "vtk_files",
        participant_dir / "warping_files"
    ]
    
    print(f"\n=== Creating Directory Structure for {participant_id} ===")
    
    try:
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {directory}")
        
        print(f"✅ Directory structure created successfully")
        return True
    
    except Exception as e:
        print(f"❌ Error creating directory structure: {str(e)}")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mesh_processing_utils.py <command> [args...]")
        print("\nCommands:")
        print("  check_vtk <directory>                    - Check for VTK files")
        print("  check_stl <directory>                    - Check for STL files")
        print("  check_obj <directory>                    - Check for OBJ files")
        print("  validate <template_dir> <target_dir>     - Validate mesh directories")
        print("  create_dirs <base_dir> <participant_id>  - Create directory structure")
        print("\nExamples:")
        print("  python mesh_processing_utils.py check_vtk ./template_mesh_registration")
        print("  python mesh_processing_utils.py check_obj ./template_geometry")
        print("  python mesh_processing_utils.py validate ./template_mesh_registration ./target_vtk")
        print("  python mesh_processing_utils.py create_dirs ../participant_data participant_id")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "check_vtk":
        if len(sys.argv) < 3:
            print("❌ Usage: check_vtk <directory>")
            sys.exit(1)
        directory = sys.argv[2]
        success = check_vtk_files(directory)
        sys.exit(0 if success else 1)
    
    elif command == "check_stl":
        if len(sys.argv) < 3:
            print("❌ Usage: check_stl <directory>")
            sys.exit(1)
        directory = sys.argv[2]
        success = check_stl_files(directory)
        sys.exit(0 if success else 1)
    
    elif command == "check_obj":
        if len(sys.argv) < 3:
            print("❌ Usage: check_obj <directory>")
            sys.exit(1)
        directory = sys.argv[2]
        success = check_obj_files(directory)
        sys.exit(0 if success else 1)
    
    elif command == "validate":
        if len(sys.argv) < 4:
            print("❌ Usage: validate <template_dir> <target_dir> [file_type]")
            sys.exit(1)
        template_dir = sys.argv[2]
        target_dir = sys.argv[3]
        file_type = sys.argv[4] if len(sys.argv) > 4 else "vtk"
        success = validate_mesh_directories(template_dir, target_dir, file_type)
        sys.exit(0 if success else 1)
    
    elif command == "create_dirs":
        if len(sys.argv) < 4:
            print("❌ Usage: create_dirs <base_dir> <participant_id>")
            sys.exit(1)
        base_dir = sys.argv[2]
        participant_id = sys.argv[3]
        success = create_directory_structure(base_dir, participant_id)
        sys.exit(0 if success else 1)
    
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)