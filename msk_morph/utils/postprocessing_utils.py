#!/usr/bin/env python3
"""
Post-Processing Utilities Module
Functions for processing Deformetrica registration outputs.
"""

import os
import glob
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Import the mesh converter utilities
from utils.mesh_converter import convert_single_mesh

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PostProcessingPipeline:
    """
    Pipeline for post-processing Deformetrica registration outputs.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the post-processing pipeline.
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        if not verbose:
            logger.setLevel(logging.WARNING)
    
    def _find_target_in_subdir(self, subdir: Path, target_filename: str) -> Optional[Path]:
        """
        Search for the target file within a mesh subdirectory.

        Handles two directory layouts:

        Single-step registration (flat):
            <subdir>/
                DeterministicAtlas__Reconstruction__bone_trial__subject_subject_0.vtk

        Iterative registration (nested):
            <subdir>/
                iter_1_of_2__defkw20.0__tmplkw20.0/
                    DeterministicAtlas__Reconstruction__bone_trial__subject_subject_0.vtk
                iter_2_of_2__defkw10.0__tmplkw10.0/   <- last step = final result
                    DeterministicAtlas__Reconstruction__bone_trial__subject_subject_0.vtk

        For the iterative layout the file from the *last* iteration subdirectory
        (sorted alphabetically, which preserves step order due to the naming
        convention) is returned so that postprocessing always uses the most
        refined registration result.

        Args:
            subdir: Path object for the per-mesh registration output directory
            target_filename: Deformetrica output filename to search for

        Returns:
            Path to the target file, or None if not found
        """
        # --- Case 1: file sits directly in the mesh subdirectory (single-step) ---
        direct_candidate = subdir / target_filename
        if direct_candidate.exists():
            return direct_candidate

        # --- Case 2: file is nested inside iter_* subdirectories (iterative) ---
        iter_subdirs = sorted(
            [d for d in subdir.iterdir() if d.is_dir() and d.name.startswith("iter_")]
        )

        if iter_subdirs:
            # Use the last iteration directory — highest step index = final result
            last_iter_dir = iter_subdirs[-1]
            nested_candidate = last_iter_dir / target_filename
            if nested_candidate.exists():
                logger.info(
                    f"   Iterative registration detected: using final step "
                    f"'{last_iter_dir.name}'"
                )
                return nested_candidate
            else:
                logger.warning(
                    f"⚠️  Last iteration dir found ('{last_iter_dir.name}') but "
                    f"'{target_filename}' is missing inside it"
                )
                return None

        # --- No match in either layout ---
        return None

    def find_deformetrica_output_files(self, registration_output_dir: str, 
                                       target_filename: str = "DeterministicAtlas__Reconstruction__bone_trial__subject_subject_0.vtk") -> Dict[str, str]:
        """
        Find all Deformetrica output files in registration subdirectories.

        Supports both single-step (flat) and iterative (nested iter_* dirs)
        registration output layouts. See _find_target_in_subdir for details.
        
        Args:
            registration_output_dir: Base registration output directory
            target_filename: Specific filename to look for in each subdirectory
            
        Returns:
            Dict[str, str]: Dictionary mapping mesh_name to full file path
        """
        mesh_files = {}
        registration_path = Path(registration_output_dir)
        
        if not registration_path.exists():
            logger.error(f"❌ Registration output directory does not exist: {registration_output_dir}")
            return mesh_files
        
        # Top-level entries are per-mesh subdirectories
        subdirs = [d for d in registration_path.iterdir() if d.is_dir()]
        
        if not subdirs:
            logger.warning(f"⚠️  No subdirectories found in {registration_output_dir}")
            return mesh_files
        
        logger.info(f"Searching for '{target_filename}' in {len(subdirs)} mesh subdirectory(ies)...")
        
        for subdir in sorted(subdirs):
            mesh_name = subdir.name
            target_file_path = self._find_target_in_subdir(subdir, target_filename)
            
            if target_file_path is not None:
                mesh_files[mesh_name] = str(target_file_path)
                if self.verbose:
                    logger.info(f"✅ Found: {mesh_name} -> {target_file_path}")
            else:
                logger.warning(
                    f"⚠️  Target file not found for '{mesh_name}' "
                    f"(checked flat and iterative layouts)"
                )
        
        logger.info(f"Found {len(mesh_files)} mesh file(s) for post-processing")
        return mesh_files
    
    def convert_and_scale_meshes(self, mesh_files: Dict[str, str], 
                               output_dir: str,
                               scale_factor: float = 1.0,
                               output_format: str = "stl") -> Dict[str, bool]:
        """
        Convert and scale meshes from VTK to STL format.
        
        Args:
            mesh_files: Dictionary mapping mesh_name to input file path
            output_dir: Output directory for converted files
            scale_factor: Scale factor for conversion (default: 1.0)
            output_format: Output format ('stl' or 'vtk')
            
        Returns:
            Dict[str, bool]: Dictionary mapping mesh_name to success status
        """
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = {}
        
        logger.info(f"Converting {len(mesh_files)} meshes...")
        logger.info(f"Scale factor: {scale_factor}")
        logger.info(f"Output format: {output_format.upper()}")
        logger.info(f"Output directory: {output_dir}")
        
        for mesh_name, input_file_path in mesh_files.items():
            logger.info(f"\nProcessing {mesh_name}...")
            
            try:
                # Create output filename
                output_filename = f"{mesh_name}.{output_format.lower()}"
                output_file_path = output_path / output_filename
                
                logger.info(f"  Input: {input_file_path}")
                logger.info(f"  Output: {output_file_path}")
                logger.info(f"  Scale: {scale_factor}")
                
                # Convert using the mesh converter utility
                success = convert_single_mesh(
                    input_file=input_file_path,
                    output_file=str(output_file_path),
                    scale_factor=scale_factor,
                    verbose=self.verbose
                )
                
                if success:
                    logger.info(f"  ✅ Successfully converted {mesh_name}")
                else:
                    logger.error(f"  ❌ Failed to convert {mesh_name}")
                
                results[mesh_name] = success
                
            except Exception as e:
                logger.error(f"  ❌ Error converting {mesh_name}: {str(e)}")
                results[mesh_name] = False
        
        # Summary
        successful = sum(results.values())
        total = len(results)
        logger.info(f"\n=== CONVERSION SUMMARY ===")
        logger.info(f"Successfully converted: {successful}/{total} meshes")
        
        if successful < total:
            logger.info("Failed conversions:")
            for mesh_name, success in results.items():
                if not success:
                    logger.info(f"  ❌ {mesh_name}")
        
        return results
    
    def run_step1_conversion(self, registration_output_dir: str,
                           postprocessing_output_dir: str,
                           target_filename: str = "DeterministicAtlas__Reconstruction__bone_trial__subject_subject_0.vtk",
                           scale_factor: float = 1.0,
                           output_format: str = "stl") -> Dict[str, bool]:
        """
        Run step 1 of post-processing: convert and scale Deformetrica outputs.
        
        Args:
            registration_output_dir: Directory containing registration results
            postprocessing_output_dir: Output directory for post-processed files
            target_filename: Specific Deformetrica output file to process
            scale_factor: Scale factor for conversion (default: 1.0)
            output_format: Output format ('stl' or 'vtk')
            
        Returns:
            Dict[str, bool]: Dictionary mapping mesh_name to success status
        """
        logger.info("=== POST-PROCESSING STEP 1: CONVERSION AND SCALING ===")
        
        # Step 1: Find all target files
        mesh_files = self.find_deformetrica_output_files(
            registration_output_dir=registration_output_dir,
            target_filename=target_filename
        )
        
        if not mesh_files:
            logger.error("❌ No mesh files found for processing")
            return {}
        
        # Step 2: Convert and scale meshes
        results = self.convert_and_scale_meshes(
            mesh_files=mesh_files,
            output_dir=postprocessing_output_dir,
            scale_factor=scale_factor,
            output_format=output_format
        )
        
        logger.info("=== POST-PROCESSING STEP 1 COMPLETED ===")
        return results


def run_postprocessing_step1(registration_output_dir: str = "./temp/registration_output",
                           postprocessing_output_dir: str = "./temp/postprocessing_output",
                           target_filename: str = "DeterministicAtlas__Reconstruction__bone_trial__subject_subject_0.vtk",
                           scale_factor: float = 1.0,
                           output_format: str = "stl",
                           verbose: bool = True) -> Dict[str, bool]:
    """
    Convenience function to run post-processing step 1.
    
    Args:
        registration_output_dir: Directory containing Deformetrica registration results
        postprocessing_output_dir: Output directory for post-processed files
        target_filename: Specific Deformetrica output file to process
        scale_factor: Scale factor for conversion (default: 1.0)
        output_format: Output format ('stl' or 'vtk')
        verbose: Enable verbose logging
        
    Returns:
        Dict[str, bool]: Dictionary mapping mesh_name to success status
    """
    pipeline = PostProcessingPipeline(verbose=verbose)
    return pipeline.run_step1_conversion(
        registration_output_dir=registration_output_dir,
        postprocessing_output_dir=postprocessing_output_dir,
        target_filename=target_filename,
        scale_factor=scale_factor,
        output_format=output_format
    )

def verify_postprocessing_outputs(postprocessing_output_dir: str,
                                expected_meshes: Optional[List[str]] = None,
                                output_format: str = "stl",
                                verbose: bool = True) -> Tuple[List[str], List[str]]:
    """
    Verify that post-processing outputs were created successfully.
    
    Args:
        postprocessing_output_dir: Directory to check for output files
        expected_meshes: List of expected mesh names (optional)
        output_format: Expected output format
        verbose: Enable verbose logging
        
    Returns:
        Tuple[List[str], List[str]]: (found_files, missing_files)
    """
    output_path = Path(postprocessing_output_dir)
    
    if not output_path.exists():
        if verbose:
            print(f"❌ Output directory does not exist: {postprocessing_output_dir}")
        return [], expected_meshes or []
    
    # Find all files with the expected format
    pattern = f"*.{output_format.lower()}"
    found_files = list(output_path.glob(pattern))
    found_names = [f.stem for f in found_files]
    
    if verbose:
        print(f"✅ Found {len(found_files)} {output_format.upper()} files in {postprocessing_output_dir}:")
        for file_path in sorted(found_files):
            file_size = file_path.stat().st_size
            print(f"  ✅ {file_path.name} ({file_size:,} bytes)")
    
    missing_files = []
    if expected_meshes:
        missing_files = [mesh for mesh in expected_meshes if mesh not in found_names]
        if verbose and missing_files:
            print(f"⚠️  Missing expected files: {missing_files}")
    
    return found_names, missing_files


if __name__ == "__main__":
    """Command line interface for post-processing."""
    
    if len(sys.argv) < 2:
        print("Usage: python postprocessing_utils.py <registration_output_dir> [postprocessing_output_dir]")
        print("\nExample:")
        print("  python postprocessing_utils.py ./temp/registration_output ./temp/postprocessing_output")
        sys.exit(1)
    
    registration_dir = sys.argv[1]
    postprocessing_dir = sys.argv[2] if len(sys.argv) > 2 else "./temp/postprocessing_output"
    
    # Run post-processing step 1
    results = run_postprocessing_step1(
        registration_output_dir=registration_dir,
        postprocessing_output_dir=postprocessing_dir,
        verbose=True
    )
    
    # Verify outputs
    found_files, missing_files = verify_postprocessing_outputs(
        postprocessing_output_dir=postprocessing_dir,
        verbose=True
    )
    
    if all(results.values()):
        print("\n✅ All meshes converted successfully!")
    else:
        print("\n❌ Some conversions failed.")
        sys.exit(1)