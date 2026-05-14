#!/usr/bin/env python3
"""
Mesh Alignment Utilities Module
Functions for anatomical and ICP-based mesh alignment.
"""

import os
import copy
import glob
import numpy as np
from typing import Dict, Tuple, Optional, Any
from .mesh_loader import load_with_fallbacks, load_multiple_meshes
from .mesh_vtk_utils import update_vtk_vertices, create_aligned_filename

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("⚠️  Open3D not available. Mesh alignment functionality will be limited.")

def create_axes_definition(anterior: str, superior: str, right: str) -> Dict[str, str]:
    """
    Helper function to create axes definition dictionary.
    
    Args:
        anterior: Axis direction for anterior (e.g., 'x', '-y', 'z', etc.)
        superior: Axis direction for superior (e.g., 'y', 'z', '-x', etc.)
        right: Axis direction for right (e.g., 'z', '-x', 'y', etc.)
    
    Returns:
        dict: Axes definition dictionary
    """
    return {
        'anterior': anterior,
        'superior': superior,
        'right': right
    }

def axis_string_to_vector(axis_str: str) -> np.ndarray:
    """Convert axis string like 'x', '-y', 'z' to unit vector."""
    if axis_str == 'x':
        return np.array([1, 0, 0])
    elif axis_str == '-x':
        return np.array([-1, 0, 0])
    elif axis_str == 'y':
        return np.array([0, 1, 0])
    elif axis_str == '-y':
        return np.array([0, -1, 0])
    elif axis_str == 'z':
        return np.array([0, 0, 1])
    elif axis_str == '-z':
        return np.array([0, 0, -1])
    else:
        raise ValueError(f"❌ Invalid axis string: {axis_str}")

def extract_points(geometry) -> np.ndarray:
    """Extract points from Open3D geometry."""
    if not OPEN3D_AVAILABLE:
        raise ImportError("❌ Open3D is required for mesh alignment")
    
    if isinstance(geometry, o3d.geometry.TriangleMesh):
        return np.asarray(geometry.vertices)
    elif isinstance(geometry, o3d.geometry.PointCloud):
        return np.asarray(geometry.points)
    else:
        raise TypeError(f"❌ Unsupported geometry type: {type(geometry)}")

def geometry_to_pointcloud(geometry):
    """Convert geometry to Open3D point cloud."""
    if not OPEN3D_AVAILABLE:
        raise ImportError("❌ Open3D is required for mesh alignment")
    
    if isinstance(geometry, o3d.geometry.PointCloud):
        return copy.deepcopy(geometry)
    elif isinstance(geometry, o3d.geometry.TriangleMesh):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.asarray(geometry.vertices))
        return pcd
    else:
        raise TypeError(f"❌ Unsupported geometry type: {type(geometry)}")

def create_rotation_matrix(template_axes_dict: Dict[str, str], 
                          target_axes_dict: Dict[str, str]) -> np.ndarray:
    """Create rotation matrix from template axes to target axes."""
    # Get unit vectors for each anatomical direction in both coordinate systems
    template_anterior = axis_string_to_vector(template_axes_dict['anterior'])
    template_superior = axis_string_to_vector(template_axes_dict['superior'])
    template_right = axis_string_to_vector(template_axes_dict['right'])
    
    target_anterior = axis_string_to_vector(target_axes_dict['anterior'])
    target_superior = axis_string_to_vector(target_axes_dict['superior'])
    target_right = axis_string_to_vector(target_axes_dict['right'])
    
    # Create transformation matrices
    # Template coordinate system matrix (columns are the basis vectors)
    template_matrix = np.column_stack([template_anterior, template_superior, template_right])
    
    # Target coordinate system matrix
    target_matrix = np.column_stack([target_anterior, target_superior, target_right])
    
    # Rotation matrix = Target * Template^(-1)
    rotation_matrix = target_matrix @ template_matrix.T
    
    return rotation_matrix

def compute_bounding_box_scaling(template, target, verbose: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute scaling factors for each axis based on bounding box dimensions.
    
    Args:
        template: Template geometry
        target: Target geometry
        verbose: Whether to print detailed information
    
    Returns:
        tuple: (scaling_factors, scaling_matrix_4x4)
            - scaling_factors: 3D vector of scaling factors [sx, sy, sz]
            - scaling_matrix_4x4: 4x4 homogeneous transformation matrix
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("✕ Open3D is required for bounding box scaling")
    
    # Extract points
    template_points = extract_points(template)
    target_points = extract_points(target)
    
    # Compute bounding boxes
    template_bbox_min = np.min(template_points, axis=0)
    template_bbox_max = np.max(template_points, axis=0)
    template_bbox_size = template_bbox_max - template_bbox_min
    
    target_bbox_min = np.min(target_points, axis=0)
    target_bbox_max = np.max(target_points, axis=0)
    target_bbox_size = target_bbox_max - target_bbox_min
    
    # Compute scaling factors (avoid division by zero)
    scaling_factors = np.zeros(3)
    for i in range(3):
        if template_bbox_size[i] > 1e-10:  # Avoid division by very small numbers
            scaling_factors[i] = target_bbox_size[i] / template_bbox_size[i]
        else:
            scaling_factors[i] = 1.0
            if verbose:
                print(f"⚠️  Template bounding box dimension {i} is very small, using scaling factor of 1.0")
    
    if verbose:
        print(f"Template bounding box size: {template_bbox_size}")
        print(f"Target bounding box size: {target_bbox_size}")
        print(f"Computed scaling factors [x, y, z]: {scaling_factors}")
    
    # Create 4x4 scaling matrix
    scaling_matrix = np.eye(4)
    scaling_matrix[0, 0] = scaling_factors[0]
    scaling_matrix[1, 1] = scaling_factors[1]
    scaling_matrix[2, 2] = scaling_factors[2]
    
    return scaling_factors, scaling_matrix

def anatomical_axes_alignment(template, target, 
                            template_axes: Dict[str, str],
                            target_axes: Dict[str, str],
                            verbose: bool = True) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """
    Align template to target using anatomical axes definitions.
    Both template and target are moved to origin, with target translation recorded.
    
    Args:
        template: Source geometry to be aligned
        target: Target geometry to align to
        template_axes: Dictionary defining template anatomical axes
        target_axes: Dictionary defining target anatomical axes
        verbose: Whether to print detailed information
    
    Returns:
        tuple: (aligned_template_at_origin, target_at_origin_points, target_original_centroid, transformation_matrix)
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("❌ Open3D is required for anatomical alignment")
    
    # Extract points
    template_points = extract_points(template)
    target_points = extract_points(target)

    if verbose:
        print(f"✅ Template points: {template_points.shape[0]}")
        print(f"✅ Target points: {target_points.shape[0]}")
        print(f"Template axes: {template_axes}")
        print(f"Target axes: {target_axes}")

    # Step 1: Calculate centroids
    template_centroid = np.mean(template_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)

    if verbose:
        print(f"Template centroid: {template_centroid}")
        print(f"Target centroid: {target_centroid}")

    # Step 2: Move both template and target to origin
    template_at_origin = copy.deepcopy(template)
    target_at_origin = copy.deepcopy(target)
    
    # Translate template to origin
    template_translation_to_origin = np.eye(4)
    template_translation_to_origin[:3, 3] = -template_centroid
    template_at_origin.transform(template_translation_to_origin)
    
    # Translate target to origin
    target_translation_to_origin = np.eye(4)
    target_translation_to_origin[:3, 3] = -target_centroid
    target_at_origin.transform(target_translation_to_origin)

    # Step 3: Create rotation matrix from anatomical axes definitions
    rotation_matrix = create_rotation_matrix(template_axes, target_axes)

    if verbose:
        # Calculate rotation angle for display
        trace = np.trace(rotation_matrix)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1)) * 180 / np.pi
        print(f"✅ Anatomical rotation angle: {angle:.2f} degrees")

    # Step 4: Apply rotation to template (now at origin)
    rotation_transformation = np.eye(4)
    rotation_transformation[:3, :3] = rotation_matrix
    template_at_origin.transform(rotation_transformation)

    # Step 5: Create combined transformation matrix for tracking
    combined_transformation = rotation_transformation @ template_translation_to_origin

    # Get target points at origin for return
    target_at_origin_points = extract_points(target_at_origin)

    if verbose:
        print("✅ Anatomical axes alignment completed - both meshes at origin")
        # Verify alignment by checking centroids
        template_aligned_points = extract_points(template_at_origin)
        template_final_centroid = np.mean(template_aligned_points, axis=0)
        target_final_centroid = np.mean(target_at_origin_points, axis=0)
        print(f"Template centroid at origin: {template_final_centroid}")
        print(f"Target centroid at origin: {target_final_centroid}")

    return template_at_origin, target_at_origin_points, target_centroid, combined_transformation

def icp_based_alignment(template, target, 
                       max_iterations: int = 50, 
                       tolerance: float = 1e-6, 
                       verbose: bool = True, 
                       save_debug_files: bool = False, 
                       debug_folder: Optional[str] = None, 
                       filename: Optional[str] = None) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """
    Align template to target using ICP (Iterative Closest Point) algorithm.
    Both template and target are already at origin.
    
    Args:
        template: Source geometry to be aligned (already at origin)
        target: Target geometry to align to (already at origin)
        max_iterations: Maximum number of ICP iterations
        tolerance: Convergence tolerance for the alignment
        verbose: Whether to print detailed information
        save_debug_files: Whether to save intermediate visualizations
        debug_folder: Folder to save debug files
        filename: Base filename for debug files
    
    Returns:
        tuple: (aligned_template_at_origin, target_at_origin_points, target_centroid, transformation_matrix)
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("❌ Open3D is required for ICP alignment")
    
    # Extract points for verification
    template_points = extract_points(template)
    target_points = extract_points(target)

    if verbose:
        print(f"✅ ICP: Template points: {template_points.shape[0]}")
        print(f"✅ ICP: Target points: {target_points.shape[0]}")
        print("ICP: Both meshes should already be at origin")

    # Verify both are at origin
    template_centroid = np.mean(template_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    
    if verbose:
        print(f"ICP: Template centroid: {template_centroid}")
        print(f"ICP: Target centroid: {target_centroid}")
    """
    # Save debug visualization if requested
    if save_debug_files and debug_folder and filename:
        base_name = filename.rsplit('.', 1)[0]
        
        # Save template before ICP (already at origin)
        template_before_icp_path = os.path.join(debug_folder, f"{base_name}_template_before_icp.vtk")
        if isinstance(template, o3d.geometry.TriangleMesh):
            template_before_icp_vertices = np.asarray(template.vertices)
        else:
            template_before_icp_vertices = np.asarray(template.points)
        
        # We need the original template path to preserve mesh structure
        original_template_path = os.path.join("./template_mesh_registration", filename)
        try:
            update_vtk_vertices(original_template_path, template_before_icp_vertices, template_before_icp_path, verbose=False)
            if verbose:
                print(f"✅ Saved template before ICP: {template_before_icp_path}")
        except Exception as e:
            if verbose:
                print(f"⚠️  Could not save debug file: {e}")
    """
    # Convert to point clouds for ICP
    template_pcd = geometry_to_pointcloud(template)
    target_pcd = geometry_to_pointcloud(target)
    
    # Calculate bounding boxes to understand the scale
    template_bbox = template_pcd.get_axis_aligned_bounding_box()
    target_bbox = target_pcd.get_axis_aligned_bounding_box()
    template_extent = template_bbox.get_extent()
    target_extent = target_bbox.get_extent()
    
    # Calculate adaptive threshold based on the size of the objects
    avg_extent = (np.mean(template_extent) + np.mean(target_extent)) / 2
    threshold = max(avg_extent * 0.05, 1.0)  # At least 1.0, or 5% of average extent
    
    if verbose:
        print(f"Template extent: {template_extent}")
        print(f"Target extent: {target_extent}")
        print(f"Adaptive threshold: {threshold:.6f}")
    
    # Estimate normals for better ICP performance
    template_pcd.estimate_normals()
    target_pcd.estimate_normals()
    
    # Check initial alignment by computing initial distance
    initial_distances = np.asarray(template_pcd.compute_point_cloud_distance(target_pcd))
    initial_mean_distance = np.mean(initial_distances)
    
    if verbose:
        print(f"Initial mean distance: {initial_mean_distance:.6f}")
        print(f"Points within threshold: {np.sum(initial_distances < threshold)} / {len(initial_distances)}")

    if verbose:
        print("Starting ICP alignment...")

    # Try multiple ICP approaches for robustness
    best_result = None
    best_fitness = -1
    
    # Approach 1: Point-to-point ICP
    try:
        reg_p2point = o3d.pipelines.registration.registration_icp(
            template_pcd, target_pcd, threshold,
            np.eye(4),  # Initial transformation
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=tolerance,
                relative_rmse=tolerance,
                max_iteration=max_iterations
            )
        )
        
        if verbose:
            print(f"✅ Point-to-point ICP fitness: {reg_p2point.fitness:.6f}, RMSE: {reg_p2point.inlier_rmse:.6f}")
        
        if reg_p2point.fitness > best_fitness:
            best_result = reg_p2point
            best_fitness = reg_p2point.fitness
            
    except Exception as e:
        if verbose:
            print(f"⚠️  Point-to-point ICP failed: {e}")
    
    # Approach 2: Point-to-plane ICP (if normals are good)
    try:
        reg_p2plane = o3d.pipelines.registration.registration_icp(
            template_pcd, target_pcd, threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=tolerance,
                relative_rmse=tolerance,
                max_iteration=max_iterations
            )
        )
        
        if verbose:
            print(f"✅ Point-to-plane ICP fitness: {reg_p2plane.fitness:.6f}, RMSE: {reg_p2plane.inlier_rmse:.6f}")
        
        if reg_p2plane.fitness > best_fitness:
            best_result = reg_p2plane
            best_fitness = reg_p2plane.fitness
            
    except Exception as e:
        if verbose:
            print(f"⚠️  Point-to-plane ICP failed: {e}")
    
    # Use the best result
    if best_result is None:
        if verbose:
            print("⚠️  ICP failed to converge, using identity transformation")
        best_result = type('MockResult', (), {
            'transformation': np.eye(4),
            'fitness': 0.0,
            'inlier_rmse': float('inf')
        })()
    
    reg_result = best_result

    if verbose:
        print(f"✅ Best ICP result - Fitness: {reg_result.fitness:.6f}, RMSE: {reg_result.inlier_rmse:.6f}")
        
        # Check if transformation is meaningful (not identity)
        transformation_magnitude = np.linalg.norm(reg_result.transformation - np.eye(4))
        print(f"Transformation magnitude: {transformation_magnitude:.6f}")
        
        if transformation_magnitude < 1e-6:
            print("⚠️  ICP transformation is very small (nearly identity)")
        
        # Show rotation angle
        rotation_matrix = reg_result.transformation[:3, :3]
        trace = np.trace(rotation_matrix)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1)) * 180 / np.pi
        translation = reg_result.transformation[:3, 3]
        print(f"Rotation angle: {angle:.2f} degrees")
        print(f"Translation: {translation}")

    # Apply ICP transformation to the template (which is already at origin)
    template_aligned = copy.deepcopy(template)
    template_aligned.transform(reg_result.transformation)
    
    # Get target points for return (target stays at origin)
    target_at_origin_points = extract_points(target)

    if verbose:
        print("✅ ICP alignment completed - template refined at origin")

    # Return: aligned template (at origin), target points (at origin), original target centroid=0, transformation
    return template_aligned, target_at_origin_points, np.array([0.0, 0.0, 0.0]), reg_result.transformation

def anatomical_mesh_alignment_workflow(template_folder: str, 
                                     target_folder: str, 
                                     file_extension: str = "*.vtk", 
                                     template_axes: Dict[str, str] = None,
                                     target_axes: Dict[str, str] = None,
                                     output_folder: str = "./aligned_output",
                                     use_icp_refinement: bool = True,
                                     use_bounding_box_scaling: bool = False,
                                     icp_max_iterations: int = 50,
                                     icp_tolerance: float = 1e-6,
                                     verbose: bool = True) -> Dict[str, Any]:
    """
    Main workflow for multi-mesh alignment using anatomical axes definitions.
    
    Args:
        template_folder: Path to folder containing template meshes
        target_folder: Path to folder containing target meshes
        file_extension: File pattern to match
        template_axes: Dictionary defining template anatomical axes
        target_axes: Dictionary defining target anatomical axes
        output_folder: Folder to save aligned meshes
        use_icp_refinement: Whether to use ICP refinement after anatomical alignment
        use_bounding_box_scaling: Whether to scale template based on bounding box comparison before ICP
        icp_max_iterations: Maximum iterations for ICP
        icp_tolerance: Convergence tolerance for ICP
        verbose: Whether to print detailed information
    
    Returns:
        Dict[str, Any]: Dictionary containing alignment results for each file
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("❌ Open3D is required for mesh alignment workflow")
    
    # Set default axes if not provided
    if template_axes is None:
        template_axes = {'anterior': 'x', 'superior': 'y', 'right': 'z'}
    if target_axes is None:
        target_axes = {'anterior': '-y', 'superior': 'z', 'right': '-x'}
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Step 1: Load all meshes from both folders
    if verbose:
        print("Step 1: Loading meshes...")
    
    template_meshes = load_multiple_meshes(template_folder, file_extension, verbose)
    target_meshes = load_multiple_meshes(target_folder, file_extension, verbose)
    
    # Verify same filenames exist in both folders
    template_files = set(template_meshes.keys())
    target_files = set(target_meshes.keys())
    common_files = template_files.intersection(target_files)
    
    if not common_files:
        raise ValueError("❌ No common filenames found between template and target folders")
    
    if verbose:
        print(f"✅ Found {len(common_files)} common files: {sorted(common_files)}")
        print(f"Template axes definition: {template_axes}")
        print(f"Target axes definition: {target_axes}")
    
    # Step 2: Perform individual alignments and save results
    if verbose:
        alignment_method = "Anatomical Axes"
        if use_bounding_box_scaling:
            alignment_method += " + Bounding Box Scaling"
        if use_icp_refinement:
            alignment_method += " + ICP"
        print(f"Step 2: Performing individual alignments using {alignment_method}...")
    
    alignment_results = {}
    
    # Create debug subfolder
    debug_folder = os.path.join(output_folder, "debug_intermediate")
    os.makedirs(debug_folder, exist_ok=True)
    
    successful_alignments = 0
    
    for filename in sorted(common_files):
        if verbose:
            print(f"\n✅ Processing {filename}...")
        
        try:
            # Get corresponding meshes
            template_individual = template_meshes[filename]
            target_individual = target_meshes[filename]
            
            # Save intermediate meshes for debugging
            base_name = filename.rsplit('.', 1)[0]  # Remove extension
            
            # Save original template (before any alignment)
            template_original_debug_path = os.path.join(debug_folder, f"{base_name}_template_original_1.vtk")
            if isinstance(template_individual, o3d.geometry.TriangleMesh):
                template_original_vertices = np.asarray(template_individual.vertices)
            else:
                template_original_vertices = np.asarray(template_individual.points)
            
            original_template_path = os.path.join(template_folder, filename)
            update_vtk_vertices(original_template_path, template_original_vertices, template_original_debug_path, verbose=False)
            
            # Save original target (before any alignment)
            target_original_debug_path = os.path.join(debug_folder, f"{base_name}_target_original_2.vtk")
            if isinstance(target_individual, o3d.geometry.TriangleMesh):
                target_original_vertices = np.asarray(target_individual.vertices)
            else:
                target_original_vertices = np.asarray(target_individual.points)
            
            original_target_path = os.path.join(target_folder, filename)
            update_vtk_vertices(original_target_path, target_original_vertices, target_original_debug_path, verbose=False)
            
            # Step 2a: Perform anatomical axes alignment - both meshes moved to origin
            template_after_anatomical, target_at_origin, target_original_centroid, anatomical_transformation = anatomical_axes_alignment(
                template_individual, target_individual,
                template_axes=template_axes,
                target_axes=target_axes,
                verbose=verbose if filename == sorted(common_files)[0] else False
            )
            
            # Create target geometry at origin for ICP (if needed)
            target_at_origin_geometry = copy.deepcopy(target_individual)
            target_translation_to_origin = np.eye(4)
            target_translation_to_origin[:3, 3] = -target_original_centroid
            target_at_origin_geometry.transform(target_translation_to_origin)
            
            # Save target at origin
            target_at_origin_debug_path = os.path.join(debug_folder, f"{base_name}_target_at_origin_3.vtk")
            if isinstance(target_at_origin_geometry, o3d.geometry.TriangleMesh):
                target_at_origin_vertices = np.asarray(target_at_origin_geometry.vertices)
            else:
                target_at_origin_vertices = np.asarray(target_at_origin_geometry.points)
            update_vtk_vertices(original_target_path, target_at_origin_vertices, target_at_origin_debug_path, verbose=False)
            
            # Save template after anatomical alignment (at origin, before scaling (and ICP))
            template_after_anatomical_debug_path = os.path.join(debug_folder, f"{base_name}_template_axes_aligned_at_origin_4.vtk")
            if isinstance(template_after_anatomical, o3d.geometry.TriangleMesh):
                template_after_anatomical_vertices = np.asarray(template_after_anatomical.vertices)
            else:
                template_after_anatomical_vertices = np.asarray(template_after_anatomical.points)
            update_vtk_vertices(original_template_path, template_after_anatomical_vertices, template_after_anatomical_debug_path, verbose=False)

            # Step 2b: Optionally perform bounding box scaling (both at origin)
            if use_bounding_box_scaling:
                # Compute scaling factors based on bounding boxes
                scaling_factors, scaling_matrix = compute_bounding_box_scaling(
                    template_after_anatomical, target_at_origin_geometry,
                    verbose=verbose if filename == sorted(common_files)[0] else False
                )
                
                # Apply scaling to template
                template_after_scaling = copy.deepcopy(template_after_anatomical)
                template_after_scaling.transform(scaling_matrix)
                
                # Save template after scaling (at origin, before ICP)
                template_scaled_debug_path = os.path.join(debug_folder, f"{base_name}_template_scaled_at_origin_5.vtk")
                if isinstance(template_after_scaling, o3d.geometry.TriangleMesh):
                    template_scaled_vertices = np.asarray(template_after_scaling.vertices)
                else:
                    template_scaled_vertices = np.asarray(template_after_scaling.points)
                update_vtk_vertices(original_template_path, template_scaled_vertices, template_scaled_debug_path, verbose=False)
                
                if verbose and filename == sorted(common_files)[0]:
                    print(f"✅ Applied bounding box scaling: {scaling_factors}")
            else:
                # No scaling applied
                template_after_scaling = template_after_anatomical
                scaling_matrix = np.eye(4)
            
            # Step 2c: Optionally perform ICP refinement (both at origin)
            if use_icp_refinement:
                template_final_at_origin, target_final_points, _, icp_transformation = icp_based_alignment(
                    template_after_scaling, target_at_origin_geometry,
                    max_iterations=icp_max_iterations,
                    tolerance=icp_tolerance,
                    verbose=verbose if filename == sorted(common_files)[0] else False,
                    save_debug_files=True,
                    debug_folder=debug_folder,
                    filename=filename
                )
                
                # The ICP transformation is applied to template already at origin (and possibly scaled)
                combined_transformation = icp_transformation @ scaling_matrix @ anatomical_transformation

                # Save template after anatomical alignment (at origin, before scaling (and ICP))
                template_after_icp_debug_path = os.path.join(debug_folder, f"{base_name}_template_after_icp_at_origin_6.vtk")
                if isinstance(template_final_at_origin, o3d.geometry.TriangleMesh):
                    template_final_at_origin_vertices = np.asarray(template_final_at_origin.vertices)
                else:
                    template_final_at_origin_vertices = np.asarray(template_final_at_origin.points)
                update_vtk_vertices(original_template_path, template_final_at_origin_vertices, template_after_icp_debug_path, verbose=False)
                
            else:
                # Use only anatomical alignment (both at origin, possibly with scaling)
                template_final_at_origin = template_after_scaling
                target_final_points = target_at_origin
                combined_transformation = scaling_matrix @ anatomical_transformation
            
            # Step 3: Transform aligned template back to original target location
            template_at_target_location = copy.deepcopy(template_final_at_origin)
            
            # Create transformation to move from origin back to target's original location
            translation_back_to_target = np.eye(4)
            translation_back_to_target[:3, 3] = target_original_centroid
            
            template_at_target_location.transform(translation_back_to_target)
            
            if verbose and filename == sorted(common_files)[0]:
                print(f"✅ Transformed aligned template back to target's original location: {target_original_centroid}")
            
            # Save template at target's original location (MAIN OUTPUT)
            template_output_path = os.path.join(output_folder, create_aligned_filename(filename, "_template_aligned"))
            if isinstance(template_at_target_location, o3d.geometry.TriangleMesh):
                template_at_target_vertices = np.asarray(template_at_target_location.vertices)
            else:
                template_at_target_vertices = np.asarray(template_at_target_location.points)
            
            update_vtk_vertices(original_template_path, template_at_target_vertices, template_output_path, verbose=False)
            
            # Save original target mesh (reference output)
            target_original_output_path = os.path.join(output_folder, create_aligned_filename(filename, "_target_original"))
            update_vtk_vertices(original_target_path, target_original_vertices, target_original_output_path, verbose=False)
            
            # Store results
            alignment_results[filename] = {
                'anatomical_transformation': anatomical_transformation,
                'combined_transformation': combined_transformation,
                'target_original_centroid': target_original_centroid,
                'template_aligned_path': template_output_path,
                'target_original_path': target_original_output_path,
                'alignment_method': alignment_method,
                'template_axes': template_axes.copy(),
                'target_axes': target_axes.copy(),
                'icp_used': use_icp_refinement,
                'success': True
            }
            
            successful_alignments += 1
            
            if verbose:
                print(f"✅ Saved aligned meshes for {filename} using {alignment_method}")
        
        except Exception as e:
            if verbose:
                print(f"❌ Failed to process {filename}: {e}")
            
            # Store failed result
            alignment_results[filename] = {
                'success': False,
                'error': str(e),
                'alignment_method': alignment_method
            }
    
    if verbose:
        print(f"\n=== MESH ALIGNMENT SUMMARY ===")
        print(f"Successfully aligned: {successful_alignments}/{len(common_files)} mesh pairs")
        print(f"Alignment method: {alignment_method}")
        print(f"Results saved to: {output_folder}")
        print(f"Debug intermediate files saved to: {debug_folder}")
        print(f"\nAxes definitions used:")
        print(f"  Template: {template_axes}")
        print(f"  Target: {target_axes}")
        
        if successful_alignments < len(common_files):
            print(f"\n⚠️  Failed alignments:")
            for filename, result in alignment_results.items():
                if not result.get('success', False):
                    print(f"  ❌ {filename}: {result.get('error', 'Unknown error')}")
    
    return alignment_results