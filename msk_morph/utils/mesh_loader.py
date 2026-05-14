#!/usr/bin/env python3
"""
Mesh Loading Utilities Module
Functions for loading meshes with multiple fallback methods.
"""

import os
import glob
import numpy as np
from typing import Dict, Any, Optional

try:
    import open3d as o3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False

def load_mesh(file_path: str, verbose: bool = True) -> Any:
    """
    Load a mesh using Open3D.
    
    Args:
        file_path: Path to the mesh file
        verbose: Whether to print detailed information
        
    Returns:
        Open3D geometry object
    """
    if not OPEN3D_AVAILABLE:
        raise ImportError("❌ Open3D is required for mesh loading")
    
    try:
        if file_path.lower().endswith('.vtk'):
            # Try to load as triangle mesh first
            mesh = o3d.io.read_triangle_mesh(file_path)
            if len(mesh.vertices) > 0:
                if verbose:
                    print(f"✅ Loaded triangle mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
                return mesh
            
            # If no triangles, try as point cloud
            pcd = o3d.io.read_point_cloud(file_path)
            if len(pcd.points) > 0:
                if verbose:
                    print(f"✅ Loaded point cloud: {len(pcd.points)} points")
                return pcd
        else:
            # For other formats, use generic loader
            mesh = o3d.io.read_triangle_mesh(file_path)
            if len(mesh.vertices) > 0:
                if verbose:
                    print(f"✅ Loaded triangle mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")
                return mesh
    except Exception as e:
        if verbose:
            print(f"❌ Open3D loading failed: {e}")
        
    raise ValueError(f"❌ Failed to load mesh from {file_path}")

def load_with_fallbacks(file_path: str, verbose: bool = True) -> Any:
    """
    Load a mesh or point cloud with multiple fallback methods.

    Args:
        file_path: Path to the file
        verbose: Whether to print detailed information

    Returns:
        Data object (Open3D geometry or NumPy array)
    """
    # First try Open3D loader
    try:
        data = load_mesh(file_path, verbose=verbose)
        if OPEN3D_AVAILABLE:
            if isinstance(data, o3d.geometry.TriangleMesh) and len(data.vertices) > 0:
                return data
            if isinstance(data, o3d.geometry.PointCloud) and len(data.points) > 0:
                return data
    except Exception as e:
        if verbose:
            print(f"⚠️  Standard loading failed: {e}")

    # If Open3D fails, try numpy direct load for VTK
    try:
        from vtk.util.numpy_support import vtk_to_numpy
        import vtk

        if verbose:
            print("⚠️  Attempting to load with VTK library...")

        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(file_path)
        reader.Update()

        polydata = reader.GetOutput()
        points = polydata.GetPoints()

        if points and points.GetNumberOfPoints() > 0:
            points_array = vtk_to_numpy(points.GetData())

            if verbose:
                print(f"✅ Loaded {points_array.shape[0]} points using VTK library")

            if OPEN3D_AVAILABLE:
                # Create an Open3D point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points_array)
                return pcd
            else:
                return points_array
    except ImportError:
        if verbose:
            print("⚠️  VTK library not available for fallback loading")
    except Exception as e:
        if verbose:
            print(f"⚠️  VTK direct loading failed: {e}")

    # Last resort - try other libraries or methods
    try:
        import meshio
        if verbose:
            print("⚠️  Attempting to load with meshio...")

        mesh = meshio.read(file_path)

        if len(mesh.points) > 0:
            if verbose:
                print(f"✅ Loaded {len(mesh.points)} points using meshio")

            if OPEN3D_AVAILABLE:
                # Convert to Open3D point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(mesh.points)

                # Check if we have triangular cells to make a mesh
                if len(mesh.cells) > 0:
                    for cell_block in mesh.cells:
                        if cell_block.type == "triangle":
                            o3d_mesh = o3d.geometry.TriangleMesh()
                            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.points)
                            o3d_mesh.triangles = o3d.utility.Vector3iVector(cell_block.data)
                            if verbose:
                                print(f"✅ Converted to triangle mesh with {len(cell_block.data)} faces")
                            return o3d_mesh

                return pcd
            else:
                return mesh.points
    except ImportError:
        if verbose:
            print("⚠️  meshio library not available for fallback loading")
    except Exception as e:
        if verbose:
            print(f"⚠️  meshio loading failed: {e}")

    raise ValueError(f"❌ All loading methods failed for {file_path}")

def load_multiple_meshes(folder_path: str, 
                        file_extension: str = "*.vtk", 
                        verbose: bool = True) -> Dict[str, Any]:
    """
    Load multiple meshes from a folder.
    
    Args:
        folder_path: Path to folder containing meshes
        file_extension: File pattern to match (e.g., "*.vtk", "*.ply")
        verbose: Whether to print detailed information
    
    Returns:
        dict: Dictionary with filename as key and loaded mesh as value
    """
    meshes = {}
    file_pattern = os.path.join(folder_path, file_extension)
    file_paths = glob.glob(file_pattern)
    
    if not file_paths:
        raise ValueError(f"❌ No files found matching pattern: {file_pattern}")
    
    if verbose:
        print(f"✅ Found {len(file_paths)} files in {folder_path}")
    
    successful_loads = 0
    for file_path in sorted(file_paths):
        filename = os.path.basename(file_path)
        try:
            mesh = load_with_fallbacks(file_path, verbose=False)
            meshes[filename] = mesh
            successful_loads += 1
            if verbose:
                print(f"  ✅ Loaded: {filename}")
        except Exception as e:
            if verbose:
                print(f"  ❌ Failed to load {filename}: {e}")
    
    if verbose:
        if successful_loads == len(file_paths):
            print(f"✅ Successfully loaded all {successful_loads} meshes")
        elif successful_loads > 0:
            print(f"⚠️  Loaded {successful_loads}/{len(file_paths)} meshes successfully")
        else:
            print(f"❌ Failed to load any meshes")
    
    return meshes

def validate_loaded_meshes(meshes: Dict[str, Any], verbose: bool = True) -> Dict[str, bool]:
    """
    Validate that loaded meshes contain valid data.
    
    Args:
        meshes: Dictionary of loaded meshes
        verbose: Whether to print detailed information
        
    Returns:
        Dict[str, bool]: Dictionary mapping filename to validation status
    """
    if verbose:
        print("=== Mesh Validation ===")
    
    validation_results = {}
    
    for filename, mesh in meshes.items():
        try:
            is_valid = False
            
            if OPEN3D_AVAILABLE:
                if isinstance(mesh, o3d.geometry.TriangleMesh):
                    vertex_count = len(mesh.vertices)
                    triangle_count = len(mesh.triangles)
                    is_valid = vertex_count > 0
                    if verbose:
                        status = "✅" if is_valid else "❌"
                        print(f"  {status} {filename}: Triangle mesh - {vertex_count} vertices, {triangle_count} triangles")
                
                elif isinstance(mesh, o3d.geometry.PointCloud):
                    point_count = len(mesh.points)
                    is_valid = point_count > 0
                    if verbose:
                        status = "✅" if is_valid else "❌"
                        print(f"  {status} {filename}: Point cloud - {point_count} points")
                
                else:
                    if verbose:
                        print(f"  ⚠️  {filename}: Unknown Open3D geometry type")
            
            elif isinstance(mesh, np.ndarray):
                point_count = mesh.shape[0] if len(mesh.shape) >= 2 else 0
                is_valid = point_count > 0
                if verbose:
                    status = "✅" if is_valid else "❌"
                    print(f"  {status} {filename}: NumPy array - {point_count} points")
            
            else:
                if verbose:
                    print(f"  ⚠️  {filename}: Unknown mesh type: {type(mesh)}")
            
            validation_results[filename] = is_valid
            
        except Exception as e:
            validation_results[filename] = False
            if verbose:
                print(f"  ❌ {filename}: Validation error - {e}")
    
    # Summary
    if verbose:
        valid_count = sum(validation_results.values())
        total_count = len(validation_results)
        
        if valid_count == total_count:
            print(f"✅ All {total_count} meshes validated successfully")
        elif valid_count > 0:
            print(f"⚠️  {valid_count}/{total_count} meshes validated successfully")
        else:
            print(f"❌ No meshes validated successfully")
    
    return validation_results

def get_mesh_info(mesh: Any, verbose: bool = True) -> Dict[str, Any]:
    """
    Get detailed information about a loaded mesh.
    
    Args:
        mesh: Loaded mesh object
        verbose: Whether to print detailed information
        
    Returns:
        Dict[str, Any]: Dictionary containing mesh information
    """
    info = {
        'type': 'unknown',
        'vertex_count': 0,
        'face_count': 0,
        'has_normals': False,
        'has_colors': False,
        'bounding_box': None
    }
    
    try:
        if OPEN3D_AVAILABLE:
            if isinstance(mesh, o3d.geometry.TriangleMesh):
                info['type'] = 'triangle_mesh'
                info['vertex_count'] = len(mesh.vertices)
                info['face_count'] = len(mesh.triangles)
                info['has_normals'] = len(mesh.vertex_normals) > 0
                info['has_colors'] = len(mesh.vertex_colors) > 0
                
                if info['vertex_count'] > 0:
                    vertices = np.asarray(mesh.vertices)
                    info['bounding_box'] = {
                        'min': vertices.min(axis=0).tolist(),
                        'max': vertices.max(axis=0).tolist(),
                        'center': vertices.mean(axis=0).tolist(),
                        'extent': (vertices.max(axis=0) - vertices.min(axis=0)).tolist()
                    }
            
            elif isinstance(mesh, o3d.geometry.PointCloud):
                info['type'] = 'point_cloud'
                info['vertex_count'] = len(mesh.points)
                info['has_normals'] = len(mesh.normals) > 0
                info['has_colors'] = len(mesh.colors) > 0
                
                if info['vertex_count'] > 0:
                    points = np.asarray(mesh.points)
                    info['bounding_box'] = {
                        'min': points.min(axis=0).tolist(),
                        'max': points.max(axis=0).tolist(),
                        'center': points.mean(axis=0).tolist(),
                        'extent': (points.max(axis=0) - points.min(axis=0)).tolist()
                    }
        
        elif isinstance(mesh, np.ndarray):
            info['type'] = 'numpy_array'
            info['vertex_count'] = mesh.shape[0] if len(mesh.shape) >= 2 else 0
            
            if info['vertex_count'] > 0 and mesh.shape[1] >= 3:
                info['bounding_box'] = {
                    'min': mesh.min(axis=0)[:3].tolist(),
                    'max': mesh.max(axis=0)[:3].tolist(),
                    'center': mesh.mean(axis=0)[:3].tolist(),
                    'extent': (mesh.max(axis=0)[:3] - mesh.min(axis=0)[:3]).tolist()
                }
        
        if verbose:
            print(f"Mesh info:")
            print(f"  Type: {info['type']}")
            print(f"  Vertices: {info['vertex_count']}")
            if info['face_count'] > 0:
                print(f"  Faces: {info['face_count']}")
            if info['has_normals']:
                print(f"  ✅ Has normals")
            if info['has_colors']:
                print(f"  ✅ Has colors")
            if info['bounding_box']:
                bbox = info['bounding_box']
                print(f"  Bounding box: {bbox['extent']}")
                print(f"  Center: {bbox['center']}")
    
    except Exception as e:
        if verbose:
            print(f"❌ Error getting mesh info: {e}")
    
    return info

if __name__ == "__main__":
    """Command line interface for mesh loading utilities."""
    
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mesh_loader.py <command> [args...]")
        print("\nCommands:")
        print("  load <file_path>                    - Load single mesh file")
        print("  batch <folder_path> [pattern]       - Load multiple meshes")
        print("  validate <folder_path> [pattern]    - Load and validate meshes")
        print("  info <file_path>                    - Get detailed mesh information")
        print("\nExamples:")
        print("  python mesh_loader.py load mesh.vtk")
        print("  python mesh_loader.py batch ./meshes *.vtk")
        print("  python mesh_loader.py validate ./meshes")
        print("  python mesh_loader.py info mesh.vtk")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "load":
        if len(sys.argv) < 3:
            print("❌ Usage: load <file_path>")
            sys.exit(1)
        
        file_path = sys.argv[2]
        
        try:
            mesh = load_with_fallbacks(file_path, verbose=True)
            info = get_mesh_info(mesh, verbose=True)
            print(f"✅ Successfully loaded mesh from {file_path}")
        except Exception as e:
            print(f"❌ Failed to load mesh: {e}")
            sys.exit(1)
    
    elif command == "batch":
        if len(sys.argv) < 3:
            print("❌ Usage: batch <folder_path> [pattern]")
            sys.exit(1)
        
        folder_path = sys.argv[2]
        pattern = sys.argv[3] if len(sys.argv) > 3 else "*.vtk"
        
        try:
            meshes = load_multiple_meshes(folder_path, pattern, verbose=True)
            if meshes:
                print(f"✅ Successfully loaded {len(meshes)} meshes")
            else:
                print(f"❌ No meshes loaded")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Failed to load meshes: {e}")
            sys.exit(1)
    
    elif command == "validate":
        if len(sys.argv) < 3:
            print("❌ Usage: validate <folder_path> [pattern]")
            sys.exit(1)
        
        folder_path = sys.argv[2]
        pattern = sys.argv[3] if len(sys.argv) > 3 else "*.vtk"
        
        try:
            meshes = load_multiple_meshes(folder_path, pattern, verbose=True)
            validation_results = validate_loaded_meshes(meshes, verbose=True)
            
            if all(validation_results.values()):
                print(f"✅ All meshes validated successfully")
            else:
                print(f"⚠️  Some meshes failed validation")
                sys.exit(1)
        except Exception as e:
            print(f"❌ Failed to validate meshes: {e}")
            sys.exit(1)
    
    elif command == "info":
        if len(sys.argv) < 3:
            print("❌ Usage: info <file_path>")
            sys.exit(1)
        
        file_path = sys.argv[2]
        
        try:
            mesh = load_with_fallbacks(file_path, verbose=False)
            info = get_mesh_info(mesh, verbose=True)
            print(f"✅ Mesh information retrieved successfully")
        except Exception as e:
            print(f"❌ Failed to get mesh info: {e}")
            sys.exit(1)
    
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)