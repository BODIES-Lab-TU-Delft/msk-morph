#!/usr/bin/env python3
"""
VTK Mesh Utilities Module
Functions for VTK file manipulation and vertex updating.
"""

import numpy as np
from typing import Union
from pathlib import Path

def create_aligned_filename(original_path: str, suffix: str = "_aligned") -> str:
    """
    Create aligned filename by adding a suffix before the file extension.
    
    Args:
        original_path: Original file path
        suffix: Suffix to add before extension
        
    Returns:
        str: New filename with suffix
    """
    name_part, ext_part = original_path.rsplit('.', 1)
    aligned_filename = f"{name_part}{suffix}.{ext_part}"
    return aligned_filename

def update_vtk_vertices(original_vtk_path: Union[str, Path], 
                       new_vertices: np.ndarray, 
                       output_vtk_path: Union[str, Path], 
                       verbose: bool = True) -> bool:
    """
    Create a new VTK file by replacing vertices while keeping original mesh structure.
    
    Args:
        original_vtk_path: Path to original VTK file
        new_vertices: New vertex coordinates as numpy array
        output_vtk_path: Path for output VTK file
        verbose: Whether to print detailed information
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Ensure new_vertices is a numpy array
    new_vertices = np.asarray(new_vertices)

    if verbose:
        print(f"Original VTK: {original_vtk_path}")
        print(f"New vertices shape: {new_vertices.shape}")
        print(f"Output VTK: {output_vtk_path}")

    # Create output directory if needed
    output_path = Path(output_vtk_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Try PyVista first (best for VTK files)
    try:
        import pyvista as pv

        # Load original mesh
        mesh = pv.read(str(original_vtk_path))

        if verbose:
            print(f"Original mesh: {mesh.n_points} points, {mesh.n_cells} cells")

        # Verify vertex count matches
        if mesh.n_points != len(new_vertices):
            raise ValueError(f"Vertex count mismatch: original has {mesh.n_points}, new has {len(new_vertices)}")

        # Update vertices
        mesh.points = new_vertices

        # Save new VTK file
        mesh.save(str(output_vtk_path))

        if verbose:
            print("Successfully updated VTK file using PyVista")

        return True

    except ImportError:
        if verbose:
            print("PyVista not available, trying VTK library...")
    except Exception as e:
        if verbose:
            print(f"PyVista failed: {e}, trying VTK library...")

    # Try VTK library directly
    try:
        import vtk
        from vtk.util.numpy_support import numpy_to_vtk

        # Read original VTK file
        reader = vtk.vtkPolyDataReader()
        reader.SetFileName(str(original_vtk_path))
        reader.Update()

        polydata = reader.GetOutput()

        if verbose:
            print(f"Original mesh: {polydata.GetNumberOfPoints()} points, {polydata.GetNumberOfCells()} cells")

        # Verify vertex count matches
        if polydata.GetNumberOfPoints() != len(new_vertices):
            raise ValueError(f"Vertex count mismatch: original has {polydata.GetNumberOfPoints()}, new has {len(new_vertices)}")

        # Create new points
        new_points = vtk.vtkPoints()
        new_points.SetData(numpy_to_vtk(new_vertices))

        # Create new polydata with updated vertices
        new_polydata = vtk.vtkPolyData()
        new_polydata.SetPoints(new_points)

        # Copy all other data (cells, connectivity, etc.)
        if polydata.GetPolys():
            new_polydata.SetPolys(polydata.GetPolys())
        if polydata.GetVerts():
            new_polydata.SetVerts(polydata.GetVerts())
        if polydata.GetLines():
            new_polydata.SetLines(polydata.GetLines())
        if polydata.GetStrips():
            new_polydata.SetStrips(polydata.GetStrips())

        # Copy point data (if any)
        if polydata.GetPointData():
            new_polydata.GetPointData().ShallowCopy(polydata.GetPointData())

        # Copy cell data (if any)
        if polydata.GetCellData():
            new_polydata.GetCellData().ShallowCopy(polydata.GetCellData())

        # Write new VTK file
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(str(output_vtk_path))
        writer.SetInputData(new_polydata)
        writer.Write()

        if verbose:
            print("Successfully updated VTK file using VTK library")

        return True

    except ImportError:
        if verbose:
            print("VTK library not available, using manual approach...")
    except Exception as e:
        if verbose:
            print(f"VTK library failed: {e}, using manual approach...")

    # Manual approach - read and write VTK text format
    try:
        if verbose:
            print("Using manual VTK text file approach...")

        # Read original VTK file
        with open(str(original_vtk_path), 'r') as f:
            lines = f.readlines()

        # Find where points data starts and ends
        points_start = -1
        points_count = 0
        cells_start = -1

        for i, line in enumerate(lines):
            if line.strip().startswith('POINTS'):
                points_start = i
                points_count = int(line.split()[1])
                break

        for i, line in enumerate(lines[points_start+1:], points_start+1):
            if line.strip().startswith('POLYGONS') or line.strip().startswith('CELLS') or line.strip().startswith('TRIANGLE_STRIPS'):
                cells_start = i
                break

        if points_start == -1:
            raise ValueError("Could not find POINTS section in VTK file")

        if verbose:
            print(f"Found {points_count} points starting at line {points_start}")

        # Verify vertex count matches
        if points_count != len(new_vertices):
            raise ValueError(f"Vertex count mismatch: original has {points_count}, new has {len(new_vertices)}")

        # Create new lines with updated vertices
        new_lines = lines[:points_start+1].copy()  # Header + POINTS line

        # Add new vertex data
        for vertex in new_vertices:
            new_lines.append(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")

        # Add remaining lines (cells, etc.)
        if cells_start != -1:
            new_lines.extend(lines[cells_start:])

        # Write new VTK file
        with open(str(output_vtk_path), 'w') as f:
            f.writelines(new_lines)

        if verbose:
            print("Successfully updated VTK file using manual approach")

        return True

    except Exception as e:
        if verbose:
            print(f"Manual approach failed: {e}")
        return False