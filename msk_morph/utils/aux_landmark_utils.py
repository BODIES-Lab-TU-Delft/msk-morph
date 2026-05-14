#!/usr/bin/env python3
"""
Auxiliary Landmark Computation Utilities Module
Functions for computing derived landmarks using various methods.
"""

import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_midpoint(landmark1_coords: np.ndarray, landmark2_coords: np.ndarray) -> np.ndarray:
    """
    Calculate the midpoint between two landmarks.
    
    Args:
        landmark1_coords: 3D coordinates of first landmark
        landmark2_coords: 3D coordinates of second landmark
        
    Returns:
        np.ndarray: 3D coordinates of midpoint
    """
    midpoint = (landmark1_coords + landmark2_coords) / 2
    return midpoint


def fit_sphere_to_vertices(vertices: np.ndarray, vertex_indices: list) -> np.ndarray:
    """
    Fit a sphere to a set of vertices and return the center coordinates.
    Uses algebraic least squares method.
    
    Args:
        vertices: All mesh vertices (N x 3 array)
        vertex_indices: List of vertex indices to use for sphere fitting
        
    Returns:
        np.ndarray: 3D coordinates of sphere center (hip joint center)
    """
    # Extract the specified vertices
    points = vertices[vertex_indices]
    
    if len(points) < 4:
        logger.warning("⚠️ Not enough points for sphere fitting, using centroid")
        center = np.mean(points, axis=0)
        return center
    
    try:
        # Algebraic sphere fitting using linear least squares approach
        # Based on the equation: x² + y² + z² + D*x + E*y + F*z + G = 0
        # Where center = (-D/2, -E/2, -F/2) and radius = sqrt((D² + E² + F²)/4 - G)
        
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        
        # Set up the linear system for: x² + y² + z² + D*x + E*y + F*z + G = 0
        A = np.column_stack([x, y, z, np.ones(len(points))])
        b = -(x**2 + y**2 + z**2)
        
        # Solve using least squares
        params, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        D, E, F, G = params
        
        # Calculate center and radius
        center = np.array([-D/2, -E/2, -F/2])
        radius_squared = (D**2 + E**2 + F**2)/4 - G
        
        if radius_squared <= 0:
            raise ValueError("❌ Degenerate sphere - negative radius squared")
            
        radius = np.sqrt(radius_squared)
        
        logger.debug(f"✅ Sphere fit: center={center}, radius={radius:.3f}")
        
        return center
        
    except Exception as e:
        logger.warning(f"⚠️ Algebraic sphere fitting failed ({e}), using centroid fallback")
        center = np.mean(points, axis=0)
        return center


def compute_point_in_plane(L1: np.ndarray, L2: np.ndarray, L3: np.ndarray) -> np.ndarray:
    """
    Compute a point in a plane defined by four landmarks.
    
    Args:
        L1: First landmark coordinates (3D), defines the direction of the resulting vector for the calculated landmark
        L2: Second landmark coordinates (3D)
        L3: Third landmark coordinates (3D)
        
    Returns:
        np.ndarray: 3D coordinates of computed point
    """
    # Middle point between first two landmarks
    mL1L2 = compute_midpoint(L1,L2)

    # Vector between first two landmarks
    u = L1 - mL1L2
    
    # Vector between last two landmarks
    Y = L3 - mL1L2
    
    # Cross product to get normal to the plane
    n = np.cross(Y, u)
    
    # Cross product to get Z vector
    Z = np.cross(n, Y)
    
    # Normalize Z to unit length
    Z_norm = Z / np.linalg.norm(Z)

    # Calculate half the distance between mL1L2 and L1
    half_distance = np.linalg.norm(L1 - mL1L2) / 2.0

    # Scale the normalized Z vector by half_distance
    Z_scaled = Z_norm * half_distance

    # Compute output point
    output = mL1L2 + Z_scaled
    
    return output


if __name__ == "__main__":
    """Test the auxiliary landmark computation functions."""
    
    # Test midpoint calculation
    print("=== Testing Midpoint Calculation ===")
    landmark1 = np.array([1.0, 2.0, 3.0])
    landmark2 = np.array([3.0, 4.0, 5.0])
    midpoint = compute_midpoint(landmark1, landmark2)
    print(f"Landmark 1: {landmark1}")
    print(f"Landmark 2: {landmark2}")
    print(f"Midpoint: {midpoint}")
    print(f"Expected: [2.0, 3.0, 4.0]")
    
    # Test sphere fitting
    print("\n=== Testing Sphere Fitting ===")
    # Create points on a sphere with center at (1, 2, 3) and radius 5
    center_true = np.array([1.0, 2.0, 3.0])
    radius_true = 5.0
    theta = np.linspace(0, 2*np.pi, 50)
    phi = np.linspace(0, np.pi, 50)
    sphere_points = []
    for t in theta[::5]:
        for p in phi[::5]:
            x = center_true[0] + radius_true * np.sin(p) * np.cos(t)
            y = center_true[1] + radius_true * np.sin(p) * np.sin(t)
            z = center_true[2] + radius_true * np.cos(p)
            sphere_points.append([x, y, z])
    
    vertices = np.array(sphere_points)
    vertex_indices = list(range(len(vertices)))
    
    center_computed = fit_sphere_to_vertices(vertices, vertex_indices)
    print(f"True center: {center_true}")
    print(f"Computed center: {center_computed}")
    print(f"Error: {np.linalg.norm(center_computed - center_true):.6f}")
    
    # Test point in plane
    print("\n=== Testing Point in Plane ===")
    L1 = np.array([1.0, 0.0, 0.0])
    L2 = np.array([-1.0, 0.0, 0.0])
    L3 = np.array([0.0, 2.0, 0.0])
    
    result = compute_point_in_plane(L1, L2, L3)
    print(f"L1: {L1}")
    print(f"L2: {L2}")
    print(f"L3: {L3}")
    print(f"Computed point: {result}")