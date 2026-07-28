#!/usr/bin/env python3
"""
Landmark Processing Utilities Module
Functions for processing anatomical landmarks from settings file configuration.
"""

import os
import csv
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

# Import auxiliary landmark computation functions
from utils.aux_landmark_utils import compute_midpoint, fit_sphere_to_vertices, compute_point_in_plane

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import pymeshlab as ml
    PYMESHLAB_AVAILABLE = True
except ImportError:
    PYMESHLAB_AVAILABLE = False
    logger.warning("⚠️ PyMeshLab not available. Mesh loading will be limited.")

def find_mesh_file(directory: str, mesh_name: str) -> Optional[str]:
    """
    Find a mesh file in the given directory, checking for STL and OBJ formats.
    
    Args:
        directory: Directory to search
        mesh_name: Base name of the mesh (without extension)
        
    Returns:
        Full path to the mesh file if found, None otherwise
    """
    from pathlib import Path
    
    # Check for different file extensions
    extensions = ['.stl', '.STL', '.obj', '.OBJ']
    
    for ext in extensions:
        mesh_path = Path(directory) / f"{mesh_name}{ext}"
        if mesh_path.exists():
            return str(mesh_path)
    
    return None

class LandmarkProcessor:
    """
    Process anatomical landmarks based on settings file configuration.
    """
    
    def __init__(self, settings_file_path: str, template_geom_dir: str, verbose: bool = True):
        """
        Initialize the landmark processor.
        
        Args:
            settings_file_path: Path to YAML settings file
            template_geom_dir: Directory containing template geometry files
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        if not verbose:
            logger.setLevel(logging.WARNING)
        
        self.template_geom_dir = template_geom_dir
        
        # Load settings from YAML file
        self.settings = self._load_settings(settings_file_path)
        
        # Cache for loaded meshes
        self.mesh_cache = {}
    
    def _load_settings(self, settings_file_path: str) -> Dict:
        """Load settings from YAML file."""
        try:
            with open(settings_file_path, 'r') as f:
                settings = yaml.safe_load(f)
            
            if self.verbose:
                logger.info(f"✅ Loaded settings from: {settings_file_path}")
                logger.info(f"   Found {len(settings['landmarks'])} landmark definitions")
            
            return settings
            
        except Exception as e:
            logger.error(f"❌ Failed to load settings file: {e}")
            raise
    
    def load_mesh_with_pymeshlab(self, mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load mesh using PyMeshLab and return vertices and faces.
        
        Args:
            mesh_path: Path to mesh file
            
        Returns:
            Tuple of (vertices, faces) as numpy arrays
        """
        if not PYMESHLAB_AVAILABLE:
            raise ImportError("❌ PyMeshLab is required for mesh loading")
        
        # Check cache first
        if mesh_path in self.mesh_cache:
            return self.mesh_cache[mesh_path]
        
        ms = ml.MeshSet()
        ms.load_new_mesh(str(mesh_path))
        mesh = ms.current_mesh()
        
        vertices = mesh.vertex_matrix()
        faces = mesh.face_matrix()
        
        # Cache the loaded mesh
        self.mesh_cache[mesh_path] = (vertices, faces)
        
        return vertices, faces
    
    def compute_barycentric_coordinates(self, point: np.ndarray, mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute barycentric coordinates for a point on a mesh.
        Finds the closest polygon face and computes weights.
        
        Args:
            point: 3D coordinates of the point
            mesh_path: Path to the mesh file
            
        Returns:
            Tuple of (vertex_indices, barycentric_weights)
        """
        vertices, faces = self.load_mesh_with_pymeshlab(mesh_path)
        
        # Find closest face
        face_indices, bary_weights = self._find_closest_face_and_weights(point, vertices, faces)
        
        return face_indices, bary_weights
    
    def _find_closest_face_and_weights(self, point: np.ndarray, vertices: np.ndarray, 
                                      faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the closest polygon face to a point and compute barycentric weights.
        Based on station2polygon.py logic.
        """
        min_dist = float('inf')
        closest_face_indices = None
        closest_proj = None
        
        for face in faces:
            # Get valid vertex indices (remove NaN values)
            face_indices = face[~np.isnan(face)].astype(int)
            polygon = vertices[face_indices]
            
            # Find closest point on this polygon
            proj = self._closest_point_on_polygon(point, polygon)
            dist = np.linalg.norm(point - proj)
            
            if dist < min_dist:
                min_dist = dist
                closest_proj = proj
                closest_face_indices = face_indices
        
        # Compute barycentric weights
        weights = self._compute_barycentric_weights(closest_proj, closest_face_indices, vertices)
        
        return closest_face_indices, weights
    
    def _closest_point_on_polygon(self, point: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        """Find closest point on a polygon to a given point."""
        # Project point onto polygon plane
        normal = np.cross(polygon[1] - polygon[0], polygon[2] - polygon[0])
        normal /= np.linalg.norm(normal)
        v0 = polygon[0]
        to_point = point - v0
        dist_to_plane = np.dot(to_point, normal)
        proj_point = point - dist_to_plane * normal
        
        # Check if projected point is inside polygon
        u = polygon[1] - v0
        u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        
        def to_2d(p):
            return np.array([np.dot(p - v0, u), np.dot(p - v0, v)])
        
        poly_2d = [to_2d(p) for p in polygon]
        point_2d = to_2d(proj_point)
        
        if self._is_point_in_polygon(point_2d, poly_2d):
            return proj_point
        
        # If outside, find closest edge
        closest = None
        min_dist = np.inf
        for i in range(len(polygon)):
            a = polygon[i]
            b = polygon[(i + 1) % len(polygon)]
            cp = self._closest_point_on_segment(proj_point, a, b)
            dist = np.linalg.norm(cp - point)
            if dist < min_dist:
                closest = cp
                min_dist = dist
        
        return closest
    
    def _is_point_in_polygon(self, point: np.ndarray, polygon: List) -> bool:
        """Check if 2D point is inside 2D polygon."""
        x, y = point
        inside = False
        n = len(polygon)
        px, py = zip(*polygon)
        j = n - 1
        for i in range(n):
            if ((py[i] > y) != (py[j] > y)) and \
                    (x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i]) + px[i]):
                inside = not inside
            j = i
        return inside
    
    def _closest_point_on_segment(self, p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Find closest point on line segment to a given point."""
        ab = b - a
        t = np.dot(p - a, ab) / np.dot(ab, ab)
        t = np.clip(t, 0, 1)
        return a + t * ab
    
    def _compute_barycentric_weights(self, proj_point: np.ndarray, face_indices: np.ndarray, 
                                    vertices: np.ndarray) -> np.ndarray:
        """Compute barycentric weights for a point on a face."""
        verts = vertices[face_indices]
        
        # Solve: verts.T @ weights = proj_point
        # Add an extra row for affine weights sum-to-one
        A = np.vstack((verts.T, np.ones(len(verts))))
        x = np.append(proj_point, 1)
        
        weights, residuals, rank, s = np.linalg.lstsq(A, x, rcond=None)
        return weights
    
    def calculate_barycentric_position(self, vertices: np.ndarray, 
                                      vertex_indices: np.ndarray, 
                                      bary_weights: np.ndarray) -> np.ndarray:
        """
        Calculate 3D position from barycentric coordinates.
        
        Args:
            vertices: All mesh vertices.
            vertex_indices: Indices of vertices forming the face
            bary_weights: Barycentric weights
            
        Returns:
            3D position as numpy array
        """
        # Get the vertices of the face
        face_vertices = vertices[vertex_indices]
        
        # Calculate weighted position
        position = np.sum(bary_weights[:, None] * face_vertices, axis=0)
        
        return position
    
    def process_landmarks_from_settings(self, mesh_directory: str) -> Dict[str, np.ndarray]:
        """
        Process all landmarks based on settings file configuration.
        
        Args:
            mesh_directory: Directory containing target mesh files (STL format)
            
        Returns:
            Dictionary mapping landmark names to their 3D coordinates
        """
        if self.verbose:
            logger.info("=== PROCESSING LANDMARKS FROM SETTINGS ===")
            logger.info(f"Template geometry directory: {self.template_geom_dir}")
            logger.info(f"Target mesh directory: {mesh_directory}")
        
        # Dictionary to store all computed landmarks
        all_landmarks = {}
        
        # Step 1: Process all MEASURED landmarks first
        if self.verbose:
            logger.info("\n=== STEP 1: Processing Measured Landmarks ===")
        
        for landmark_def in self.settings['landmarks']:
            if landmark_def.get('measured', False):
                landmark_name = landmark_def['name']
                mesh_name = landmark_def['measurement']['mesh']
                template_location = np.array(landmark_def['measurement']['location'])
                
                if self.verbose:
                    logger.info(f"\n✅ Processing measured landmark: {landmark_name}")
                    logger.info(f"   Mesh: {mesh_name}")
                    logger.info(f"   Template location: {template_location}")
                
                # Compute barycentric coordinates in template mesh
                template_mesh_path = find_mesh_file(self.template_geom_dir, mesh_name)
                if template_mesh_path is None:
                    raise FileNotFoundError(f"Mesh file not found: {mesh_name} in {self.template_geom_dir}")
                vertex_indices, bary_weights = self.compute_barycentric_coordinates(
                    template_location, template_mesh_path
                )
                
                # Apply barycentric coordinates to target mesh
                target_mesh_path = find_mesh_file(mesh_directory, mesh_name)
                if target_mesh_path is None:
                    raise FileNotFoundError(f"Mesh file not found: {mesh_name} in {mesh_directory}")
                                
                if os.path.exists(target_mesh_path):
                    target_vertices, _ = self.load_mesh_with_pymeshlab(target_mesh_path)
                    position = self.calculate_barycentric_position(
                        target_vertices, vertex_indices, bary_weights
                    )
                    
                    all_landmarks[landmark_name] = position
                    
                    if self.verbose:
                        logger.info(f"   ✅ Computed position: {position}")
                else:
                    logger.warning(f"   ⚠️ Target mesh not found: {target_mesh_path}")
        
        # Step 2: Process all COMPUTED landmarks in order
        if self.verbose:
            logger.info("\n=== STEP 2: Processing Computed Landmarks ===")
        
        for landmark_def in self.settings['landmarks']:
            if not landmark_def.get('measured', False):
                landmark_name = landmark_def['name']
                computation = landmark_def.get('computation', {})
                method = computation.get('method')
                
                if self.verbose:
                    logger.info(f"\n✅ Processing computed landmark: {landmark_name}")
                    logger.info(f"   Method: {method}")
                
                try:
                    if method == 'midpoint':
                        # Get the two landmarks to compute midpoint
                        landmark_names = computation['landmarks_midpoint']
                        L1 = all_landmarks[landmark_names[0]]
                        L2 = all_landmarks[landmark_names[1]]
                        
                        position = compute_midpoint(L1, L2)
                        all_landmarks[landmark_name] = position
                        
                        if self.verbose:
                            logger.info(f"   Input landmarks: {landmark_names}")
                            logger.info(f"   ✅ Computed midpoint: {position}")
                    
                    elif method == 'sphere':
                        # Fit sphere to vertices
                        mesh_name = computation['mesh']
                        vertex_indices = computation['vertices_list']
                        
                        target_mesh_path = find_mesh_file(mesh_directory, mesh_name)
                        if target_mesh_path is None:
                            logger.warning(f"⚠️ Target mesh not found: {mesh_name}")
                            continue
                        target_vertices, _ = self.load_mesh_with_pymeshlab(target_mesh_path)
                        
                        position = fit_sphere_to_vertices(target_vertices, vertex_indices)
                        all_landmarks[landmark_name] = position
                        
                        if self.verbose:
                            logger.info(f"   Mesh: {mesh_name}")
                            logger.info(f"   Number of vertices: {len(vertex_indices)}")
                            logger.info(f"   ✅ Computed sphere center: {position}")
                    
                    elif method == 'point_in_plane':
                        # Compute point in plane from 4 landmarks
                        landmark_names = computation['landmarks_plane']
                        L1 = all_landmarks[landmark_names[0]]
                        L2 = all_landmarks[landmark_names[1]]
                        L3 = all_landmarks[landmark_names[2]]
                        
                        position = compute_point_in_plane(L1, L2, L3)
                        all_landmarks[landmark_name] = position
                        
                        if self.verbose:
                            logger.info(f"   Input landmarks: {landmark_names}")
                            logger.info(f"   ✅ Computed point in plane: {position}")
                    
                    elif method == 'copy':
                        # Copy from another landmark
                        copied_landmark = computation['copied_landmark']
                        position = all_landmarks[copied_landmark].copy()
                        all_landmarks[landmark_name] = position
                        
                        if self.verbose:
                            logger.info(f"   Copied from: {copied_landmark}")
                            logger.info(f"   ✅ Copied position: {position}")
                    
                    else:
                        logger.warning(f"   ⚠️ Unknown computation method: {method}")
                
                except KeyError as e:
                    logger.error(f"   ❌ Missing required landmark for computation: {e}")
                except Exception as e:
                    logger.error(f"   ❌ Error computing landmark: {e}")
        
        if self.verbose:
            logger.info(f"\n✅ Total landmarks processed: {len(all_landmarks)}")
        
        return all_landmarks
    
    def create_landmark_csv_files(self, all_landmarks: Dict[str, np.ndarray],
                                mesh_directory: str,
                                output_directory: str) -> Dict[str, bool]:
        """
        Create CSV files for landmarks based on settings configuration.
        
        Args:
            all_landmarks: Dictionary of all computed landmark coordinates
            mesh_directory: Directory containing mesh files
            output_directory: Output directory for CSV files
            
        Returns:
            Dictionary mapping file names to success status
        """
        # Create output directory
        os.makedirs(output_directory, exist_ok=True)
        
        results = {}
        
        # Step 1: Create mesh CSV files
        if self.verbose:
            logger.info("\n=== CREATING MESH CSV FILES ===")
        
        mesh_files = self.settings['csv_file_names'][0]['mesh']
        mesh_order = self.settings['csv_mesh_files_order']
        
        for mesh_name in mesh_files:
            csv_filename = f"{mesh_name}_mesh_target_landmarks.csv"
            csv_path = os.path.join(output_directory, csv_filename)
            mesh_path = find_mesh_file(mesh_directory, mesh_name)
            
            # Check if file already exists
            if os.path.exists(csv_path):
                if self.verbose:
                    logger.info(f"\n⚠️ SKIPPING mesh CSV (already exists): {csv_filename}")
                results[csv_filename] = True  # Mark as successful (file exists)
                continue
            
            if self.verbose:
                logger.info(f"\n✅ Creating mesh CSV: {csv_filename}")
            
            try:
                with open(csv_path, 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    
                    # Write header
                    csv_writer.writerow(['name', 'x', 'y', 'z'])
                    
                    # Write landmarks in specified order
                    landmark_count = 0
                    if mesh_name in mesh_order:
                        for landmark_name in mesh_order[mesh_name]:
                            if landmark_name in all_landmarks:
                                coords = all_landmarks[landmark_name]
                                csv_writer.writerow([
                                    landmark_name,
                                    f'{coords[0]:.6f}',
                                    f'{coords[1]:.6f}',
                                    f'{coords[2]:.6f}'
                                ])
                                landmark_count += 1
                    
                    # Write all mesh vertices
                    if mesh_path is not None and os.path.exists(mesh_path):
                        vertices, _ = self.load_mesh_with_pymeshlab(mesh_path)
                        vertex_count = 0
                        
                        for i, vertex in enumerate(vertices):
                            csv_writer.writerow([
                                f'landmark_{i}',
                                f'{vertex[0]:.6f}',
                                f'{vertex[1]:.6f}',
                                f'{vertex[2]:.6f}'
                            ])
                            vertex_count += 1
                        
                        if self.verbose:
                            logger.info(f"   ✅ Written {landmark_count} landmarks and {vertex_count} vertices")
                    else:
                        if self.verbose:
                            logger.info(f"   ✅ Written {landmark_count} landmarks (mesh file not found for vertices)")
                
                results[csv_filename] = True
                
            except Exception as e:
                logger.error(f"   ❌ Failed to create {csv_filename}: {e}")
                results[csv_filename] = False
        
        # Step 2: Create stations CSV files
        if self.verbose:
            logger.info("\n=== CREATING STATIONS CSV FILES ===")
        
        stations_files = self.settings['csv_file_names'][1]['stations']
        stations_order = self.settings['csv_stations_files_order']
        
        for stations_name in stations_files:
            csv_filename = f"{stations_name}_stations_target_landmarks.csv"
            csv_path = os.path.join(output_directory, csv_filename)
            
            # Check if file already exists
            if os.path.exists(csv_path):
                if self.verbose:
                    logger.info(f"\n⚠️ SKIPPING stations CSV (already exists): {csv_filename}")
                results[csv_filename] = True  # Mark as successful (file exists)
                continue
            
            if self.verbose:
                logger.info(f"\n✅ Creating stations CSV: {csv_filename}")
            
            try:
                with open(csv_path, 'w', newline='') as csvfile:
                    csv_writer = csv.writer(csvfile)
                    
                    # Write header
                    csv_writer.writerow(['name', 'x', 'y', 'z'])
                    
                    # Write landmarks in specified order (no vertices for stations files)
                    landmark_count = 0
                    if stations_name in stations_order:
                        for landmark_name in stations_order[stations_name]:
                            if landmark_name in all_landmarks:
                                coords = all_landmarks[landmark_name]
                                csv_writer.writerow([
                                    landmark_name,
                                    f'{coords[0]:.6f}',
                                    f'{coords[1]:.6f}',
                                    f'{coords[2]:.6f}'
                                ])
                                landmark_count += 1
                            else:
                                logger.warning(f"   ⚠️ Landmark not found: {landmark_name}")
                    
                    if self.verbose:
                        logger.info(f"   ✅ Written {landmark_count} landmarks")
                
                results[csv_filename] = True
                
            except Exception as e:
                logger.error(f"   ❌ Failed to create {csv_filename}: {e}")
                results[csv_filename] = False
        
        return results

def run_landmark_processing(settings_file_path: str,
                           template_geom_dir: str,
                           mesh_directory: str,
                           output_directory: str,
                           verbose: bool = True) -> Tuple[Dict[str, np.ndarray], Dict[str, bool]]:
    """
    Run complete landmark processing workflow using settings file.
    
    Args:
        settings_file_path: Path to YAML settings file
        template_geom_dir: Directory containing template geometry files
        mesh_directory: Directory containing target mesh files (STL format)
        output_directory: Output directory for CSV files
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (processed_landmarks, csv_creation_results)
    """
    processor = LandmarkProcessor(
        settings_file_path=settings_file_path,
        template_geom_dir=template_geom_dir,
        verbose=verbose
    )
    
    # Process all landmarks
    all_landmarks = processor.process_landmarks_from_settings(mesh_directory)
    
    # Create CSV files
    csv_results = processor.create_landmark_csv_files(
        all_landmarks, mesh_directory, output_directory
    )
    
    return all_landmarks, csv_results

def verify_landmark_csv_files(output_directory: str, 
                             settings_file_path: str,
                             verbose: bool = True) -> Tuple[List[str], List[str]]:
    """
    Verify that landmark CSV files were created successfully.
    
    Args:
        output_directory: Directory to check for output files
        settings_file_path: Path to YAML settings file
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (found_files, missing_files)
    """
    output_path = Path(output_directory)
    
    if not output_path.exists():
        if verbose:
            logger.warning(f"⚠️ Output directory does not exist: {output_directory}")
        return [], []
    
    # Load settings to get expected files
    with open(settings_file_path, 'r') as f:
        settings = yaml.safe_load(f)
    
    expected_files = []
    
    # Expected mesh files
    for mesh_name in settings['csv_file_names'][0]['mesh']:
        expected_files.append(f"{mesh_name}_mesh_target_landmarks.csv")
    
    # Expected stations files
    for stations_name in settings['csv_file_names'][1]['stations']:
        expected_files.append(f"{stations_name}_stations_target_landmarks.csv")
    
    # Find existing files
    found_files = []
    missing_files = []
    
    for expected_file in expected_files:
        file_path = output_path / expected_file
        if file_path.exists():
            found_files.append(expected_file)
            if verbose:
                file_size = file_path.stat().st_size
                # Count lines
                with open(file_path, 'r') as f:
                    line_count = sum(1 for _ in f)
                logger.info(f"✅ {expected_file} ({line_count-1} entries, {file_size:,} bytes)")
        else:
            missing_files.append(expected_file)
            if verbose:
                logger.warning(f"❌ {expected_file} not found")
    
    if verbose:
        logger.info(f"\n=== CSV FILE VERIFICATION ===")
        logger.info(f"Found: {len(found_files)}/{len(expected_files)} files")
        
        if missing_files:
            logger.warning(f"Missing files: {missing_files}")
    
    return found_files, missing_files

def create_template_landmark_csv_files(settings_file_path: str,
                                      template_geom_dir: str,
                                      output_directory: str,
                                      verbose: bool = True) -> Dict[str, bool]:
    """
    Create template CSV files with landmark definitions and mesh vertices.
    These files contain both template landmarks AND all mesh vertices (for mesh files).
    
    Args:
        settings_file_path: Path to YAML settings file
        template_geom_dir: Directory containing template geometry STL files
        output_directory: Output directory for template CSV files
        verbose: Enable verbose logging
        
    Returns:
        Dictionary mapping file names to success status
    """
    if verbose:
        logger.info("=== CREATING TEMPLATE LANDMARK CSV FILES ===")
        logger.info(f"Settings file: {settings_file_path}")
        logger.info(f"Template geometry directory: {template_geom_dir}")
        logger.info(f"Output directory: {output_directory}")
    
    # Load settings
    try:
        with open(settings_file_path, 'r') as f:
            settings = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"❌ Failed to load settings file: {e}")
        return {}
    
    # Check if template CSV files already exist
    if os.path.exists(output_directory):
        mesh_files = settings['csv_file_names'][0]['mesh']
        stations_files = settings['csv_file_names'][1]['stations']
        
        # Check if all expected files exist
        all_exist = True
        existing_files = []
        
        for mesh_name in mesh_files:
            csv_file = os.path.join(output_directory, f"{mesh_name}_mesh_template_landmarks.csv")
            if os.path.exists(csv_file):
                existing_files.append(f"{mesh_name}_mesh_template_landmarks.csv")
            else:
                all_exist = False
                break
        
        if all_exist:
            for stations_name in stations_files:
                csv_file = os.path.join(output_directory, f"{stations_name}_stations_template_landmarks.csv")
                if os.path.exists(csv_file):
                    existing_files.append(f"{stations_name}_stations_template_landmarks.csv")
                else:
                    all_exist = False
                    break
        
        if all_exist:
            if verbose:
                logger.info(f"✅ Template CSV files already exist in: {output_directory}")
                logger.info(f"   Found {len(existing_files)} template CSV file(s)")
                logger.info("   Skipping template CSV creation (files already exist)")
                for filename in existing_files:
                    logger.info(f"   ✅ {filename}")
            
            # Return success status for all files
            results = {filename: True for filename in existing_files}
            return results
    
    # If files don't exist, proceed with creation
    if verbose:
        logger.info("No existing template CSV files found. Creating new files...")
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    results = {}
    
    # Compute ALL landmarks (measured + computed) for template
    all_template_landmarks = {}
    
    # Step 1: Get measured landmarks from settings
    for landmark_def in settings['landmarks']:
        if landmark_def.get('measured', False):
            landmark_name = landmark_def['name']
            template_location = np.array(landmark_def['measurement']['location'])
            all_template_landmarks[landmark_name] = template_location
    
    if verbose:
        logger.info(f"✅ Found {len(all_template_landmarks)} measured template landmarks")
    
    # Step 2: Compute derived landmarks for template
    if verbose:
        logger.info("\n=== Computing Derived Template Landmarks ===")
    
    for landmark_def in settings['landmarks']:
        if not landmark_def.get('measured', False):
            landmark_name = landmark_def['name']
            computation = landmark_def.get('computation', {})
            method = computation.get('method')
            
            if verbose:
                logger.info(f"✅ Computing template landmark: {landmark_name} (method: {method})")
            
            try:
                if method == 'midpoint':
                    landmark_names = computation['landmarks_midpoint']
                    L1 = all_template_landmarks[landmark_names[0]]
                    L2 = all_template_landmarks[landmark_names[1]]
                    position = compute_midpoint(L1, L2)
                    all_template_landmarks[landmark_name] = position
                
                elif method == 'sphere':
                    # For template, we need to fit sphere to template mesh
                    mesh_name = computation['mesh']
                    vertex_indices = computation['vertices_list']
                    
                    template_mesh_path = find_mesh_file(template_geom_dir, mesh_name)
                    if template_mesh_path is None:
                        logger.warning(f"⚠️ Template mesh not found: {mesh_name}")
                        continue
                    if os.path.exists(template_mesh_path):
                        if PYMESHLAB_AVAILABLE:
                            ms = ml.MeshSet()
                            ms.load_new_mesh(str(template_mesh_path))
                            mesh = ms.current_mesh()
                            vertices = mesh.vertex_matrix()
                            
                            position = fit_sphere_to_vertices(vertices, vertex_indices)
                            all_template_landmarks[landmark_name] = position
                        else:
                            logger.warning(f"⚠️ PyMeshLab not available, skipping sphere fit for {landmark_name}")
                    else:
                        logger.warning(f"⚠️ Template mesh not found: {template_mesh_path}")
                
                elif method == 'point_in_plane':
                    landmark_names = computation['landmarks_plane']
                    L1 = all_template_landmarks[landmark_names[0]]
                    L2 = all_template_landmarks[landmark_names[1]]
                    L3 = all_template_landmarks[landmark_names[2]]
                    position = compute_point_in_plane(L1, L2, L3)
                    all_template_landmarks[landmark_name] = position
                
                elif method == 'copy':
                    copied_landmark = computation['copied_landmark']
                    position = all_template_landmarks[copied_landmark].copy()
                    all_template_landmarks[landmark_name] = position
                
            except KeyError as e:
                logger.error(f"❌ Missing required landmark: {e}")
            except Exception as e:
                logger.error(f"❌ Error computing template landmark: {e}")
    
    if verbose:
        logger.info(f"\n✅ Total template landmarks computed: {len(all_template_landmarks)}")
    
    # Step 3: Create mesh template CSV files (with vertices)
    if verbose:
        logger.info("\n=== CREATING TEMPLATE MESH CSV FILES ===")
    
    mesh_files = settings['csv_file_names'][0]['mesh']
    mesh_order = settings['csv_mesh_files_order']
    
    for mesh_name in mesh_files:
        csv_filename = f"{mesh_name}_mesh_template_landmarks.csv"
        csv_path = os.path.join(output_directory, csv_filename)
        mesh_path = find_mesh_file(template_geom_dir, mesh_name)
        
        if verbose:
            logger.info(f"\n✅ Creating template mesh CSV: {csv_filename}")
        
        try:
            with open(csv_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                
                # Write header
                csv_writer.writerow(['name', 'x', 'y', 'z'])
                
                # Write landmarks in specified order
                landmark_count = 0
                if mesh_name in mesh_order:
                    for landmark_name in mesh_order[mesh_name]:
                        if landmark_name in all_template_landmarks:
                            coords = all_template_landmarks[landmark_name]
                            csv_writer.writerow([
                                landmark_name,
                                f'{coords[0]:.6f}',
                                f'{coords[1]:.6f}',
                                f'{coords[2]:.6f}'
                            ])
                            landmark_count += 1
                
                # Write all mesh vertices
                if mesh_path is not None:
                    if PYMESHLAB_AVAILABLE:
                        ms = ml.MeshSet()
                        ms.load_new_mesh(str(mesh_path))
                        mesh = ms.current_mesh()
                        vertices = mesh.vertex_matrix()
                        vertex_count = 0
                        
                        for i, vertex in enumerate(vertices):
                            csv_writer.writerow([
                                f'landmark_{i}',
                                f'{vertex[0]:.6f}',
                                f'{vertex[1]:.6f}',
                                f'{vertex[2]:.6f}'
                            ])
                            vertex_count += 1
                        
                        if verbose:
                            logger.info(f"   ✅ Written {landmark_count} landmarks and {vertex_count} vertices")
                    else:
                        logger.warning(f"   ⚠️ PyMeshLab not available, only landmarks written")
                        if verbose:
                            logger.info(f"   ✅ Written {landmark_count} landmarks (no vertices)")
                else:
                    if verbose:
                        logger.info(f"   ✅ Written {landmark_count} landmarks (mesh file not found)")
            
            results[csv_filename] = True
            
        except Exception as e:
            logger.error(f"   ❌ Failed to create {csv_filename}: {e}")
            results[csv_filename] = False
    
    # Step 4: Create stations template CSV files (all landmarks, no vertices)
    if verbose:
        logger.info("\n=== CREATING TEMPLATE STATIONS CSV FILES ===")
    
    stations_files = settings['csv_file_names'][1]['stations']
    stations_order = settings['csv_stations_files_order']
    
    for stations_name in stations_files:
        csv_filename = f"{stations_name}_stations_template_landmarks.csv"
        csv_path = os.path.join(output_directory, csv_filename)
        
        if verbose:
            logger.info(f"\n✅ Creating template stations CSV: {csv_filename}")
        
        try:
            with open(csv_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                
                # Write header
                csv_writer.writerow(['name', 'x', 'y', 'z'])
                
                # Write ALL landmarks in specified order (both measured and computed)
                landmark_count = 0
                if stations_name in stations_order:
                    for landmark_name in stations_order[stations_name]:
                        if landmark_name in all_template_landmarks:
                            coords = all_template_landmarks[landmark_name]
                            csv_writer.writerow([
                                landmark_name,
                                f'{coords[0]:.6f}',
                                f'{coords[1]:.6f}',
                                f'{coords[2]:.6f}'
                            ])
                            landmark_count += 1
                        else:
                            logger.warning(f"   ⚠️ Template landmark not found: {landmark_name}")
                
                if verbose:
                    logger.info(f"   ✅ Written {landmark_count} landmarks")
            
            results[csv_filename] = True
            
        except Exception as e:
            logger.error(f"   ❌ Failed to create {csv_filename}: {e}")
            results[csv_filename] = False
    
    if verbose:
        successful = sum(results.values())
        total = len(results)
        logger.info(f"\n=== TEMPLATE CSV CREATION SUMMARY ===")
        logger.info(f"Successfully created: {successful}/{total} template CSV files")
    
    return results

if __name__ == "__main__":
    """Command line interface for landmark processing."""
    
    import sys
    
    if len(sys.argv) < 5:
        print("Usage: python landmark_processing_utils.py <settings_file> <template_geom_dir> <mesh_directory> <output_directory>")
        print("\nExample:")
        print("  python landmark_processing_utils.py ./MSK-Morph_settings.yaml ./template_geometry ./temp/postprocessing_output ../participant_data/participant_id/warping_files")
        sys.exit(1)
    
    settings_file = sys.argv[1]
    template_geom_dir = sys.argv[2]
    mesh_dir = sys.argv[3]
    output_dir = sys.argv[4]
    
    # Run landmark processing
    processed_landmarks, csv_results = run_landmark_processing(
        settings_file_path=settings_file,
        template_geom_dir=template_geom_dir,
        mesh_directory=mesh_dir,
        output_directory=output_dir,
        verbose=True
    )
    
    # Verify outputs
    found_files, missing_files = verify_landmark_csv_files(
        output_directory=output_dir,
        settings_file_path=settings_file,
        verbose=True
    )
    
    # Summary
    successful = sum(csv_results.values())
    total = len(csv_results)
    
    if successful == total and not missing_files:
        print(f"\n✅ Successfully processed all landmarks and created all {total} CSV files!")
    else:
        print(f"\n⚠️ Processed landmarks with some issues.")
        print(f"   Created: {successful}/{total} files")
        if missing_files:
            print(f"   Missing: {missing_files}")
        sys.exit(1)