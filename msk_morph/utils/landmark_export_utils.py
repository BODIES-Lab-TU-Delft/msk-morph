#!/usr/bin/env python3
"""
Landmark Export Utilities Module
Functions for exporting landmarks to OpenSim XML and TRC formats.
"""

import os
import csv
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from xml.etree.ElementTree import Element, SubElement, ElementTree
import xml.etree.ElementTree as ET

# Import auxiliary landmark computation functions
from utils.aux_landmark_utils import compute_midpoint

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LandmarkExporter:
    """
    Export landmarks to OpenSim XML and TRC formats based on settings file.
    """
    
    def __init__(self, settings_file_path: str, verbose: bool = True):
        """
        Initialize the landmark exporter.
        
        Args:
            settings_file_path: Path to YAML settings file
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        if not verbose:
            logger.setLevel(logging.WARNING)
        
        # Load settings from YAML file
        self.settings = self._load_settings(settings_file_path)
        
        # Build mappings from settings
        self._build_mappings_from_settings()
    
    def _load_settings(self, settings_file_path: str) -> Dict:
        """Load settings from YAML file."""
        try:
            with open(settings_file_path, 'r') as f:
                settings = yaml.safe_load(f)
            
            if self.verbose:
                logger.info(f"✅ Loaded settings from: {settings_file_path}")
            
            return settings
            
        except Exception as e:
            logger.error(f"❌ Failed to load settings file: {e}")
            raise
    
    def _build_mappings_from_settings(self):
        """Build landmark mappings from settings file."""
        # Extract landmarks that should be in marker files
        self.landmarks_for_markers = []
        self.landmark_name_mapping = {}
        self.body_frame_mapping = {}
        
        for landmark_def in self.settings['landmarks']:
            if landmark_def.get('in_marker_file', False):
                landmark_name = landmark_def['name']
                marker_details = landmark_def.get('details_marker', {})
                
                marker_name = marker_details.get('marker_name')
                body_frame = marker_details.get('osim_body_frame')
                
                if marker_name and body_frame:
                    self.landmarks_for_markers.append(landmark_def)
                    self.landmark_name_mapping[landmark_name] = marker_name
                    self.body_frame_mapping[marker_name] = body_frame
        
        # Get marker order from settings
        self.landmark_order = self.settings.get('marker_order', [])
        
        if self.verbose:
            logger.info(f"✅ Found {len(self.landmarks_for_markers)} landmarks for marker export")
            logger.info(f"✅ Marker order: {len(self.landmark_order)} markers")
    
    def load_landmarks_from_csv_files(self, csv_directory: str) -> Dict[str, np.ndarray]:
        """
        Load landmarks from CSV files created by landmark processing pipeline.
        
        Args:
            csv_directory: Directory containing CSV files with landmarks
            
        Returns:
            Dictionary mapping landmark names to coordinates
        """
        landmarks = {}
        
        # Get list of stations files from settings
        stations_files = self.settings['csv_file_names'][1]['stations']
        
        if self.verbose:
            logger.info(f"Loading landmarks from CSV directory: {csv_directory}")
        
        for stations_name in stations_files:
            csv_file = f"{stations_name}_stations_target_landmarks.csv"
            csv_path = os.path.join(csv_directory, csv_file)
            
            if not os.path.exists(csv_path):
                if self.verbose:
                    logger.warning(f"⚠️ CSV file not found: {csv_path}")
                continue
            
            if self.verbose:
                logger.info(f"✅ Loading landmarks from: {csv_file}")
            
            try:
                with open(csv_path, 'r') as f:
                    csv_reader = csv.DictReader(f)
                    for row in csv_reader:
                        landmark_name = row['name']
                        x = float(row['x'])
                        y = float(row['y'])
                        z = float(row['z'])
                        
                        landmarks[landmark_name] = np.array([x, y, z])
                        
                        if self.verbose:
                            logger.debug(f"  ✅ {landmark_name}: [{x:.6f}, {y:.6f}, {z:.6f}]")
                            
            except Exception as e:
                logger.error(f"❌ Failed to load {csv_file}: {e}")
        
        if self.verbose:
            logger.info(f"✅ Total landmarks loaded from CSV: {len(landmarks)}")
        
        return landmarks
    
    def calculate_marker_only_landmarks(self, landmarks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calculate landmarks that only appear in marker files (not in CSV files).
        These are landmarks with in_csv_files: false and in_marker_file: true.
        
        Args:
            landmarks: Dictionary of already loaded landmarks
            
        Returns:
            Updated dictionary with marker-only landmarks added
        """
        updated_landmarks = landmarks.copy()
        
        if self.verbose:
            logger.info("=== Calculating Marker-Only Landmarks ===")
        
        for landmark_def in self.settings['landmarks']:
            # Check if this is a marker-only landmark
            if (landmark_def.get('in_marker_file', False) and 
                not landmark_def.get('in_csv_files', True)):
                
                landmark_name = landmark_def['name']
                
                if self.verbose:
                    logger.info(f"\n✅ Processing marker-only landmark: {landmark_name}")
                
                # Check if already exists (shouldn't, but just in case)
                if landmark_name in updated_landmarks:
                    if self.verbose:
                        logger.info(f"   ⚠️ Already exists, skipping")
                    continue
                
                # Get computation details
                computation = landmark_def.get('computation', {})
                method = computation.get('method')
                
                if self.verbose:
                    logger.info(f"   Method: {method}")
                
                try:
                    if method == 'midpoint':
                        # Calculate midpoint
                        landmark_names = computation['landmarks_midpoint']
                        L1 = updated_landmarks[landmark_names[0]]
                        L2 = updated_landmarks[landmark_names[1]]
                        
                        position = compute_midpoint(L1, L2)
                        updated_landmarks[landmark_name] = position
                        
                        if self.verbose:
                            logger.info(f"   Input landmarks: {landmark_names}")
                            logger.info(f"   ✅ Computed midpoint: {position}")
                    
                    elif method == 'copy':
                        # Copy from another landmark
                        copied_landmark = computation['copied_landmark']
                        position = updated_landmarks[copied_landmark].copy()
                        updated_landmarks[landmark_name] = position
                        
                        if self.verbose:
                            logger.info(f"   Copied from: {copied_landmark}")
                            logger.info(f"   ✅ Copied position: {position}")
                    
                    else:
                        logger.warning(f"   ⚠️ Unknown or unsupported method for marker-only landmark: {method}")
                
                except KeyError as e:
                    logger.error(f"   ❌ Missing required landmark: {e}")
                except Exception as e:
                    logger.error(f"   ❌ Error computing landmark: {e}")
        
        return updated_landmarks
    
    def convert_landmark_names(self, landmarks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Convert landmark names from internal format to marker format using settings.
        
        Args:
            landmarks: Dictionary with internal landmark names
            
        Returns:
            Dictionary with converted marker names
        """
        converted = {}
        
        for internal_name, coords in landmarks.items():
            if internal_name in self.landmark_name_mapping:
                marker_name = self.landmark_name_mapping[internal_name]
                converted[marker_name] = coords
                
                if self.verbose:
                    logger.debug(f"✅ Converted {internal_name} -> {marker_name}")
            else:
                if self.verbose:
                    logger.debug(f"⚠️ No marker mapping found for: {internal_name}")
        
        return converted
    
    def export_to_xml(self, landmarks: Dict[str, np.ndarray], 
                    output_path: str,
                    markerset_name: str = "markerset") -> bool:
        """
        Export landmarks to OpenSim XML format.
        
        Args:
            landmarks: Dictionary of landmark coordinates (with marker names as keys)
            output_path: Output XML file path
            markerset_name: Name for the marker set
            
        Returns:
            True if successful
        """
        try:
            # Check if file already exists
            if os.path.exists(output_path):
                if self.verbose:
                    logger.info(f"⚠️ SKIPPING XML export (file already exists): {output_path}")
                return True  # Return success since file exists
            
            if self.verbose:
                logger.info(f"Exporting landmarks to XML: {output_path}")
            
            # Create root element
            root = Element("OpenSimDocument", Version="40000")
            
            # Create MarkerSet
            markerset = SubElement(root, "MarkerSet", name=markerset_name)
            objects = SubElement(markerset, "objects")
            
            # Add markers in defined order
            marker_count = 0
            for marker_name in self.landmark_order:
                if marker_name in landmarks:
                    coords = landmarks[marker_name]
                    
                    # Create marker element
                    marker = SubElement(objects, "Marker", name=marker_name)
                    
                    # Add socket_parent_frame
                    if marker_name in self.body_frame_mapping:
                        socket_frame = SubElement(marker, "socket_parent_frame")
                        socket_frame.text = self.body_frame_mapping[marker_name]
                    else:
                        logger.warning(f"⚠️ No body frame mapping for {marker_name}")
                        socket_frame = SubElement(marker, "socket_parent_frame")
                        socket_frame.text = "/bodyset/pelvis/pelvis_physicalbodyoffset"
                    
                    # Add location (coordinates already in meters)
                    location = SubElement(marker, "location")
                    location.text = f"{coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f}"
                    
                    # Add fixed property
                    fixed = SubElement(marker, "fixed")
                    fixed.text = "true"
                    
                    marker_count += 1
                    
                    if self.verbose:
                        logger.debug(f"  ✅ Added marker {marker_name}: {coords}")
                else:
                    if self.verbose:
                        logger.warning(f"  ⚠️ Missing landmark: {marker_name}")
            
            # Create output directory if needed
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Write XML file with proper formatting
            self._format_and_write_xml(root, output_path)
            
            if self.verbose:
                logger.info(f"✅ Successfully exported {marker_count} markers to XML")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to export XML: {e}")
            return False
    
    def _format_and_write_xml(self, root: Element, output_path: str):
        """
        Format and write XML with proper indentation.
        """
        # Add XML declaration and formatting
        self._indent_xml(root)
        
        # Create tree and write
        tree = ElementTree(root)
        
        # Write with XML declaration
        with open(output_path, 'wb') as f:
            f.write(b'<?xml version="1.0" encoding="utf-8"?>\n')
            tree.write(f, encoding='utf-8')
    
    def _indent_xml(self, elem: Element, level: int = 0):
        """
        Add indentation to XML elements for readable formatting.
        """
        i = "\n" + level * "   "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "   "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for child in elem:
                self._indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
    
    def export_to_trc(self, landmarks: Dict[str, np.ndarray], 
                    output_path: str,
                    participant_id: str = "participantID",
                    num_frames: int = 3,
                    data_rate: float = 100.0) -> bool:
        """
        Export landmarks to TRC (Track Row Column) format.
        
        Args:
            landmarks: Dictionary of landmark coordinates (with marker names as keys)
            output_path: Output TRC file path
            participant_id: Participant ID for the file path reference
            num_frames: Number of frames to write (default: 3)
            data_rate: Data acquisition rate in Hz (default: 100.0)
            
        Returns:
            True if successful
        """
        try:
            # Check if file already exists
            if os.path.exists(output_path):
                if self.verbose:
                    logger.info(f"⚠️ SKIPPING TRC export (file already exists): {output_path}")
                return True  # Return success since file exists
            
            if self.verbose:
                logger.info(f"Exporting landmarks to TRC: {output_path}")
            
            # Get ordered landmarks
            ordered_landmarks = []
            for marker_name in self.landmark_order:
                if marker_name in landmarks:
                    ordered_landmarks.append((marker_name, landmarks[marker_name]))
                else:
                    if self.verbose:
                        logger.warning(f"⚠️ Missing landmark for TRC: {marker_name}")
            
            num_markers = len(ordered_landmarks)
            
            if num_markers == 0:
                logger.error("❌ No landmarks to export to TRC")
                return False
            
            # Create output directory if needed
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', newline='') as f:
                # Write header line 1
                f.write(f"PathFileType\t4\t(X/Y/Z)\t../participant_data/{participant_id}/markers.trc\n")
                
                # Write header line 2
                f.write(f"DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n")
                
                # Write header line 3
                f.write(f"{data_rate:.6f}\t{data_rate:.6f}\t{num_frames}\t{num_markers}\tm\t{data_rate:.6f}\t0\t{num_frames}\n")
                
                # Write column headers line 1
                header1 = "Frame#\tTime"
                for marker_name, _ in ordered_landmarks:
                    header1 += f"\t{marker_name}\t\t\t"
                header1 += "\t\n"
                f.write(header1)
                
                # Write column headers line 2
                header2 = "\t"
                for i, (marker_name, _) in enumerate(ordered_landmarks):
                    marker_num = i + 1  # Marker number (1-indexed)
                    header2 += f"\tX{marker_num}\tY{marker_num}\tZ{marker_num}"
                header2 += "\t\n\n"
                f.write(header2)
                
                # Write data frames (repeat same coordinates for all frames)
                for frame in range(1, num_frames + 1):
                    time = (frame - 1) * (1.0 / data_rate)
                    line = f"{frame}\t{time:.6f}"
                    
                    for marker_name, coords in ordered_landmarks:
                        # Coordinates already in meters
                        x = coords[0]
                        y = coords[1]
                        z = coords[2]
                        line += f"\t{x:.6f}\t{y:.6f}\t{z:.6f}"
                    
                    line += "\t\n"
                    f.write(line)
            
            if self.verbose:
                logger.info(f"✅ Successfully exported {num_markers} markers to TRC with {num_frames} frames")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to export TRC: {e}")
            return False
    
    def export_landmarks(self, csv_directory: str,
                        output_directory: str,
                        participant_id: str = "participantID",
                        export_xml: bool = True,
                        export_trc: bool = True) -> Dict[str, bool]:
        """
        Complete workflow to export landmarks to XML and TRC formats.
        
        Args:
            csv_directory: Directory containing CSV files with landmarks
            output_directory: Output directory for XML and TRC files
            participant_id: Participant ID for file references
            export_xml: Whether to export XML format
            export_trc: Whether to export TRC format
            
        Returns:
            Dictionary with export results
        """
        results = {}
        
        if self.verbose:
            logger.info("=== LANDMARK EXPORT WORKFLOW ===")
            logger.info(f"CSV directory: {csv_directory}")
            logger.info(f"Output directory: {output_directory}")
            logger.info(f"Participant ID: {participant_id}")
        
        # Step 1: Load landmarks from CSV files
        landmarks = self.load_landmarks_from_csv_files(csv_directory)
        
        if not landmarks:
            logger.error("❌ No landmarks loaded from CSV files")
            return {'xml': False, 'trc': False}
        
        # Step 2: Calculate marker-only landmarks (those not in CSV files)
        landmarks_with_marker_only = self.calculate_marker_only_landmarks(landmarks)
        
        # Step 3: Convert landmark names to marker names
        converted_landmarks = self.convert_landmark_names(landmarks_with_marker_only)
        
        if self.verbose:
            logger.info(f"✅ Final landmarks for export: {list(converted_landmarks.keys())}")
        
        # Step 4: Export to XML
        if export_xml:
            xml_path = os.path.join(output_directory, "markers.xml")
            results['xml'] = self.export_to_xml(converted_landmarks, xml_path)
        else:
            results['xml'] = True
        
        # Step 5: Export to TRC
        if export_trc:
            trc_path = os.path.join(output_directory, "markers.trc")
            results['trc'] = self.export_to_trc(converted_landmarks, trc_path, participant_id)
        else:
            results['trc'] = True
        
        return results


def export_landmarks_to_opensim_formats(settings_file_path: str,
                                       csv_directory: str,
                                       output_directory: str,
                                       participant_id: str = "participant_id",
                                       verbose: bool = True) -> Dict[str, bool]:
    """
    Convenience function to export landmarks to OpenSim XML and TRC formats.
    
    Args:
        settings_file_path: Path to YAML settings file
        csv_directory: Directory containing CSV files with landmarks
        output_directory: Output directory for XML and TRC files
        participant_id: Participant ID for file references
        verbose: Enable verbose logging
        
    Returns:
        Dictionary with export results
    """
    exporter = LandmarkExporter(settings_file_path=settings_file_path, verbose=verbose)
    return exporter.export_landmarks(
        csv_directory=csv_directory,
        output_directory=output_directory,
        participant_id=participant_id,
        export_xml=True,
        export_trc=True
    )


def verify_opensim_exports(output_directory: str, verbose: bool = True) -> Tuple[bool, bool]:
    """
    Verify that OpenSim export files were created successfully.
    
    Args:
        output_directory: Directory to check for output files
        verbose: Enable verbose logging
        
    Returns:
        Tuple of (xml_exists, trc_exists)
    """
    xml_path = os.path.join(output_directory, "markers.xml")
    trc_path = os.path.join(output_directory, "markers.trc")
    
    xml_exists = os.path.exists(xml_path)
    trc_exists = os.path.exists(trc_path)
    
    if verbose:
        logger.info("=== OPENSIM EXPORT VERIFICATION ===")
        
        if xml_exists:
            file_size = os.path.getsize(xml_path)
            logger.info(f"✅ markers.xml found ({file_size:,} bytes)")
            
            # Count markers in XML
            try:
                tree = ET.parse(xml_path)
                markers = tree.findall(".//Marker")
                logger.info(f"   ✅ Contains {len(markers)} markers")
            except Exception as e:
                logger.warning(f"   ⚠️ Could not parse XML: {e}")
        else:
            logger.error("❌ markers.xml not found")
        
        if trc_exists:
            file_size = os.path.getsize(trc_path)
            logger.info(f"✅ markers.trc found ({file_size:,} bytes)")
            
            # Read TRC header info
            try:
                with open(trc_path, 'r') as f:
                    lines = f.readlines()
                    if len(lines) >= 3:
                        # Parse header line 3 (data info)
                        header_parts = lines[2].strip().split('\t')
                        if len(header_parts) >= 4:
                            data_rate = header_parts[0]
                            num_frames = header_parts[2]
                            num_markers = header_parts[3]
                            logger.info(f"   ✅ {num_markers} markers, {num_frames} frames, {data_rate}Hz")
            except Exception as e:
                logger.warning(f"   ⚠️ Could not parse TRC: {e}")
        else:
            logger.error("❌ markers.trc not found")
    
    return xml_exists, trc_exists


if __name__ == "__main__":
    """Command line interface for landmark export."""
    
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python landmark_export_utils.py <settings_file> <csv_directory> <output_directory> [participant_id]")
        print("\nExamples:")
        print("  python landmark_export_utils.py ./MSK-Morph_settings.yaml ../participant_data/participant_id/warping_files ../participant_data/participant_id participant_id")
        sys.exit(1)
    
    settings_file = sys.argv[1]
    csv_dir = sys.argv[2]
    output_dir = sys.argv[3]
    participant = sys.argv[4] if len(sys.argv) > 4 else "participant_id"
    
    # Export landmarks
    results = export_landmarks_to_opensim_formats(
        settings_file_path=settings_file,
        csv_directory=csv_dir,
        output_directory=output_dir,
        participant_id=participant,
        verbose=True
    )
    
    if all(results.values()):
        print(f"✅ Successfully exported landmarks for participant {participant}")
    else:
        print(f"❌ Failed to export landmarks for participant {participant}")
        sys.exit(1)