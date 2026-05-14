#!/usr/bin/env python3
"""
Multi-format Mesh Converter (STL/OBJ <--> VTK)
Converts between STL, OBJ, and VTK formats with configurable scaling.
Uses VTK directly to ensure proper polydata format output.
"""

import os
import sys
from pathlib import Path
import vtk
import numpy as np
from typing import List, Optional, Tuple, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MeshConverter:
    """
    A class for converting between STL, OBJ, and VTK mesh formats with scaling capabilities.
    Uses VTK directly to ensure proper polydata format.
    """
    
    SUPPORTED_FORMATS = {
        'stl': ['.stl', '.STL'],
        'vtk': ['.vtk', '.VTK'],
        'obj': ['.obj', '.OBJ']
    }
    
    def __init__(self, scale_factor: float = 1.0, verbose: bool = True):
        """
        Initialize the mesh converter.
        
        Args:
            scale_factor (float): Factor to scale mesh coordinates (default: 1.0, no scaling)
            verbose (bool): Enable verbose logging
        """
        self.scale_factor = scale_factor
        self.verbose = verbose
        
        if not verbose:
            logger.setLevel(logging.WARNING)
    
    def _scale_polydata(self, polydata, scale_factor):
        """
        Scale the points in a vtkPolyData object.
        
        Args:
            polydata: vtkPolyData object to scale
            scale_factor: Factor to scale coordinates
            
        Returns:
            vtkPolyData: Scaled polydata object
        """
        if scale_factor == 1.0:
            return polydata
            
        # Get points
        points = polydata.GetPoints()
        scaled_points = vtk.vtkPoints()
        scaled_points.SetNumberOfPoints(points.GetNumberOfPoints())
        
        # Scale each point
        for i in range(points.GetNumberOfPoints()):
            point = points.GetPoint(i)
            scaled_point = [coord * scale_factor for coord in point]
            scaled_points.SetPoint(i, scaled_point)
        
        # Create new polydata with scaled points
        scaled_polydata = vtk.vtkPolyData()
        scaled_polydata.DeepCopy(polydata)
        scaled_polydata.SetPoints(scaled_points)
        
        return scaled_polydata
    
    def convert_stl_to_vtk_single(self, input_file: Union[str, Path], 
                                 output_file: Union[str, Path], 
                                 scale_factor: Optional[float] = None) -> bool:
        """
        Convert a single STL file to VTK polydata format.
        
        Args:
            input_file: Path to input STL file
            output_file: Path to output VTK file
            scale_factor: Override the instance scale factor
            
        Returns:
            bool: True if conversion successful, False otherwise
        """
        scale = scale_factor if scale_factor is not None else self.scale_factor
        
        try:
            # Read STL file
            reader = vtk.vtkSTLReader()
            reader.SetFileName(str(input_file))
            reader.Update()
            
            # Get the polydata
            polydata = reader.GetOutput()
            
            # Scale if needed
            if scale != 1.0:
                polydata = self._scale_polydata(polydata, scale)
            
            # Write VTK file
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(str(output_file))
            writer.SetInputData(polydata)
            writer.SetFileTypeToBinary()
            writer.Write()
            
            logger.info(f"✅ Converted: {Path(input_file).name} → {Path(output_file).name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to convert {Path(input_file).name}: {str(e)}")
            return False
    
    def convert_obj_to_vtk_single(self, input_file: Union[str, Path], 
                                  output_file: Union[str, Path], 
                                  scale_factor: Optional[float] = None) -> bool:
        """
        Convert a single OBJ file to VTK polydata format.
        
        Args:
            input_file: Path to input OBJ file
            output_file: Path to output VTK file
            scale_factor: Override the instance scale factor
            
        Returns:
            bool: True if conversion successful, False otherwise
        """
        scale = scale_factor if scale_factor is not None else self.scale_factor
        
        try:
            # Read OBJ file
            reader = vtk.vtkOBJReader()
            reader.SetFileName(str(input_file))
            reader.Update()
            
            # Get the polydata
            polydata = reader.GetOutput()
            
            # Scale if needed
            if scale != 1.0:
                polydata = self._scale_polydata(polydata, scale)
            
            # Write VTK file
            writer = vtk.vtkPolyDataWriter()
            writer.SetFileName(str(output_file))
            writer.SetInputData(polydata)
            writer.SetFileTypeToBinary()
            writer.Write()
            
            logger.info(f"✅ Converted: {Path(input_file).name} → {Path(output_file).name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to convert {Path(input_file).name}: {str(e)}")
            return False
    
    def convert_vtk_to_stl_single(self, input_file: Union[str, Path], 
                                  output_file: Union[str, Path], 
                                  scale_factor: Optional[float] = None) -> bool:
        """
        Convert a single VTK file to STL format.
        
        Args:
            input_file: Path to input VTK file
            output_file: Path to output STL file
            scale_factor: Override the instance scale factor
            
        Returns:
            bool: True if conversion successful, False otherwise
        """
        scale = scale_factor if scale_factor is not None else self.scale_factor
        
        try:
            # Read VTK file
            reader = vtk.vtkPolyDataReader()
            reader.SetFileName(str(input_file))
            reader.Update()
            
            # Get the polydata
            polydata = reader.GetOutput()
            
            # Scale if needed
            if scale != 1.0:
                polydata = self._scale_polydata(polydata, scale)
            
            # Write STL file
            writer = vtk.vtkSTLWriter()
            writer.SetFileName(str(output_file))
            writer.SetInputData(polydata)
            writer.SetFileType(vtk.VTK_BINARY)
            writer.Write()
            
            logger.info(f"✅ Converted: {Path(input_file).name} → {Path(output_file).name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to convert {Path(input_file).name}: {str(e)}")
            return False
    
    def convert_single_file(self, input_file: Union[str, Path], 
                          output_file: Union[str, Path], 
                          scale_factor: Optional[float] = None) -> bool:
        """
        Convert a single file between formats based on extensions.
        
        Args:
            input_file: Path to input file
            output_file: Path to output file
            scale_factor: Override the instance scale factor
            
        Returns:
            bool: True if conversion successful, False otherwise
        """
        input_path = Path(input_file)
        output_path = Path(output_file)
        
        input_ext = input_path.suffix.lower()
        output_ext = output_path.suffix.lower()
        
        # Determine conversion type
        if input_ext in ['.stl'] and output_ext in ['.vtk']:
            return self.convert_stl_to_vtk_single(input_file, output_file, scale_factor)
        elif input_ext in ['.obj'] and output_ext in ['.vtk']:
            return self.convert_obj_to_vtk_single(input_file, output_file, scale_factor)
        elif input_ext in ['.vtk'] and output_ext in ['.stl']:
            return self.convert_vtk_to_stl_single(input_file, output_file, scale_factor)
        else:
            logger.error(f"❌ Unsupported conversion: {input_ext} → {output_ext}")
            return False
    
    def convert_directory(self, input_dir: Union[str, Path], 
                         output_dir: Union[str, Path], 
                         from_format: str, 
                         to_format: str,
                         scale_factor: Optional[float] = None) -> Tuple[int, int]:
        """
        Convert all files in a directory from one format to another.
        
        Args:
            input_dir: Directory containing input files
            output_dir: Directory for output files
            from_format: Input format ('stl', 'obj', or 'vtk')
            to_format: Output format ('stl', 'obj', or 'vtk')
            scale_factor: Override the instance scale factor
            
        Returns:
            Tuple[int, int]: (successful_conversions, total_files)
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        scale = scale_factor if scale_factor is not None else self.scale_factor
        
        if not input_path.exists():
            logger.error(f"❌ Input directory does not exist: {input_path}")
            return 0, 0
        
        # Validate formats
        if from_format not in self.SUPPORTED_FORMATS:
            logger.error(f"❌ Unsupported input format: {from_format}")
            return 0, 0
        
        if to_format not in self.SUPPORTED_FORMATS:
            logger.error(f"❌ Unsupported output format: {to_format}")
            return 0, 0
        
        # Find files with the specified input format
        input_files = []
        for ext in self.SUPPORTED_FORMATS[from_format]:
            input_files.extend(input_path.glob(f"*{ext}"))
        
        if not input_files:
            logger.warning(f"⚠️  No {from_format.upper()} files found in {input_path}")
            return 0, 0
        
        logger.info(f"✅ Found {len(input_files)} {from_format.upper()} file(s) to convert")
        logger.info(f"Converting from {from_format.upper()} to {to_format.upper()}")
        if scale != 1.0:
            logger.info(f"Scale factor: {scale}")
        
        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Convert each file
        successful = 0
        for input_file in input_files:
            output_file = output_path / f"{input_file.stem}.{to_format.lower()}"
            if self.convert_single_file(input_file, output_file, scale):
                successful += 1
        
        if successful == len(input_files):
            logger.info(f"✅ Conversion complete: {successful}/{len(input_files)} files converted successfully")
        elif successful > 0:
            logger.warning(f"⚠️  Partial conversion: {successful}/{len(input_files)} files converted successfully")
        else:
            logger.error(f"❌ Conversion failed: {successful}/{len(input_files)} files converted successfully")
        
        return successful, len(input_files)
    
    def stl_to_vtk(self, input_path: Union[str, Path], 
                   output_path: Union[str, Path], 
                   scale_factor: Optional[float] = None) -> Union[bool, Tuple[int, int]]:
        """
        Convert STL to VTK format.
        
        Args:
            input_path: STL file or directory containing STL files
            output_path: VTK file or output directory
            scale_factor: Scale factor for conversion
            
        Returns:
            bool for single file conversion, (successful, total) tuple for directory
        """
        path = Path(input_path)
        
        if path.is_file():
            return self.convert_stl_to_vtk_single(input_path, output_path, scale_factor)
        elif path.is_dir():
            return self.convert_directory(input_path, output_path, 'stl', 'vtk', scale_factor)
        else:
            logger.error(f"❌ Input path does not exist: {input_path}")
            return False
    
    def obj_to_vtk(self, input_path: Union[str, Path], 
                   output_path: Union[str, Path], 
                   scale_factor: Optional[float] = None) -> Union[bool, Tuple[int, int]]:
        """
        Convert OBJ to VTK format.
        
        Args:
            input_path: OBJ file or directory containing OBJ files
            output_path: VTK file or output directory
            scale_factor: Scale factor for conversion
            
        Returns:
            bool for single file conversion, (successful, total) tuple for directory
        """
        path = Path(input_path)
        
        if path.is_file():
            return self.convert_obj_to_vtk_single(input_path, output_path, scale_factor)
        elif path.is_dir():
            return self.convert_directory(input_path, output_path, 'obj', 'vtk', scale_factor)
        else:
            logger.error(f"❌ Input path does not exist: {input_path}")
            return False
    
    def vtk_to_stl(self, input_path: Union[str, Path], 
                   output_path: Union[str, Path], 
                   scale_factor: Optional[float] = None) -> Union[bool, Tuple[int, int]]:
        """
        Convert VTK to STL format.
        
        Args:
            input_path: VTK file or directory containing VTK files
            output_path: STL file or output directory
            scale_factor: Scale factor for conversion
            
        Returns:
            bool for single file conversion, (successful, total) tuple for directory
        """
        path = Path(input_path)
        
        if path.is_file():
            return self.convert_vtk_to_stl_single(input_path, output_path, scale_factor)
        elif path.is_dir():
            return self.convert_directory(input_path, output_path, 'vtk', 'stl', scale_factor)
        else:
            logger.error(f"❌ Input path does not exist: {input_path}")
            return False

def convert_stl_to_vtk_batch(input_dir: str, output_dir: str, scale_factor: float = 1.0, verbose: bool = False):
    """
    Convenience function for batch STL to VTK conversion.
    
    Args:
        input_dir: Directory containing STL files
        output_dir: Output directory for VTK files
        scale_factor: Scale factor (default: 1.0)
        verbose: Enable verbose logging
        
    Returns:
        Tuple[int, int]: (successful_conversions, total_files)
    """
    converter = MeshConverter(scale_factor=scale_factor, verbose=verbose)
    return converter.stl_to_vtk(input_dir, output_dir)

def convert_obj_to_vtk_batch(input_dir: str, output_dir: str, scale_factor: float = 1.0, verbose: bool = False):
    """
    Convenience function for batch OBJ to VTK conversion.
    
    Args:
        input_dir: Directory containing OBJ files
        output_dir: Output directory for VTK files
        scale_factor: Scale factor (default: 1.0)
        verbose: Enable verbose logging
        
    Returns:
        Tuple[int, int]: (successful_conversions, total_files)
    """
    converter = MeshConverter(scale_factor=scale_factor, verbose=verbose)
    return converter.obj_to_vtk(input_dir, output_dir)

def convert_vtk_to_stl_batch(input_dir: str, output_dir: str, scale_factor: float = 1.0, verbose: bool = False):
    """
    Convenience function for batch VTK to STL conversion.
    
    Args:
        input_dir: Directory containing VTK files
        output_dir: Output directory for STL files
        scale_factor: Scale factor (default: 1.0)
        verbose: Enable verbose logging
        
    Returns:
        Tuple[int, int]: (successful_conversions, total_files)
    """
    converter = MeshConverter(scale_factor=scale_factor, verbose=verbose)
    return converter.vtk_to_stl(input_dir, output_dir)

def convert_single_mesh(input_file: str, output_file: str, scale_factor: float = 1.0, verbose: bool = False):
    """
    Convenience function for single file conversion.
    Automatically detects format based on file extensions.
    
    Args:
        input_file: Path to input file
        output_file: Path to output file
        scale_factor: Scale factor (default: 1.0, no scaling)
        verbose: Enable verbose logging
        
    Returns:
        bool: True if conversion successful, False otherwise
    """
    converter = MeshConverter(scale_factor=scale_factor, verbose=verbose)
    return converter.convert_single_file(input_file, output_file)

def main():
    """Command line interface for the mesh converter."""
    
    if len(sys.argv) < 2:
        print("Usage: python mesh_converter.py <mode> <input_path> <output_path> [scale_factor]")
        print("\nModes:")
        print("  stl2vtk        - Convert STL to VTK (auto-detects file/directory)")
        print("  obj2vtk        - Convert OBJ to VTK (auto-detects file/directory)")
        print("  vtk2stl        - Convert VTK to STL (auto-detects file/directory)")
        print("  stl2vtk-batch  - Batch convert STL directory to VTK")
        print("  obj2vtk-batch  - Batch convert OBJ directory to VTK")
        print("  vtk2stl-batch  - Batch convert VTK directory to STL")
        print("  single         - Convert single file (auto-detect format)")
        print("\nExamples:")
        print("  # Convert single file")
        print("  python mesh_converter.py stl2vtk model.stl model.vtk")
        print("  python mesh_converter.py obj2vtk model.obj model.vtk")
        print("  python mesh_converter.py single model.stl model.vtk 1000.0")
        print("\n  # Convert directory (auto-detect)")
        print("  python mesh_converter.py stl2vtk ./stl_files ./vtk_files")
        print("  python mesh_converter.py obj2vtk ./obj_files ./vtk_files")
        print("  python mesh_converter.py vtk2stl ./vtk_files ./stl_files")
        print("\n  # Batch conversion (explicit)")
        print("  python mesh_converter.py stl2vtk-batch ./stl_files ./vtk_files 1000.0")
        print("  python mesh_converter.py obj2vtk-batch ./obj_files ./vtk_files 1.0")
        print("  python mesh_converter.py vtk2stl-batch ./vtk_files ./stl_files 0.001")
        print("\n  # With scaling")
        print("  python mesh_converter.py stl2vtk ./input ./output 1000.0")
        print("  python mesh_converter.py vtk2stl ./vtk_dir ./stl_dir 0.001")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    
    # Check if enough arguments provided
    if len(sys.argv) < 4:
        print(f"Error: Not enough arguments for mode '{mode}'")
        print(f"Usage: python mesh_converter.py {mode} <input_path> <output_path> [scale_factor]")
        sys.exit(1)
    
    input_path = sys.argv[2]
    output_path = sys.argv[3]
    
    # Parse scale factor
    scale_factor = 1.0
    if len(sys.argv) > 4:
        try:
            scale_factor = float(sys.argv[4])
        except ValueError:
            print("Warning: Invalid scale factor. Using default value of 1.0 (no scaling).")
    
    # Create converter instance
    converter = MeshConverter(scale_factor=scale_factor, verbose=True)
    
    # Perform conversion based on mode
    if mode == 'stl2vtk':
        result = converter.stl_to_vtk(input_path, output_path)
    elif mode == 'obj2vtk':
        result = converter.obj_to_vtk(input_path, output_path)
    elif mode == 'vtk2stl':
        result = converter.vtk_to_stl(input_path, output_path)
    elif mode == 'stl2vtk-batch':
        result = convert_stl_to_vtk_batch(input_path, output_path, scale_factor, verbose=True)
    elif mode == 'obj2vtk-batch':
        result = convert_obj_to_vtk_batch(input_path, output_path, scale_factor, verbose=True)
    elif mode == 'vtk2stl-batch':
        result = convert_vtk_to_stl_batch(input_path, output_path, scale_factor, verbose=True)
    elif mode == 'single':
        result = convert_single_mesh(input_path, output_path, scale_factor, verbose=True)
    else:
        print(f"Unknown mode '{mode}'.")
        print("Valid modes: stl2vtk, obj2vtk, vtk2stl, stl2vtk-batch, obj2vtk-batch, vtk2stl-batch, single")
        sys.exit(1)
    
    # Report results
    if isinstance(result, bool):
        if result:
            print("Conversion completed successfully!")
        else:
            print("Conversion failed!")
            sys.exit(1)
    elif isinstance(result, tuple):
        successful, total = result
        if successful == total:
            print(f"All conversions completed successfully! ({successful}/{total})")
        elif successful > 0:
            print(f"Partial conversion completed ({successful}/{total})")
            sys.exit(1)
        else:
            print(f"All conversions failed! ({successful}/{total})")
            sys.exit(1)

if __name__ == "__main__":
    main()