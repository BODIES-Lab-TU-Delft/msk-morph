#!/usr/bin/env python3
"""
Deformetrica Registration Utilities Module
Functions for running Deformetrica mesh registration using Python API.
"""

import subprocess
import os
import sys
from pathlib import Path
import logging
import tempfile
import time
from typing import Optional, Dict, List, Tuple, Union
import glob


class DeformetricaRegistration:
    """
    A Python pipeline to run Deformetrica using Python API directly without XML files.
    """
    
    def __init__(self, wsl_distribution: Optional[str] = None, 
                 working_directory: Optional[str] = None,
                 conda_env: Optional[str] = None):
        """
        Initialize the Deformetrica Registration Pipeline.
        
        Args:
            wsl_distribution: Specific WSL distribution to use (e.g., 'Ubuntu-20.04')
            working_directory: Working directory path in WSL format
            conda_env: Name of the conda environment where Deformetrica is installed
        """
        self.wsl_distribution = wsl_distribution
        self.working_directory = working_directory
        self.conda_env = conda_env
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
    def convert_windows_path_to_wsl(self, windows_path: str) -> str:
        """Convert Windows path to WSL path format."""
        # Ensure we have an absolute path first
        path = Path(windows_path).resolve()
        
        # Extract drive and path components
        drive = path.parts[0].replace(':', '').lower()
        path_parts = path.parts[1:]
        
        # Build WSL path
        if path_parts:
            wsl_path = f"/mnt/{drive}/" + "/".join(path_parts)
        else:
            wsl_path = f"/mnt/{drive}"
        
        # Clean up the path
        wsl_path = wsl_path.replace('\\', '/')
        
        return wsl_path
        
    def downsample_meshes(self, 
                         template_file: str, 
                         target_files: List[str], 
                         output_dir: str,
                         reduction_factor: float = 0.5,
                         output_suffix: str = "_downsampled") -> Tuple[str, List[str]]:
        """
        Downsample target meshes using VTK quadric decimation.
        Template file is left unchanged.
        
        Args:
            template_file: Path to template mesh file (will NOT be downsampled)
            target_files: List of paths to target mesh files (will be downsampled)
            output_dir: Directory to save downsampled files
            reduction_factor: Target reduction factor (0.5 = 50% reduction, default: 0.5)
            output_suffix: Suffix to add to downsampled files (default: "_downsampled")
            
        Returns:
            Tuple of (original_template_path, list_of_downsampled_target_paths)
        """
        try:
            import vtk
        except ImportError:
            self.logger.error("VTK is not installed. Install it with: pip install vtk")
            raise ImportError("VTK is required for mesh downsampling")
        
        downsampled_targets = []
        
        # Create output directory for downsampled files
        os.makedirs(output_dir, exist_ok=True)
        
        # Template file remains unchanged
        self.logger.info(f"Template file will NOT be downsampled: {template_file}")
        
        # Downsample only target files
        for target_file in target_files:
            self.logger.info(f"Downsampling target mesh: {target_file}")
            try:
                # Convert to absolute path to avoid path issues
                target_file = os.path.abspath(target_file)
                
                # Read VTK file
                if target_file.endswith('.vtk'):
                    reader = vtk.vtkPolyDataReader()
                elif target_file.endswith('.vtp'):
                    reader = vtk.vtkXMLPolyDataReader()
                else:
                    self.logger.warning(f"Unknown file format for {target_file}, assuming legacy VTK format")
                    reader = vtk.vtkPolyDataReader()
                
                reader.SetFileName(target_file)
                reader.Update()
                polydata = reader.GetOutput()
                
                # Get original mesh info
                original_points = polydata.GetNumberOfPoints()
                original_cells = polydata.GetNumberOfCells()
                self.logger.info(f"Original target - Points: {original_points}, Cells: {original_cells}")
                
                # Apply quadric decimation
                decimator = vtk.vtkQuadricDecimation()
                decimator.SetInputData(polydata)
                decimator.SetTargetReduction(reduction_factor)
                decimator.AttributeErrorMetricOn()
                decimator.VolumePreservationOn()
                decimator.Update()
                
                simplified_polydata = decimator.GetOutput()
                
                # Get simplified mesh info
                simplified_points = simplified_polydata.GetNumberOfPoints()
                simplified_cells = simplified_polydata.GetNumberOfCells()
                actual_reduction = (1 - simplified_cells / original_cells) * 100
                self.logger.info(f"Simplified target - Points: {simplified_points}, Cells: {simplified_cells}")
                self.logger.info(f"Target reduction: {actual_reduction:.1f}%")
                
                # Create downsampled filename in the specified output directory
                target_path = Path(target_file)
                downsampled_target = os.path.join(output_dir, f"{target_path.stem}{output_suffix}{target_path.suffix}")
                
                # Use absolute path for output as well
                downsampled_target = os.path.abspath(downsampled_target)
                
                # Write VTK file
                if target_file.endswith('.vtp'):
                    writer = vtk.vtkXMLPolyDataWriter()
                else:
                    writer = vtk.vtkPolyDataWriter()
                
                writer.SetFileName(downsampled_target)
                writer.SetInputData(simplified_polydata)
                writer.Write()
                
                # Verify file was created
                if os.path.exists(downsampled_target):
                    downsampled_targets.append(downsampled_target)
                    self.logger.info(f"Saved downsampled target: {downsampled_target}")
                else:
                    self.logger.error(f"Failed to create downsampled file: {downsampled_target}")
                    raise FileNotFoundError(f"Could not create downsampled file: {downsampled_target}")
                
            except Exception as e:
                self.logger.error(f"Failed to downsample target {target_file}: {e}")
                raise
        
        # Convert template to absolute path as well
        template_file = os.path.abspath(template_file)
        
        self.logger.info(f"Target mesh downsampling completed successfully")
        self.logger.info(f"Template (unchanged): {template_file}")
        self.logger.info(f"Downsampled targets: {downsampled_targets}")
        
        return template_file, downsampled_targets
        
    def _build_wsl_command(self, command_parts: list) -> list:
        """Build WSL command with conda environment activation."""
        cmd = ['wsl']
        
        if self.wsl_distribution:
            cmd.extend(['-d', self.wsl_distribution])
        
        command_str = ' '.join(command_parts)
        
        if self.conda_env:
            conda_init = "source ~/anaconda3/etc/profile.d/conda.sh"
            command_str = f"{conda_init} && conda activate {self.conda_env} && {command_str}"
            
        if self.working_directory:
            command_str = f"cd '{self.working_directory}' && {command_str}"
            
        cmd.extend(['-e', 'bash', '-c', command_str])
        return cmd

    def _find_warped_template_output(self, output_dir: str) -> Optional[str]:
        """
        Find the warped template VTK file produced by Deformetrica registration.
        Looks for: DeterministicAtlas__Reconstruction__bone_trial__subject_subject_0.vtk
        
        Args:
            output_dir: Directory where Deformetrica wrote its outputs
            
        Returns:
            Absolute path to the warped template VTK file, or None if not found
        """
        target_filename = "DeterministicAtlas__Reconstruction__bone_trial__subject_subject_0.vtk"
        candidate = os.path.join(output_dir, target_filename)
        
        if os.path.exists(candidate):
            self.logger.info(f"Found warped template output: {candidate}")
            return candidate
        
        # Fallback: glob for any Reconstruction vtk in the output dir
        pattern = os.path.join(output_dir, "*Reconstruction*.vtk")
        matches = glob.glob(pattern)
        if matches:
            result = sorted(matches)[-1]
            self.logger.warning(
                f"Expected '{target_filename}' not found; using fallback: {os.path.basename(result)}"
            )
            return result
        
        self.logger.error(
            f"❌ Could not find warped template output in: {output_dir}\n"
            f"   Expected: {target_filename}"
        )
        return None

    def _run_registration_step(self,
                               template_file: str,
                               target_file: str,
                               output_dir: str,
                               # Template object parameters
                               object_type: str = "SurfaceMesh",
                               attachment_type: str = "Varifold",
                               template_kernel_width: float = 20.0,
                               template_kernel_type: str = "keops",
                               template_noise_std: float = 1.0,
                               # Deformation parameters
                               deformation_kernel_width: float = 20.0,
                               deformation_kernel_type: str = "keops",
                               number_of_timepoints: int = 10,
                               # Optimization parameters
                               max_iterations: int = 100,
                               initial_step_size: float = 1e-3,
                               convergence_tolerance: float = 1e-6,
                               verbosity: str = "INFO",
                               freeze_template: bool = True,
                               gpu_mode: str = "auto",
                               save_every_n_iters: int = 100,
                               print_every_n_iters: int = 1) -> bool:
        """
        Execute a single Deformetrica registration step via WSL.
        
        Args:
            template_file: Windows path to the template mesh
            target_file: Windows path to the target mesh
            output_dir: Windows path to the output directory
            (all kernel/optimization parameters as per run_single_mesh_registration)
            
        Returns:
            True if successful, False otherwise
        """
        os.makedirs(output_dir, exist_ok=True)

        # Convert paths to WSL format
        wsl_template_file = self.convert_windows_path_to_wsl(template_file)
        wsl_target_file = self.convert_windows_path_to_wsl(target_file)

        wsl_output_dir = output_dir
        if ':' in output_dir:
            wsl_output_dir = self.convert_windows_path_to_wsl(output_dir)
        else:
            abs_output_dir = os.path.abspath(output_dir)
            wsl_output_dir = self.convert_windows_path_to_wsl(abs_output_dir)

        self.logger.info(f"WSL template path: {wsl_template_file}")
        self.logger.info(f"WSL target path:   {wsl_target_file}")
        self.logger.info(f"WSL output dir:    {wsl_output_dir}")

        # Debug: verify WSL can see the files
        self.logger.info("=== FILE EXISTENCE CHECK ===")
        for wsl_path in [wsl_template_file, wsl_target_file]:
            test_cmd = self._build_wsl_command(['ls', '-la', f'"{wsl_path}"'])
            try:
                result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    self.logger.info(f"[OK] WSL can see: {wsl_path}")
                else:
                    self.logger.error(f"[ERROR] WSL cannot see: {wsl_path}")
                    self.logger.error(f"ls error: {result.stderr}")
            except Exception as e:
                self.logger.error(f"Error checking {wsl_path}: {e}")

        # Generate and execute the Deformetrica Python script
        python_script = self._generate_deformetrica_python_script(
            template_file=wsl_template_file,
            target_files=[wsl_target_file],
            output_dir=wsl_output_dir,
            object_type=object_type,
            attachment_type=attachment_type,
            template_kernel_width=template_kernel_width,
            template_kernel_type=template_kernel_type,
            template_noise_std=template_noise_std,
            deformation_kernel_width=deformation_kernel_width,
            deformation_kernel_type=deformation_kernel_type,
            number_of_timepoints=number_of_timepoints,
            max_iterations=max_iterations,
            initial_step_size=initial_step_size,
            convergence_tolerance=convergence_tolerance,
            verbosity=verbosity,
            freeze_template=freeze_template,
            gpu_mode=gpu_mode,
            save_every_n_iters=save_every_n_iters,
            print_every_n_iters=print_every_n_iters
        )

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(python_script)
            temp_script_path = f.name

        try:
            wsl_script_path = self.convert_windows_path_to_wsl(temp_script_path)
            cmd = self._build_wsl_command(['python', wsl_script_path])

            self.logger.info(f"Executing Deformetrica Python API script...")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            self.logger.info("✅ Deformetrica registration step completed successfully")
            self.logger.info(f"STDOUT:\n{result.stdout}")

            if result.stderr:
                self.logger.warning(f"STDERR:\n{result.stderr}")

            return True

        except subprocess.CalledProcessError as e:
            self.logger.error(f"❌ Deformetrica registration step failed with return code {e.returncode}")
            self.logger.error(f"STDOUT:\n{e.stdout}")
            self.logger.error(f"STDERR:\n{e.stderr}")
            return False

        finally:
            try:
                os.unlink(temp_script_path)
            except:
                pass

    def run_single_mesh_registration(self,
                                     template_file: str,
                                     target_file: str,
                                     output_dir: str,
                                     # Iterative registration control
                                     iterative_reg: bool = False,
                                     # Kernel width parameters (scalar for single-step, list for iterative)
                                     deformation_kernel_width: Union[float, List[float]] = 20.0,
                                     template_kernel_width: Union[float, List[float]] = 20.0,
                                     # Mesh preprocessing options
                                     downsample_meshes: bool = False,
                                     downsample_reduction_factor: float = 0.5,
                                     # Template object parameters
                                     object_type: str = "SurfaceMesh",
                                     attachment_type: str = "Varifold",
                                     template_kernel_type: str = "keops",
                                     template_noise_std: float = 1.0,
                                     # Deformation parameters
                                     deformation_kernel_type: str = "keops",
                                     number_of_timepoints: int = 10,
                                     # Optimization parameters
                                     max_iterations: int = 100,
                                     initial_step_size: float = 1e-3,
                                     convergence_tolerance: float = 1e-6,
                                     verbosity: str = "INFO",
                                     freeze_template: bool = True,
                                     gpu_mode: str = "auto",
                                     save_every_n_iters: int = 100,
                                     print_every_n_iters: int = 1) -> bool:
        """
        Run Deformetrica registration for a single mesh pair.

        Supports two modes:
          - Single-step (iterative_reg=False): deformation_kernel_width and
            template_kernel_width are scalars (or single-element lists).
          - Iterative / coarse-to-fine (iterative_reg=True): both kernel width
            arguments must be equal-length lists. Each element defines one
            registration stage. The warped template output of stage N is used
            as the template input for stage N+1, keeping the original target
            fixed throughout.

        Args:
            template_file: Path to template mesh file
            target_file: Path to target mesh file
            output_dir: Base output directory for results
            iterative_reg: Enable iterative coarse-to-fine registration
            deformation_kernel_width: Scalar or list of deformation kernel widths
            template_kernel_width: Scalar or list of template kernel widths
            downsample_meshes: Whether to downsample the target mesh first
            downsample_reduction_factor: Quadric decimation target reduction (0–1)
            object_type: Deformetrica deformable object type
            attachment_type: Deformetrica attachment metric
            template_kernel_type: Kernel implementation for template metric
            template_noise_std: Noise standard deviation for template metric
            deformation_kernel_type: Kernel implementation for deformation
            number_of_timepoints: Integration steps for the geodesic
            max_iterations: Maximum optimiser iterations per step
            initial_step_size: Initial gradient-ascent step size
            convergence_tolerance: Optimiser convergence threshold
            verbosity: Logging level passed to Deformetrica
            freeze_template: Whether to freeze the template during registration
            gpu_mode: 'auto', 'on', or 'off'
            save_every_n_iters: Frequency of intermediate output saves
            print_every_n_iters: Frequency of log printing

        Returns:
            True if all steps succeeded, False otherwise
        """

        # ------------------------------------------------------------------
        # Validate and normalise kernel width arguments
        # ------------------------------------------------------------------
        if iterative_reg:
            if not isinstance(deformation_kernel_width, list):
                raise ValueError(
                    "iterative_reg=True requires deformation_kernel_width to be a list, "
                    f"got {type(deformation_kernel_width).__name__}"
                )
            if not isinstance(template_kernel_width, list):
                raise ValueError(
                    "iterative_reg=True requires template_kernel_width to be a list, "
                    f"got {type(template_kernel_width).__name__}"
                )
            if len(deformation_kernel_width) != len(template_kernel_width):
                raise ValueError(
                    f"deformation_kernel_width (len={len(deformation_kernel_width)}) and "
                    f"template_kernel_width (len={len(template_kernel_width)}) must have the same length"
                )
            def_kw_list = deformation_kernel_width
            tmpl_kw_list = template_kernel_width
        else:
            # Wrap scalars in a list so the loop below is unified
            def_kw_list = (
                deformation_kernel_width
                if isinstance(deformation_kernel_width, list)
                else [deformation_kernel_width]
            )
            tmpl_kw_list = (
                template_kernel_width
                if isinstance(template_kernel_width, list)
                else [template_kernel_width]
            )
            # For non-iterative mode use only the first element
            def_kw_list = [def_kw_list[0]]
            tmpl_kw_list = [tmpl_kw_list[0]]

        n_steps = len(def_kw_list)

        # ------------------------------------------------------------------
        # Optional mesh downsampling (applied once, before any iteration)
        # ------------------------------------------------------------------
        original_template = template_file
        original_target = target_file

        if downsample_meshes:
            self.logger.info("=== MESH DOWNSAMPLING STEP ===")
            try:
                template_file, downsampled_targets = self.downsample_meshes(
                    template_file=template_file,
                    target_files=[target_file],
                    output_dir=output_dir,
                    reduction_factor=downsample_reduction_factor,
                    output_suffix="_downsampled"
                )
                target_file = downsampled_targets[0]
                self.logger.info("✅ Mesh downsampling completed. Using downsampled meshes for registration.")
                self.logger.info(f"   Downsampled template: {template_file}")
                self.logger.info(f"   Downsampled target:   {target_file}")
            except Exception as e:
                self.logger.error(f"❌ Mesh downsampling failed: {e}")
                self.logger.info("Proceeding with original meshes...")
                template_file = original_template
                target_file = original_target
        else:
            self.logger.info("Skipping mesh downsampling (downsample_meshes=False)")

        # ------------------------------------------------------------------
        # Shared keyword arguments for every registration step
        # ------------------------------------------------------------------
        shared_kwargs = dict(
            target_file=target_file,
            object_type=object_type,
            attachment_type=attachment_type,
            template_kernel_type=template_kernel_type,
            template_noise_std=template_noise_std,
            deformation_kernel_type=deformation_kernel_type,
            number_of_timepoints=number_of_timepoints,
            max_iterations=max_iterations,
            initial_step_size=initial_step_size,
            convergence_tolerance=convergence_tolerance,
            verbosity=verbosity,
            freeze_template=freeze_template,
            gpu_mode=gpu_mode,
            save_every_n_iters=save_every_n_iters,
            print_every_n_iters=print_every_n_iters,
        )

        # ------------------------------------------------------------------
        # Registration loop
        # ------------------------------------------------------------------
        current_template = template_file

        for step_idx in range(n_steps):
            def_kw  = def_kw_list[step_idx]
            tmpl_kw = tmpl_kw_list[step_idx]

            if iterative_reg:
                step_label = f"iter_{step_idx + 1}_of_{n_steps}__defkw{def_kw}__tmplkw{tmpl_kw}"
                step_output_dir = os.path.join(output_dir, step_label)
                self.logger.info(
                    f"\n=== ITERATIVE REGISTRATION STEP {step_idx + 1}/{n_steps} "
                    f"| deformation_kw={def_kw}, template_kw={tmpl_kw} ==="
                )
            else:
                step_output_dir = output_dir
                self.logger.info(
                    f"\n=== DEFORMETRICA REGISTRATION STEP "
                    f"| deformation_kw={def_kw}, template_kw={tmpl_kw} ==="
                )

            self.logger.info(f"Template: {current_template}")
            self.logger.info(f"Output:   {step_output_dir}")

            success = self._run_registration_step(
                template_file=current_template,
                output_dir=step_output_dir,
                deformation_kernel_width=def_kw,
                template_kernel_width=tmpl_kw,
                **shared_kwargs
            )

            if not success:
                self.logger.error(
                    f"❌ Registration failed at step {step_idx + 1}/{n_steps}. "
                    f"Stopping iterative loop."
                )
                return False

            self.logger.info(f"✅ Registration step {step_idx + 1}/{n_steps} completed.")

            # Chain: use the warped template output as the next step's template
            if iterative_reg and step_idx < n_steps - 1:
                warped_template = self._find_warped_template_output(step_output_dir)
                if warped_template is None:
                    self.logger.error(
                        f"❌ Could not locate warped template output after step {step_idx + 1}. "
                        f"Cannot continue iterative registration."
                    )
                    return False
                self.logger.info(
                    f"   Chaining warped template for next step: {warped_template}"
                )
                current_template = warped_template

        return True

    def _generate_deformetrica_python_script(self, **kwargs) -> str:
        """
        Generate Python script content for Deformetrica API.
        """
        script_template = '''
import sys
import os
sys.path.insert(0, '/home/mskmorph/anaconda3/envs/deformetrica/lib/python3.8/site-packages')

from deformetrica.api import Deformetrica
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.{verbosity})

# Configure GPU/CPU
if "{gpu_mode}" == "auto":
    device = "cuda" if torch.cuda.is_available() else "cpu"
elif "{gpu_mode}" == "on":
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {{device}}")

try:
    # Create output directory
    output_dir = "{output_dir}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Deformetrica API
    deformetrica = Deformetrica(output_dir=output_dir)
    
    # Template specification
    template_specifications = {{
        "bone_trial": {{
            "deformable_object_type": "{object_type}",
            "attachment_type": "{attachment_type}",
            "kernel_width": {template_kernel_width},
            "kernel_type": "{template_kernel_type}",
            "noise_std": {template_noise_std},
            "filename": "{template_file}"
        }}
    }}
    
    # Dataset specification  
    dataset_specifications = {{
        "dataset_filenames": [
{target_files_formatted}
        ],
        "subject_ids": [f"subject_{{i}}" for i in range({num_subjects})]
    }}
    
    # Model specifications (deformation parameters)
    model_options = {{
        "deformation_kernel_width": {deformation_kernel_width},
        "deformation_kernel_type": "{deformation_kernel_type}",
        "number_of_timepoints": {number_of_timepoints},
        "freeze_template": {freeze_template},
        "use_rk2": False,
        "dimension": 3
    }}
    
    # Estimator specifications (optimization parameters)
    estimator_options = {{
        "optimization_method_type": "GradientAscent",
        "max_iterations": {max_iterations},
        "save_every_n_iters": {save_every_n_iters},
        "print_every_n_iters": {print_every_n_iters},
        "initial_step_size": {initial_step_size},
        "convergence_tolerance": {convergence_tolerance}
    }}
    
    print("Starting Deformetrica registration...")
    print(f"Template: {template_file}")
    print(f"Targets: {target_files_list}")
    print(f"Output directory: {{output_dir}}")
    
    # Use the correct API signature with proper parameter names
    if len({target_files_list}) == 1:
        print("Running registration (single template to single target)...")
        deformetrica.estimate_registration(
            template_specifications=template_specifications,
            dataset_specifications=dataset_specifications,
            model_options=model_options,
            estimator_options=estimator_options,
            write_output=True
        )
    else:
        print("Running deterministic atlas construction (multiple subjects)...")
        deformetrica.estimate_deterministic_atlas(
            template_specifications=template_specifications,
            dataset_specifications=dataset_specifications,
            model_options=model_options,
            estimator_options=estimator_options,
            write_output=True
        )
    
    print("Deformetrica analysis completed successfully!")
    
except Exception as e:
    print(f"Error during analysis: {{e}}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
        
        # Format target files for Python script - handle multiple subjects correctly
        target_files_list = kwargs['target_files']
        target_files_formatted = ""
        
        if len(target_files_list) == 1:
            # Single target file for registration
            target_files_formatted = f'            [{{"bone_trial": "{target_files_list[0]}"}}]'
        else:
            # Multiple target files for atlas construction - each subject gets its own entry
            for i, target_file in enumerate(target_files_list):
                target_files_formatted += f'            [{{"bone_trial": "{target_file}"}}]'
                if i < len(target_files_list) - 1:
                    target_files_formatted += ",\n"
        
        # Create format dictionary avoiding conflicts
        format_dict = kwargs.copy()
        format_dict['target_files_formatted'] = target_files_formatted
        format_dict['target_files_list'] = target_files_list
        format_dict['num_subjects'] = len(target_files_list)
        
        return script_template.format(**format_dict)


def find_mesh_pairs(aligned_meshes_dir: str) -> List[Tuple[str, str, str]]:
    """
    Find template-target mesh pairs from alignment output directory.
    
    Args:
        aligned_meshes_dir: Directory containing aligned meshes
        
    Returns:
        List of tuples: (mesh_name, template_aligned_path, target_original_path)
    """
    mesh_pairs = []
    
    # Find all template aligned files
    template_pattern = os.path.join(aligned_meshes_dir, "*_template_aligned.vtk")
    template_files = glob.glob(template_pattern)
    
    for template_file in template_files:
        # Extract mesh name from template file
        template_basename = os.path.basename(template_file)
        mesh_name = template_basename.replace("_template_aligned.vtk", "")
        
        # Find corresponding target file
        target_file = os.path.join(aligned_meshes_dir, f"{mesh_name}_target_original.vtk")
        
        if os.path.exists(target_file):
            mesh_pairs.append((mesh_name, template_file, target_file))
        else:
            print(f"⚠️ No target file found for {mesh_name}")
    
    return mesh_pairs


def run_batch_registration(aligned_meshes_dir: str,
                           registration_output_dir: str = "./registration_output",
                           conda_env: str = "deformetrica",
                           # Iterative registration control
                           iterative_reg: bool = False,
                           deformation_kernel_width: Union[float, List[float]] = 20.0,
                           template_kernel_width: Union[float, List[float]] = 20.0,
                           # Mesh preprocessing
                           downsample_meshes: bool = False,
                           downsample_reduction_factor: float = 0.5,
                           # Optimisation
                           max_iterations: int = 100,
                           verbose: bool = True) -> Dict[str, bool]:
    """
    Run Deformetrica registration for all mesh pairs found in aligned_meshes_dir.
    
    Args:
        aligned_meshes_dir: Directory containing aligned template and target meshes
        registration_output_dir: Base output directory for registration results
        conda_env: Conda environment name for Deformetrica
        iterative_reg: Enable iterative coarse-to-fine registration
        deformation_kernel_width: Scalar (single-step) or list (iterative) of
            deformation kernel widths. For iterative mode must be a list.
        template_kernel_width: Scalar (single-step) or list (iterative) of
            template kernel widths. Must match length of deformation_kernel_width
            when iterative_reg=True.
        downsample_meshes: Whether to downsample meshes before registration
        downsample_reduction_factor: Target reduction factor for downsampling
        max_iterations: Maximum optimisation iterations per registration step
        verbose: Whether to print detailed information
        
    Returns:
        Dictionary mapping mesh names to success status
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    # Log registration mode for clarity
    if iterative_reg:
        n_steps = len(deformation_kernel_width) if isinstance(deformation_kernel_width, list) else 1
        logger.info(
            f"Iterative registration enabled: {n_steps} step(s)\n"
            f"  deformation_kernel_width: {deformation_kernel_width}\n"
            f"  template_kernel_width:    {template_kernel_width}"
        )
    else:
        logger.info(
            f"Single-step registration\n"
            f"  deformation_kernel_width: {deformation_kernel_width}\n"
            f"  template_kernel_width:    {template_kernel_width}"
        )
    
    # Find all mesh pairs
    mesh_pairs = find_mesh_pairs(aligned_meshes_dir)
    
    if not mesh_pairs:
        logger.error(f"❌ No mesh pairs found in {aligned_meshes_dir}")
        return {}
    
    logger.info(f"Found {len(mesh_pairs)} mesh pair(s) for registration:")
    for mesh_name, template_file, target_file in mesh_pairs:
        logger.info(f"  {mesh_name}: {os.path.basename(template_file)} -> {os.path.basename(target_file)}")
    
    # Initialise Deformetrica pipeline
    deformetrica = DeformetricaRegistration(wsl_distribution="Ubuntu-2004", conda_env=conda_env)
    
    # Track results
    results = {}
    
    # Create base output directory
    os.makedirs(registration_output_dir, exist_ok=True)
    
    # Process each mesh pair
    for mesh_name, template_file, target_file in mesh_pairs:
        logger.info(f"\n=== Processing {mesh_name} ===")
        
        mesh_output_dir = os.path.join(registration_output_dir, mesh_name)
        
        start_time = time.time()
        
        try:
            success = deformetrica.run_single_mesh_registration(
                template_file=template_file,
                target_file=target_file,
                output_dir=mesh_output_dir,
                iterative_reg=iterative_reg,
                deformation_kernel_width=deformation_kernel_width,
                template_kernel_width=template_kernel_width,
                downsample_meshes=downsample_meshes,
                downsample_reduction_factor=downsample_reduction_factor,
                max_iterations=max_iterations
            )
            
            processing_time = time.time() - start_time
            
            if success:
                logger.info(f"✅ {mesh_name} registration completed successfully in {processing_time:.2f}s")
                logger.info(f"   Output saved to: {mesh_output_dir}")
            else:
                logger.error(f"❌ {mesh_name} registration failed after {processing_time:.2f}s")
            
            results[mesh_name] = success
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"❌ {mesh_name} registration failed with exception after {processing_time:.2f}s: {e}")
            results[mesh_name] = False
    
    # Summary
    successful = sum(results.values())
    total = len(results)
    logger.info(f"\n=== BATCH REGISTRATION SUMMARY ===")
    logger.info(f"Successfully processed: {successful}/{total} mesh pairs")
    
    if successful < total:
        logger.info("Failed mesh pairs:")
        for mesh_name, success in results.items():
            if not success:
                logger.info(f"  ❌ {mesh_name}")
    
    return results