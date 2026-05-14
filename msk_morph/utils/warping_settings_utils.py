#!/usr/bin/env python3
"""
Warping Settings File Utilities Module
Functions for creating participant-specific warping settings files.
"""

import os
import shutil
from pathlib import Path
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants matching main script
DEFAULT_TEMPLATE_WARPING_DIR = "./template_warping_files"
DEFAULT_TEMPLATE_FILENAME = "SettingsModelWarper_StationDefinedTemplateModel_HipJoints.xml"

class WarpingSettingsGenerator:
    """
    Generate participant-specific warping settings files from templates.
    """
    
    def __init__(self, verbose: bool = True):
        """
        Initialize the warping settings generator.
        
        Args:
            verbose: Enable verbose logging
        """
        self.verbose = verbose
        if not verbose:
            logger.setLevel(logging.WARNING)
    
    def create_participant_settings_file(self, 
                                    template_file_path: str,
                                    output_file_path: str, 
                                    participant_id: str) -> bool:
        """
        Create a participant-specific settings file from a template.
        
        Args:
            template_file_path: Path to the template XML file
            output_file_path: Path where the modified file should be saved
            participant_id: Participant ID to replace "ParticipantID" with
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if output file already exists
            if os.path.exists(output_file_path):
                if self.verbose:
                    logger.info(f"⚠️ SKIPPING warping settings file creation (file already exists): {output_file_path}")
                return True  # Return success since file exists
            
            # Check if template file exists
            if not os.path.exists(template_file_path):
                logger.error(f"❌ Template file not found: {template_file_path}")
                return False
            
            if self.verbose:
                logger.info(f"Creating participant settings file for ID: {participant_id}")
                logger.info(f"Template: {template_file_path}")
                logger.info(f"Output: {output_file_path}")
            
            # Create output directory if it doesn't exist
            output_dir = Path(output_file_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Read the template file
            with open(template_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if self.verbose:
                logger.info(f"✅ Read template file: {len(content)} characters")
            
            # Count occurrences of "ParticipantID" before replacement
            participant_id_count = content.count("ParticipantID")
            
            if participant_id_count == 0:
                logger.warning("⚠️ No 'ParticipantID' placeholders found in template file")
            else:
                if self.verbose:
                    logger.info(f"✅ Found {participant_id_count} 'ParticipantID' placeholders to replace")
            
            # Replace "ParticipantID" with the actual participant ID
            modified_content = content.replace("ParticipantID", participant_id)
            
            # Write the modified content to the output file
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            # Verify the replacement worked
            replaced_count = content.count("ParticipantID") - modified_content.count("ParticipantID")
            
            if self.verbose:
                logger.info(f"✅ Successfully replaced {replaced_count} occurrences of 'ParticipantID' with '{participant_id}'")
                logger.info(f"✅ Modified settings file saved to: {output_file_path}")
            
            # Verify the output file was created and has content
            if os.path.exists(output_file_path):
                file_size = os.path.getsize(output_file_path)
                if self.verbose:
                    logger.info(f"✅ Output file created successfully ({file_size:,} bytes)")
                return True
            else:
                logger.error("❌ Output file was not created")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error creating participant settings file: {e}")
            return False
    
    def validate_settings_file(self, settings_file_path: str, participant_id: str) -> bool:
        """
        Validate that a settings file has been properly customized for a participant.
        
        Args:
            settings_file_path: Path to the settings file to validate
            participant_id: Expected participant ID
            
        Returns:
            True if validation passes, False otherwise
        """
        try:
            if not os.path.exists(settings_file_path):
                logger.error(f"❌ Settings file not found: {settings_file_path}")
                return False
            
            with open(settings_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check that no "ParticipantID" placeholders remain
            remaining_placeholders = content.count("ParticipantID")
            if remaining_placeholders > 0:
                logger.error(f"❌ Settings file still contains {remaining_placeholders} 'ParticipantID' placeholders")
                return False
            
            # Check that the participant ID appears in the file
            participant_id_occurrences = content.count(participant_id)
            if participant_id_occurrences == 0:
                logger.error(f"❌ Participant ID '{participant_id}' not found in settings file")
                return False
            
            if self.verbose:
                logger.info(f"✅ Settings file validation passed:")
                logger.info(f"  ✅ No 'ParticipantID' placeholders remaining")
                logger.info(f"  ✅ Found {participant_id_occurrences} occurrences of '{participant_id}'")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error validating settings file: {e}")
            return False
    
    def backup_existing_settings(self, settings_file_path: str) -> bool:
        """
        Create a backup of an existing settings file before modification.
        
        Args:
            settings_file_path: Path to the settings file to backup
            
        Returns:
            True if backup successful or no file exists, False on error
        """
        if not os.path.exists(settings_file_path):
            if self.verbose:
                logger.info("✅ No existing settings file to backup")
            return True
        
        try:
            backup_path = f"{settings_file_path}.backup"
            shutil.copy2(settings_file_path, backup_path)
            
            if self.verbose:
                logger.info(f"✅ Created backup: {backup_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to create backup: {e}")
            return False

def create_warping_settings_file(participant_id: str,
                               template_directory: str = DEFAULT_TEMPLATE_WARPING_DIR,
                               template_filename: str = DEFAULT_TEMPLATE_FILENAME,
                               output_directory: Optional[str] = None,
                               create_backup: bool = True,
                               verbose: bool = True) -> bool:
    """
    Create a participant-specific warping settings file.
    
    Args:
        participant_id: Participant ID to use (REQUIRED - no default)
        template_directory: Directory containing template files (default: "./template_warping_files")
        template_filename: Name of template file (default: "SettingsModelWarper_StationDefinedTemplateModel_HipJoints.xml")
        output_directory: Output directory (default: ../participant_data/{participant_id}/warping_files)
        create_backup: Whether to backup existing settings file
        verbose: Enable verbose logging
        
    Returns:
        True if successful, False otherwise
    """
    generator = WarpingSettingsGenerator(verbose=verbose)
    
    # Set default output directory if not provided
    if output_directory is None:
        output_directory = f"../participant_data/{participant_id}/warping_files"
    
    # Construct full paths
    template_path = os.path.join(template_directory, template_filename)
    output_path = os.path.join(output_directory, template_filename)
    
    # Create backup if requested
    if create_backup:
        if not generator.backup_existing_settings(output_path):
            logger.warning("⚠️ Backup creation failed, continuing without backup")
    
    # Create the participant-specific settings file
    return generator.create_participant_settings_file(
        template_file_path=template_path,
        output_file_path=output_path,
        participant_id=participant_id
    )

def validate_warping_settings_file(settings_file_path: str,
                                 participant_id: str,
                                 verbose: bool = True) -> bool:
    """
    Validate a warping settings file.
    
    Args:
        settings_file_path: Path to the settings file
        participant_id: Expected participant ID
        verbose: Enable verbose logging
        
    Returns:
        True if validation passes, False otherwise
    """
    generator = WarpingSettingsGenerator(verbose=verbose)
    return generator.validate_settings_file(settings_file_path, participant_id)

def batch_create_settings_files(participant_ids: list,
                               template_directory: str = DEFAULT_TEMPLATE_WARPING_DIR,
                               template_filename: str = DEFAULT_TEMPLATE_FILENAME,
                               base_output_directory: str = "../participant_data",
                               verbose: bool = True) -> dict:
    """
    Create warping settings files for multiple participants.
    
    Args:
        participant_ids: List of participant IDs
        template_directory: Directory containing template files
        template_filename: Name of template file
        base_output_directory: Base directory for participant data
        verbose: Enable verbose logging
        
    Returns:
        dict: Results mapping participant_id to success status
    """
    results = {}
    
    if verbose:
        logger.info(f"Creating settings files for {len(participant_ids)} participants...")
    
    for participant_id in participant_ids:
        if verbose:
            logger.info(f"\n✅ Processing participant: {participant_id}")
        
        output_directory = f"{base_output_directory}/{participant_id}/warping_files"
        
        success = create_warping_settings_file(
            participant_id=participant_id,
            template_directory=template_directory,
            template_filename=template_filename,
            output_directory=output_directory,
            verbose=verbose
        )
        
        results[participant_id] = success
        
        if success:
            logger.info(f"✅ Successfully created settings for {participant_id}")
        else:
            logger.error(f"❌ Failed to create settings for {participant_id}")
    
    # Summary
    successful = sum(results.values())
    total = len(results)
    
    if verbose:
        logger.info(f"\n=== BATCH CREATION SUMMARY ===")
        logger.info(f"Successfully created: {successful}/{total} settings files")
        
        if successful < total:
            logger.info("Failed participants:")
            for participant_id, success in results.items():
                if not success:
                    logger.info(f"  ❌ {participant_id}")
    
    return results

if __name__ == "__main__":
    """Command line interface for warping settings file creation."""
    
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python warping_settings_utils.py <command> [args...]")
        print("\nCommands:")
        print("  create <participant_id> [template_dir] [output_dir]    - Create single settings file")
        print("  validate <settings_file> <participant_id>              - Validate settings file")
        print("  batch <participant_ids...>                             - Create multiple settings files")
        print("\nExamples:")
        print("  python warping_settings_utils.py create participant_id_1")
        print("  python warping_settings_utils.py create participant_id_2 ./template_warping_files ../participant_data/participant_id_2/warping_files")
        print("  python warping_settings_utils.py validate ../participant_data/participant_id_1/warping_files/SettingsModelWarper_StationDefinedTemplateModel_HipJoints.xml participant_id_1")
        print("  python warping_settings_utils.py batch participant_id_1 participant_id_2 participant_id_3")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "create":
        if len(sys.argv) < 3:
            print("❌ Usage: create <participant_id> [template_dir] [output_dir]")
            sys.exit(1)
        
        participant = sys.argv[2]
        template_dir = sys.argv[3] if len(sys.argv) > 3 else DEFAULT_TEMPLATE_WARPING_DIR
        output_dir = sys.argv[4] if len(sys.argv) > 4 else None
        
        # Create the settings file
        success = create_warping_settings_file(
            participant_id=participant,
            template_directory=template_dir,
            output_directory=output_dir,
            verbose=True
        )
        
        if success:
            print(f"✅ Successfully created warping settings file for participant {participant}")
        else:
            print(f"❌ Failed to create warping settings file for participant {participant}")
            sys.exit(1)
    
    elif command == "validate":
        if len(sys.argv) < 4:
            print("❌ Usage: validate <settings_file> <participant_id>")
            sys.exit(1)
        
        settings_file = sys.argv[2]
        participant = sys.argv[3]
        
        # Validate the settings file
        success = validate_warping_settings_file(
            settings_file_path=settings_file,
            participant_id=participant,
            verbose=True
        )
        
        if success:
            print(f"✅ Settings file validation passed for participant {participant}")
        else:
            print(f"❌ Settings file validation failed for participant {participant}")
            sys.exit(1)
    
    elif command == "batch":
        if len(sys.argv) < 3:
            print("❌ Usage: batch <participant_ids...>")
            sys.exit(1)
        
        participant_ids = sys.argv[2:]
        
        # Create settings files for all participants
        results = batch_create_settings_files(
            participant_ids=participant_ids,
            verbose=True
        )
        
        if all(results.values()):
            print(f"\n✅ Successfully created settings files for all {len(participant_ids)} participants!")
        else:
            print(f"\n⚠️ Created settings files for {sum(results.values())}/{len(participant_ids)} participants.")
            sys.exit(1)
    
    else:
        print(f"❌ Unknown command: {command}")
        sys.exit(1)