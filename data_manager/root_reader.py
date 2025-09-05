"""ROOT file reader for extracting physics data."""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

try:
    import ROOT
    # Disable ROOT's own command line argument parsing
    ROOT.PyConfig.IgnoreCommandLineOptions = True
    # Suppress ROOT info messages
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
except ImportError as e:
    logger.error(f"Failed to import ROOT: {e}")
    ROOT = None


class ROOTReader:
    """Read ROOT files and extract data according to configuration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize ROOT reader with configuration.
        
        Args:
            config: Configuration dictionary with data processing settings
        """
        if ROOT is None:
            raise ImportError("PyROOT not available")
            
        self.config = config
        logger.info("Initialized ROOT reader")
    
    def read_file(self, file_path: Path) -> Dict[str, np.ndarray]:
        """Read ROOT file and extract data according to config.
        
        Args:
            file_path: Path to ROOT file
            
        Returns:
            Dictionary mapping branch names to numpy arrays
            
        Raises:
            FileNotFoundError: If ROOT file doesn't exist
            RuntimeError: If ROOT file cannot be opened or tree not found
        """
        if not file_path.exists():
            raise FileNotFoundError(f"ROOT file not found: {file_path}")
        
        logger.info(f"Reading ROOT file: {file_path}")
        
        # Open ROOT file
        root_file = ROOT.TFile.Open(str(file_path))
        if not root_file or root_file.IsZombie():
            raise RuntimeError(f"Cannot open ROOT file: {file_path}")
        
        try:
            # Get tree
            tree_name = self.config['data'].get('tree_name', 'Events')
            tree = root_file.Get(tree_name)
            if not tree:
                available_keys = [key.GetName() for key in root_file.GetListOfKeys()]
                raise RuntimeError(
                    f"Tree '{tree_name}' not found in {file_path}. "
                    f"Available keys: {available_keys}"
                )
            
            # Extract data
            data = self._extract_data(tree)
            logger.info(f"Extracted {len(data)} branches with {tree.GetEntries()} events")
            
            return data
            
        finally:
            root_file.Close()
    
    def _extract_data(self, tree) -> Dict[str, np.ndarray]:
        """Extract data from ROOT tree.
        
        Args:
            tree: ROOT TTree object
            
        Returns:
            Dictionary mapping branch names to numpy arrays
        """
        data = {}
        branches = self.config['data'].get('branches', self._get_default_branches())
        n_entries = tree.GetEntries()
        
        if n_entries == 0:
            logger.warning("Tree contains no entries")
            return {}
        
        # Apply cuts if specified
        cuts = self.config['data'].get('cuts', {})
        
        for branch_name in branches:
            try:
                # Extract branch data
                branch_data = self._extract_branch(tree, branch_name, cuts)
                data[branch_name] = branch_data
                logger.debug(f"Extracted branch '{branch_name}': {branch_data.shape}")
                
            except Exception as e:
                logger.warning(f"Failed to extract branch '{branch_name}': {e}")
                continue
        
        return data
    
    def _extract_branch(self, tree, branch_name: str, 
                       cuts: Dict[str, Any]) -> np.ndarray:
        """Extract single branch with optional cuts.
        
        Args:
            tree: ROOT TTree object
            branch_name: Name of branch to extract
            cuts: Dictionary of cuts to apply
            
        Returns:
            Numpy array with branch data
        """
        branch_data = []
        
        for event in tree:
            try:
                value = getattr(event, branch_name)
                
                # Apply cuts
                if self._passes_cuts(event, cuts):
                    branch_data.append(float(value))
                    
            except AttributeError:
                logger.warning(f"Branch '{branch_name}' not found in event")
                branch_data.append(0.0)  # Default value
        
        return np.array(branch_data, dtype=np.float32)
    
    def _passes_cuts(self, event, cuts: Dict[str, Any]) -> bool:
        """Check if event passes all cuts.
        
        Args:
            event: ROOT tree event
            cuts: Dictionary of cuts {branch_name: {'min': val, 'max': val}}
            
        Returns:
            True if event passes all cuts
        """
        for branch_name, cut_values in cuts.items():
            try:
                value = getattr(event, branch_name)
                
                if 'min' in cut_values and value < cut_values['min']:
                    return False
                if 'max' in cut_values and value > cut_values['max']:
                    return False
                    
            except AttributeError:
                logger.debug(f"Cut branch '{branch_name}' not found, skipping cut")
                continue
        
        return True
    
    def _get_default_branches(self) -> List[str]:
        """Get default branch names for generic physics data.
        
        Returns:
            List of default branch names
        """
        return [
            'time',         # Time coordinate
            'x', 'y', 'z',  # Spatial coordinates  
            'amplitude',    # Signal amplitude
            'charge',       # Electric charge
            'energy',       # Energy measurement
            'particle_id'   # Particle type identifier
        ]
    
    def get_file_info(self, file_path: Path) -> Dict[str, Any]:
        """Get information about ROOT file contents.
        
        Args:
            file_path: Path to ROOT file
            
        Returns:
            Dictionary with file information
        """
        if not file_path.exists():
            raise FileNotFoundError(f"ROOT file not found: {file_path}")
        
        root_file = ROOT.TFile.Open(str(file_path))
        if not root_file or root_file.IsZombie():
            raise RuntimeError(f"Cannot open ROOT file: {file_path}")
        
        try:
            info = {
                'file_size': file_path.stat().st_size,
                'keys': [key.GetName() for key in root_file.GetListOfKeys()],
                'trees': []
            }
            
            # Get tree information
            for key in root_file.GetListOfKeys():
                obj = key.ReadObj()
                if obj.InheritsFrom("TTree"):
                    tree_info = {
                        'name': obj.GetName(),
                        'entries': obj.GetEntries(),
                        'branches': [branch.GetName() for branch in obj.GetListOfBranches()]
                    }
                    info['trees'].append(tree_info)
            
            return info
            
        finally:
            root_file.Close()