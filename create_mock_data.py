"""Create mock ROOT file for testing data processing pipeline."""

import logging
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import ROOT
    ROOT.gROOT.SetBatch(True)
    ROOT.gErrorIgnoreLevel = ROOT.kWarning
except ImportError:
    logger.error("PyROOT not available - cannot create mock data")
    exit(1)


def create_mock_root_file(output_path: Path, n_events: int = 1000) -> None:
    """Create mock ROOT file with physics-like data.
    
    Args:
        output_path: Path where to save the ROOT file
        n_events: Number of events to generate
    """
    logger.info(f"Creating mock ROOT file with {n_events} events: {output_path}")
    
    # Create ROOT file and tree
    root_file = ROOT.TFile(str(output_path), "RECREATE")
    tree = ROOT.TTree("Events", "Mock physics events")
    
    # Define branch variables (using arrays for ROOT compatibility)
    time = np.array([0.0], dtype=np.float32)
    x = np.array([0.0], dtype=np.float32)
    y = np.array([0.0], dtype=np.float32)
    z = np.array([0.0], dtype=np.float32)
    amplitude = np.array([0.0], dtype=np.float32)
    charge = np.array([0.0], dtype=np.float32)
    energy = np.array([0.0], dtype=np.float32)
    particle_id = np.array([0], dtype=np.int32)
    
    # Create branches
    tree.Branch("time", time, "time/F")
    tree.Branch("x", x, "x/F")
    tree.Branch("y", y, "y/F") 
    tree.Branch("z", z, "z/F")
    tree.Branch("amplitude", amplitude, "amplitude/F")
    tree.Branch("charge", charge, "charge/F")
    tree.Branch("energy", energy, "energy/F")
    tree.Branch("particle_id", particle_id, "particle_id/I")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate mock data
    for i in range(n_events):
        # Time: exponential distribution (typical for physics processes)
        time[0] = np.random.exponential(scale=10.0)
        
        # Spatial coordinates: spherical distribution
        r = np.random.exponential(scale=50.0)  # Distance from origin
        theta = np.random.uniform(0, 2 * np.pi)  # Azimuthal angle
        phi = np.random.uniform(0, np.pi)  # Polar angle
        
        x[0] = r * np.sin(phi) * np.cos(theta)
        y[0] = r * np.sin(phi) * np.sin(theta)
        z[0] = r * np.cos(phi)
        
        # Amplitude: log-normal distribution (typical for signal amplitudes)
        amplitude[0] = np.random.lognormal(mean=2.0, sigma=1.0)
        
        # Charge: discrete values with some noise
        base_charge = np.random.choice([-1, 0, 1], p=[0.3, 0.4, 0.3])
        charge[0] = base_charge + np.random.normal(0, 0.1)
        
        # Energy: related to amplitude with some physics-like correlation
        energy[0] = amplitude[0] * np.random.lognormal(mean=0.0, sigma=0.5)
        
        # Particle ID: discrete particle types
        particle_types = [11, 13, 22, 211, 2212]  # electron, muon, photon, pion, proton
        particle_id[0] = np.random.choice(particle_types)
        
        # Fill tree
        tree.Fill()
    
    # Write and close file
    tree.Write()
    root_file.Close()
    
    logger.info(f"Successfully created mock ROOT file: {output_path}")
    
    # Print some statistics
    logger.info("Mock data statistics:")
    logger.info(f"  Events: {n_events}")
    logger.info(f"  File size: {output_path.stat().st_size / 1024:.1f} KB")


def verify_mock_file(file_path: Path) -> None:
    """Verify the created mock file can be read.
    
    Args:
        file_path: Path to ROOT file to verify
    """
    logger.info(f"Verifying mock ROOT file: {file_path}")
    
    root_file = ROOT.TFile.Open(str(file_path))
    if not root_file or root_file.IsZombie():
        logger.error(f"Cannot open file: {file_path}")
        return
    
    tree = root_file.Get("Events")
    if not tree:
        logger.error("Cannot find Events tree")
        root_file.Close()
        return
    
    n_entries = tree.GetEntries()
    logger.info(f"  Tree entries: {n_entries}")
    
    # Get branch names
    branches = [branch.GetName() for branch in tree.GetListOfBranches()]
    logger.info(f"  Branches: {branches}")
    
    # Sample a few events
    if n_entries > 0:
        logger.info("  Sample events:")
        for i in range(min(3, n_entries)):
            tree.GetEntry(i)
            event_info = []
            for branch_name in branches:
                value = getattr(tree, branch_name)
                event_info.append(f"{branch_name}={value:.3f}")
            logger.info(f"    Event {i}: {', '.join(event_info)}")
    
    root_file.Close()
    logger.info("Mock file verification complete")


def main():
    """Main function to create mock data."""
    # Create mock ROOT file
    mock_file_path = Path("mock_data.root")
    
    try:
        create_mock_root_file(mock_file_path, n_events=1000)
        verify_mock_file(mock_file_path)
        
        logger.info("Mock data creation completed successfully")
        logger.info(f"Update your config file to use: {mock_file_path.absolute()}")
        
    except Exception as e:
        logger.error(f"Failed to create mock data: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())