#!/usr/bin/env python3
"""
Script to check the number of events in the experimental HDF5 file.
"""

import h5py
import sys
from pathlib import Path

def check_h5_events(h5_path):
    """Check the number of events in an HDF5 file."""
    
    if not Path(h5_path).exists():
        print(f"❌ File not found: {h5_path}")
        return None
    
    try:
        with h5py.File(h5_path, 'r') as f:
            print(f"📂 Analyzing: {h5_path}")
            print("=" * 60)
            
            # Check what particle types are available
            particle_types = list(f.keys())
            print(f"🔍 Available particle types: {particle_types}")
            print()
            
            total_events = 0
            
            for particle in particle_types:
                print(f"📊 Particle: {particle}")
                
                # Look for event boundary arrays
                if f"{particle}/raw/ev_starts" in f:
                    # Check for parts
                    ev_starts_group = f[f"{particle}/raw/ev_starts"]
                    parts = list(ev_starts_group.keys())
                    
                    particle_total = 0
                    print(f"   Parts found: {len(parts)}")
                    
                    for part in parts:
                        ev_starts = f[f"{particle}/raw/ev_starts/{part}/data"][:]
                        n_events = len(ev_starts) - 1  # ev_starts has n_events + 1 elements
                        particle_total += n_events
                        print(f"   {part}: {n_events:,} events")
                    
                    print(f"   Total for {particle}: {particle_total:,} events")
                    total_events += particle_total
                
                else:
                    print(f"   ❌ No ev_starts found for {particle}")
                
                print()
            
            print("=" * 60)
            print(f"🎯 TOTAL EVENTS: {total_events:,}")
            
            return total_events
            
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

def main():
    """Main function."""
    h5_path = "/home/albert/Baikal2025/data_manager/h5datasets/exp.h5"
    
    print("🔬 Checking experimental data HDF5 file...")
    print()
    
    total_events = check_h5_events(h5_path)
    
    if total_events is not None:
        print()
        print("📋 Summary:")
        print(f"   File: {h5_path}")
        print(f"   Total events: {total_events:,}")
        
        # Compare with configuration
        config_events = 1500  # From da_neutrino_baseline.yaml
        print(f"   Config requests: {config_events:,} events")
        
        if total_events >= config_events:
            print(f"   ✅ Sufficient data available ({total_events:,} >= {config_events:,})")
        else:
            print(f"   ⚠️  Insufficient data ({total_events:,} < {config_events:,})")
            print(f"   Consider reducing events_per_particle.exp to {total_events}")
    
    return total_events

if __name__ == '__main__':
    main()