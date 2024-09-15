# Description of raw .h5 files

Each h5 File has the following structure.

Root branches correspond to particles defined in the config file.

For each partile, next level branches are:

- coords_are_cluster_centered: 
Flag if OMs coordinates are shifted to center of each cluster.

- clusters_centers, (3, num_clusters): 
x, y, and z coordinates of clusters evaluated on MC sample, meters.

- ev_ids, (num_events):
Event uniqe id of the form <root_file_number>_<event_number_in_root_file>.
For multicluster events, the same second part is used.

- prime_prty, (num_events):
Properties of the primary particle (neutrino or cosmic ray):
0) polar angle
1) azimuth angle
2) energy
3) nucleon number
4) number of response muons
5) event weight

- raw, branch:
Branch for storing OMs data. See below.

- muons_prty, branch:
Branch for storing information on muons. See below. 

For each of this branches, the next one defines the initial root file in the format
part_<root_file_number>.

The data is stored in the next level branch named "data". 

## Raw branch:

### Event-wise data:

- cluster_ids, (num_evs):
ID of the cluster for the event

- num_un_strings, (num_evs):
Number of strings activated in event.

- ev_starts, (num_evs+1):
Used to navigate in 1D array for hit-wise data.
For i-th event, ev_starts[i] and ev_starts[i+1]-1 are the first and the last relevant elements in the coresponding data array.
For example, labels-ith-event = labels[ev_starts[i]:ev_starts[i+1]], where labels is 1D hit-wise array.

### Hit-wise data:

- channels, (num_hits):
Channels IDs trigerred in an event

- labels, (num_hits):
Labels indicating hit origin (noise, track, caskade). Uses the same magic numbers as the root file.

- data, (num_hits, 5):
Data registered by OMs. Note that data migh differ depending on the configuration parameters used for conversion to root files.
0) registered charge
1) time
234) x, y, z coordinates

- t_res, (num_hits):
Time residuals calculated directly during conversion.

## muons_prt branch:

- aggregate, (num_evs):
0) first muon time
1) sum of muons energies near detector

- mu_starts(num_evs+1):
Same as event starts, but for individual muons.

- individ:
0) polar angle
1) azimuth angle
234) coordinates of a point on track
5) times on track
6) energy (BMCEvent.fTracks.fMuonEnergy)

# Diagram

h5_file
│
├── particle_1 (muatm, nuatm, nue2)
│   ├── coords_are_cluster_centered (flag)
│   ├── clusters_centers (3, num_clusters)
│   ├── ev_ids (num_events)
│   ├── prime_prty (num_events, 6)
│   │   └── [polar_angle, azimuth_angle, energy, nucleon_number, num_response_muons, event_weight]
│   │
│   ├── raw
│   │   ├── cluster_ids (num_evs)
│   │   ├── num_un_strings (num_evs)
│   │   ├── ev_starts (num_evs+1)
│   │   ├── channels (num_hits)
│   │   ├── labels (num_hits)
│   │   ├── data (num_hits, 5)
│   │   │   └── [charge, time, x, y, z]
│   │   └── t_res (num_hits)
│   │
│   └── muons_prty
│       ├── aggregate (num_evs, 2)
│       │   └── [first_muon_time, sum_muon_energies]
│       ├── mu_starts (num_evs+1)
│       └── individ (num_muons, 7)
│           └── [polar_angle, azimuth_angle, x, y, z, time, energy]
│
├── particle_2
│   └── ... (same structure as particle_1)
│
└── particle_n
    └── ... (same structure as particle_1)

Note: For each branch, data is stored in part_<root_file_number>/data