# Lower chi-square: a better track or a different one?

`mc_reco` muatm, 2000 events fitted (8 muons per event on the median).

Checks first.  Our functional evaluated at BARS' **own** track reproduces BARS' `fTimeChi2` to a median relative difference of 1.1e-05, so the functional, the hit set and the frame of the shift are right.

Of the two reported chi-squares, ours is the lower one in 55% of the events in which the two tracks differ -- but their median *ratio* is 1.0000, with quartiles 0.9998 and 1.0002.  The two statements are not in conflict: in most events both fits end in the same minimum and the ratio is 1 to a few parts in 10^4, which is what the median reports; the sign flips in a minority where the difference is large.

| subset | events | our track has the lower chi-square | chi-square ours / BARS' track | angle to truth ours / BARS (deg) | ours closer to truth |
|---|---|---|---|---|---|
| all | 2000 | 55% | 22.7 / 25.0 | 15.5 / 15.5 | 51% |
| one muon | 191 | 58% | 7.2 / 8.1 | 9.7 / 10.6 | 57% |
| one muon, directions differ by > 1 deg | 56 | 77% | 7.3 / 12.4 | 7.4 / 9.6 | 61% |
| directions differ by > 1 deg | 260 | 76% | 19.4 / 38.3 | 14.6 / 15.3 | 52% |
| directions differ by > 5 deg | 174 | 75% | 20.5 / 43.1 | 16.7 / 17.7 | 51% |

The `all` row is not a fair test of either fit: the truth is the primary
particle, the reconstruction is of the brightest track of the bundle, and the
two are not the same object.  The single-muon rows are the ones to read.
