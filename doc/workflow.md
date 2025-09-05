# Development Workflow

*This document defines the development process for executing tasks from [tasklist.md](tasklist.md) according to [vision.md](../vision.md) principles.*

## Core Workflow (KISS Approach)

### Iteration Execution Cycle

**For each iteration in [tasklist.md](tasklist.md):**

1. **Plan Phase**
   - Review iteration goals and tasks
   - Propose solution with code snippets/architecture
   - **WAIT for user agreement** before implementing

2. **Implementation Phase**
   - Implement agreed solution following [conventions.md](../conventions.md)
   - Create all files and code as planned
   - Test functionality according to iteration test requirements

3. **Validation Phase**
   - Run all tests specified in iteration
   - Validate outputs make scientific sense
   - **WAIT for user confirmation** before proceeding

4. **Progress Update**
   - Mark completed tasks with ✅ in [tasklist.md](tasklist.md)
   - Update progress table with status and percentage
   - Document any issues or learnings

5. **Version Control**
   - Commit changes with descriptive message
   - Tag important milestones (e.g., `v1.0-project-setup`)

6. **Transition**
   - **Get user agreement** to move to next iteration
   - Start next cycle

## Research Experiment Iteration

**For ML experiment cycles:**

```
Design → Implement → Train → Evaluate → Document → Commit
```

**Process:**
1. **Design:** Propose model/data changes with rationale
2. **Implement:** Code changes following conventions
3. **Train:** Run experiments with proper seed setting
4. **Evaluate:** Analyze results, compare with baseline
5. **Document:** Update experiment logs and findings
6. **Commit:** Save working state with experiment ID

## Code Review and Validation

**Before implementation:**
- Present code structure and key functions
- Explain design decisions and trade-offs
- Get approval for approach

**After implementation:**
- Demonstrate working functionality
- Show test results and validation metrics
- Confirm scientific correctness

**ML Component Validation:**
- **Data:** Verify shapes, types, distributions
- **Models:** Check parameter counts, forward pass shapes
- **Training:** Monitor convergence, loss curves
- **Inference:** Validate predictions make sense

## Testing Requirements

**Each iteration must pass:**

**Unit Tests:**
```python
# Test individual functions
assert data_shape == expected_shape
assert model.forward(test_input).shape == expected_output
```

**Integration Tests:**
```bash
# Test end-to-end workflows  
python data_manager/process_data.py --config test_config.yaml
python training/train.py --config test_config.yaml --epochs 1
```

**Reproducibility Tests:**
```python
# Same config → same results
set_seed(42)
result1 = run_experiment(config)
set_seed(42) 
result2 = run_experiment(config)
assert torch.allclose(result1, result2)
```

## Documentation Updates

**After each iteration:**
- Update progress in [tasklist.md](tasklist.md)
- Document key learnings and issues
- Update any changed configurations
- Record performance metrics and baselines

**Research Progress:**
- Log experiment results in CSV format
- Save model summaries and hyperparameters
- Document successful/failed approaches
- Note reproducibility confirmations

## Version Control Practices

**Commit Strategy:**
```bash
git add .
git commit -m "feat: implement basic data manager (iteration 2)

- Add ROOT file reader with PyROOT
- Implement HDF5 writer with train/val/test splits  
- Create YAML config loading
- Test with sample ROOT file → verified data integrity

🤖 Generated with Claude Code
Co-Authored-By: Claude <noreply@anthropic.com>"
```

**Commit Types:**
- `feat:` - New feature implementation  
- `fix:` - Bug fixes
- `test:` - Add or modify tests
- `docs:` - Documentation updates
- `exp:` - Experiment results and configs

**Tagging Strategy:**
- Tag each completed iteration: `git tag v1.0-project-setup`
- Tag working models: `git tag exp-001-mlp-baseline`
- Tag major milestones: `git tag v1.0-end-to-end-pipeline`

## Reproducibility Checkpoints

**Before each commit:**
- [ ] Set and document random seeds
- [ ] Save exact config used
- [ ] Record environment info (Python/PyTorch versions)
- [ ] Test reproduction with saved config

**Data Versioning:**
- Track data processing configs in git
- Document ROOT file sources and versions
- Save data statistics and checksums
- Use consistent train/val/test splits

**Model Versioning:**
- Save model architecture and hyperparameters
- Track training configs and metrics
- Store best model weights
- Document evaluation results

## Workflow Rules

**Strict Requirements:**
1. **No implementation without agreement** - Always propose first
2. **No progression without validation** - Wait for confirmation
3. **One iteration at a time** - Complete current before next
4. **Test everything** - Every iteration must pass its tests
5. **Commit working states** - Only commit tested code
6. **Update progress** - Always mark completed tasks

**KISS Principles:**
- Implement minimal working solution first
- Add complexity only when validated
- Keep commits focused and atomic
- Document what works, not just what's planned

**Fail Fast:**
- Identify issues early in iteration
- Stop and fix problems before proceeding
- Don't accumulate technical debt
- Validate scientific correctness immediately

## Next Steps

Start with **Iteration 1: Project Setup** from [tasklist.md](tasklist.md):
1. Propose conda environment and folder structure
2. Wait for agreement
3. Implement setup
4. Test environment works
5. Commit and update progress
6. Get approval to proceed to Iteration 2