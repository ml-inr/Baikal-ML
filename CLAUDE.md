# Claude Code Assistant Guidelines

*This file serves as the central hub for all development guidelines for the Neural Network Research Project.*

## Project Guidelines Overview

This project follows KISS (Keep It Simple, Stupid) principles for neural network research focused on binary classification of time-series data. All code generation and development decisions should reference these core documents:

### 📋 Core Documentation

**[vision.md](vision.md)** - Technical Architecture & Principles
- Technologies and development stack
- Project architecture and data pipeline
- Configuration management approach
- Experiment management strategy
- Primary reference for all technical decisions

**[conventions.md](conventions.md)** - Coding Standards
- Python and PyTorch coding patterns
- Data handling and configuration conventions
- Testing approaches for ML components
- Documentation and logging standards
- Follow these rules for all generated code

**[doc/workflow.md](doc/workflow.md)** - Development Process
- Iteration execution workflow
- Code review and validation checkpoints
- Version control and reproducibility practices
- Research experiment cycles
- Must follow this process for all development

**[doc/tasklist.md](doc/tasklist.md)** - Development Plan
- 8-iteration development roadmap
- Progress tracking and task management
- Testing requirements for each iteration
- Current project status and next steps

## Code Generation Rules

**ALWAYS follow these guidelines when generating code:**

### 🎯 Core Principles (from vision.md)
- **KISS**: Choose simplest solution that works
- **YAGNI**: Don't build features until needed
- **MVP**: Start with minimum to validate ideas
- **Fail Fast**: Quick experiments, early validation
- **Iterative**: Build incrementally based on results

### 🏗️ Technical Requirements (from conventions.md)
- Use Python 3.10+ with PyTorch
- Follow PEP 8 with 88-character line limit
- Type hints for all function signatures
- Config-driven everything (YAML files)
- Reproducible experiments (seed management)
- Modular design with clear separation

### 📁 Project Structure (from vision.md)
```
/data_manager    # ROOT → HDF5 conversion
/models         # PyTorch architectures  
/training       # Training loops & optimization
/inference      # Model loading & prediction
/experiments    # Configs & saved models
/notebooks      # Jupyter exploration
```

### 🔄 Development Process (from doc/workflow.md)
1. **Plan Phase**: Propose solution with code snippets
2. **Wait for Agreement**: Never implement without approval
3. **Implementation**: Follow conventions.md standards
4. **Validation**: Test according to iteration requirements
5. **Wait for Confirmation**: Get user approval before proceeding
6. **Progress Update**: Mark tasks complete in tasklist.md
7. **Commit**: Save working state with descriptive message

### 🧪 Research Focus
- **Primary Task**: Binary classification on time-series data
- **Data Flow**: ROOT files → HDF5 → PyTorch → trained models → predictions
- **Tracking**: ClearML experiment tracking from day 1
- **Reproducibility**: Same config → same results

## Implementation Checklist

Before generating any code, verify:

- [ ] **Architecture**: Follows vision.md structure and principles
- [ ] **Standards**: Meets conventions.md coding requirements  
- [ ] **Process**: Following workflow.md development cycle
- [ ] **Testing**: Includes validation for current iteration
- [ ] **Config**: Uses YAML configuration approach
- [ ] **Reproducibility**: Includes seed management and logging
- [ ] **KISS**: Simplest solution that achieves the goal

## Key Reminders

**Never do without user approval:**
- Implement code solutions
- Move to next development iteration
- Make architectural decisions
- Add complexity or features

**Always include:**
- Type hints and docstrings
- Error handling and logging
- Configuration loading from YAML
- Device handling (CPU/GPU)
- Reproducibility features (seeds, deterministic)

**Research-specific requirements:**
- Handle variable-length time series
- Binary classification metrics (AUC, F1, precision, recall)
- Model checkpointing and best model saving
- ClearML experiment tracking
- Separate train/val/test HDF5 files

## Current Status

Check [doc/tasklist.md](doc/tasklist.md) for current iteration progress and next steps in the development plan.

---

*Follow these guidelines consistently to maintain code quality, research reproducibility, and project coherence.*