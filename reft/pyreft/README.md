# PYREFT: Unified Representation Intervention Framework

A unified, well-organized framework for representation editing and intervention methods, combining the power of PyVene and ReFT into a single, coherent package.

## 🏗️ New Architecture Overview

This is a restructured version of the original pyreft package with improved organization and cleaner separation of concerns.

### 📁 Package Structure

```
pyreft/
├── __init__.py              # Unified entry point
├── setup.py                 # Package configuration
│
├── core/                    # Core intervention framework (formerly pyvene)
│   ├── __init__.py          # Core framework exports
│   ├── base.py              # IntervenableModel, IntervenableConfig
│   ├── interventions.py     # Base intervention classes
│   ├── utils.py             # Core utilities
│   └── modeling/            # Model-specific implementations
│       ├── common.py        # Common utilities
│       ├── gpt2.py          # GPT2 implementations
│       ├── llama.py         # LLaMA implementations
│       └── ...              # Other model architectures
│
├── reft/                    # REFT-specific implementations
│   ├── __init__.py          # REFT exports
│   ├── model.py             # ReftModel
│   ├── trainer.py           # ReftTrainer classes
│   ├── config.py            # ReftConfig
│   ├── utils.py             # REFT utilities
│   └── interventions.py     # REFT intervention methods
│
├── data/                    # Data processing and management
│   ├── __init__.py          # Data exports
│   ├── dataset.py           # Dataset classes
│   ├── causal_model.py      # Causal modeling (from pyvene)
│   └── preprocessing.py     # Data preprocessing utilities
│
├── analysis/                # Analysis and visualization
│   ├── __init__.py          # Analysis exports
│   ├── visualization.py     # Visualization tools
│   ├── evaluation.py        # Model evaluation
│   └── interpretation.py    # Result interpretation
│
├── examples/                # Examples and demonstrations
│   ├── __init__.py
│   ├── basic_demo.py        # Basic usage examples
│   └── notebooks/           # Jupyter notebooks
│       └── tutorial.ipynb   # Tutorial notebook
│
└── tests/                   # Test suite
    ├── __init__.py
    ├── test_core.py         # Core framework tests
    ├── test_reft.py         # REFT tests
    └── test_data.py         # Data processing tests
```

## 🚀 Key Improvements

### 1. **Clean Separation of Concerns**
- **Core framework**: Generic intervention capabilities (formerly pyvene)
- **REFT module**: Specific representation finetuning methods
- **Data module**: Unified data processing pipeline
- **Analysis module**: Evaluation and visualization tools

### 2. **Better Import Organization**
```python
# Before (confusing nested structure):
from pyreft.pyvene.models.interventions import TrainableIntervention
from pyreft import LoreftIntervention

# After (clean, logical structure):
from pyreft.core import TrainableIntervention  # Core framework
from pyreft.reft import LoreftIntervention     # REFT-specific
# Or simply:
from pyreft import TrainableIntervention, LoreftIntervention
```

### 3. **Unified Entry Point**
```python
import pyreft

# Access everything through a single, well-organized namespace
model = pyreft.ReftModel(config, base_model)
intervention = pyreft.LoreftIntervention(**kwargs)
dataset = pyreft.ReftDataset(data)
```

### 4. **Improved Extensibility**
- Easy to add new intervention methods in `reft/interventions.py`
- Simple to support new model architectures in `core/modeling/`
- Clear place for analysis tools in `analysis/`

## 📦 Installation

```bash
cd pyreft-new
pip install -e .
```

## 🧪 Testing the New Architecture

Run the provided test script to verify everything works:

```bash
cd pyreft-new
python test_new_architecture.py
```

## 🔧 Migration Guide

### For Users
The main imports remain largely the same, but are now better organized:

```python
# Core functionality
from pyreft import IntervenableModel, ReftModel

# Interventions
from pyreft import LoreftIntervention, TrainableIntervention

# Data processing
from pyreft import ReftDataset, ReftDataCollator
```

### For Developers
- Core framework code is now in `core/`
- REFT-specific code is in `reft/`
- Add new interventions to appropriate modules
- Use the unified `__init__.py` for exports

## 🎯 Benefits

1. **Single Package**: No more confusion about pyreft vs pyvene
2. **Clear Organization**: Logical separation by functionality
3. **Better Maintenance**: Easier to find and modify specific components
4. **Improved Testing**: Clear test structure for each module
5. **Enhanced Documentation**: Each module has clear responsibilities

## 📚 Next Steps

1. Update all import statements in your existing code
2. Run the test suite to ensure compatibility
3. Explore the new organized structure
4. Contribute improvements to the appropriate modules

---

This new architecture provides a solid foundation for future development while maintaining backward compatibility where possible. 