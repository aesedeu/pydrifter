<h1 align="center">pydrifter</h1>
<p align="center"><b>An open-source framework to test and monitor ML models and systems.</b></p>
<p align="center">
<a href="https://pepy.tech/project/pydrifter" target="_blank"><img src="https://pepy.tech/badge/pydrifter" alt="PyPi Downloads"></a>
<a href="https://github.com/aesedeu/pydrifter/blob/main/LICENSE" target="_blank"><img src="https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square" alt="License"></a>
<a href="https://pypi.org/project/pydrifter/" target="_blank"><img src="https://img.shields.io/pypi/v/pydrifter" alt="PyPi"></a>

**pydrifter** is a lightweight, extensible Python library for detecting data drift between control and treatment datasets using statistical tests.  
It is designed for Data Scientists, ML Engineers, and Analysts working with production models and experiments (A/B tests, model monitoring, etc).

---

## 🚀 What is pydrifter?

`pydrifter` provides a unified interface for applying and analyzing statistical tests (e.g., KS-test, Wasserstein distance, PSI, Jensen-Shannon divergence) across multiple features in tabular datasets.

It is useful for:

- **A/B testing**: Detect whether experiment groups differ significantly.
- **Model monitoring**: Identify drift in features over time.
- **Data quality checks**: Validate dataset consistency before training or inference.

---

## 🛠️ Features

- 🧪 Plug-and-play statistical test classes with unified API  
- 📈 Visualizations for ECDF, KS-test distances, and histograms  
- 🧹 Preprocessing config with quantile filtering  
- 🧩 Easily extendable with your own test logic  
- ✅ Built-in logging, warnings, and tabulated results

---

## 📦 Installation

```bash
pip install pydrifter
```

---

## 👨‍💻 Example Usage

```python
import pandas as pd
from pydrifter import TableDriftChecker
from pydrifter.calculations import KS, Wasserstein, PSI
from pydrifter.config import DataConfig

# Define control and treatment datasets
control_df = pd.read_csv("data/control.csv")
treatment_df = pd.read_csv("data/treatment.csv")

# Configure features
data_config = DataConfig(
    numerical=["age", "salary", "click_rate"],
    categorical=["device_type"]
)

# Initialize drift checker
checker = TableDriftChecker(
    data_control=control_df,
    data_treatment=treatment_df,
    data_config=data_config
)

# Run statistical tests
checker.run_statistics(
    tests=[KS, Wasserstein, PSI],
    show_result=True
)
```

---

## 👥 Who is it for?

- Data Scientists running experiments or training models
- ML Engineers monitoring pipelines in production
- Analysts working with control/treatment comparisons

---

## 📚 Documentation

Soon

---

## 📄 License
APACHE License © 2025
Made with ❤️ by [Eugene C.]