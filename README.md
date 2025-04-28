<h1 align="center">pydrifter</h1>
<p align="center"><b>An open-source framework to test and monitor ML models and systems.</b></p>
<p align="center">
<a href="https://pepy.tech/project/pydrifter" target="_blank"><img src="https://pepy.tech/badge/pydrifter" alt="PyPi Downloads"></a>
<a href="https://github.com/aesedeu/pydrifter/blob/main/LICENSE" target="_blank"><img src="https://img.shields.io/badge/license-Apache%202.0-green?style=flat-square" alt="License"></a>
<a href="https://pypi.org/project/pydrifter/" target="_blank"><img src="https://img.shields.io/pypi/v/pydrifter" alt="PyPi"></a>
<a href="https://www.python.org/downloads/release/python-311/" target="_blank"><img src="https://img.shields.io/badge/python-3.11-blue.svg" alt="Python version"></a>

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
from pydrifter.config import TableConfig
from pydrifter.calculations import KLDivergence, PSI, Wasserstein
from pydrifter import TableDrifter

data_control = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'salary': [50000, 60000, 70000, 80000, 90000],
})

data_treatment = pd.DataFrame({
    'age': [26, 31, 36, 41, 46],
    'salary': [51000, 61000, 71000, 81000, 91000],
})

data_config = TableConfig(numerical=['age', 'salary'], categorical=[])

drifter = TableDrifter(
    data_control=data_control,
    data_treatment=data_treatment,
    data_config=data_config,
    tests=[KLDivergence, PSI, Wasserstein]

)

result, summary = drifter.run_statistics(show_result=True)

drifter.draw("age", quantiles=[0.05, 0.95])
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
Made with ❤️ by Eugene C.