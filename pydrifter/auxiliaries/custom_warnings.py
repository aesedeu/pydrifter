import warnings


def custom_warning(message, category, filename, lineno, file=None, line=None):
    print(f"{category.__name__}: {message}")
