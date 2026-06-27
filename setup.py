from setuptools import setup, find_packages

setup(
    name="trueskill-compression",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "pandas",
        "tqdm",
    ],
    python_requires=">=3.9",
)
