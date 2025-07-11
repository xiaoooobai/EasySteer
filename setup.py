from setuptools import setup, find_packages

# Version information
version = "0.1.0"  # Default version

# Build packages to install using find_packages with include patterns
packages = find_packages(include=["hidden_states*", "steer*", "reft*"])

# Project dependencies
install_requires = [
    "scikit-learn",
]

setup(
    name="easysteer",
    version=version,
    description="High-Performance LLM Steering Framework",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="ZJU-REAL",
    url="https://github.com/ZJU-REAL/EasySteer",
    packages=packages,
    python_requires=">=3.10",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="llm, steering, nlp, machine learning, deep learning",
) 