from setuptools import setup, find_packages

setup(
    name="PharmaTree",
    version="0.1",
    description="Built using concepts from the IBM Machine Learning Course on Coursera.",
    author="Jorge Lima",
    author_email="jorgelima@gmx.us",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "streamlit",
        # Add any other required dependencies explicitly
    ],
    entry_points={
        "console_scripts": [
            # Define command-line scripts here if necessary
        ],
    },
)
