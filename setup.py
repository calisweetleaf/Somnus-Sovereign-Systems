from setuptools import setup, find_packages

setup(
    name="somnus-sovereign-systems",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # We'll leave this empty since requirements.txt is already installed
    ],
    author="Somnus Systems",
    description="Somnus Sovereign Systems - AI Operating System with Persistent VMs",
    python_requires=">=3.8",
)