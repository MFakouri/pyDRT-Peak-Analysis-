from pathlib import Path
from setuptools import setup, find_packages

root = Path(__file__).parent
readme = (root/'README.md').read_text(encoding='utf-8')

setup(
    name='pyDRT-Peak-Analysis',
    version='0.4.0',
    author='Masood Fakouri Hasanabadi',
    description='DRT peak analysis tool for electrochemical impedance spectroscopy',
    long_description=readme,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'numpy>=1.24', 'scipy>=1.10', 'pandas>=1.5', 'matplotlib>=3.7',
        'scikit-learn>=1.2', 'PyQt5>=5.15.9', 'click>=8.1',
        'cvxopt>=1.3'
    ],
    entry_points={'console_scripts': ['launchGUI=pyDRTtools.cli:main']},
)
