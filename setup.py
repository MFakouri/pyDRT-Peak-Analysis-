from pathlib import Path
from setuptools import setup, find_packages

root = Path(__file__).parent
readme = (root / 'README.md').read_text(encoding='utf-8')

# Keep package metadata synchronized with requirements.txt.
requirements = [
    line.strip()
    for line in (root / 'requirements.txt').read_text(encoding='utf-8').splitlines()
    if line.strip() and not line.lstrip().startswith('#')
]

setup(
    name='pyDRT-Peak-Analysis',
    version='0.4.0',
    author='Masood Fakouri Hasanabadi',
    description='DRT peak analysis tool for electrochemical impedance spectroscopy',
    long_description=readme,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    python_requires='>=3.11,<3.14',
    install_requires=requirements,
    entry_points={'console_scripts': ['launchGUI=pyDRTtools.cli:main']},
)
