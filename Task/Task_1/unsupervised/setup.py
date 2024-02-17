from setuptools import setup, find_packages

setup(
    name='unsupervised',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'seaborn',
        'matplotlib', 
    ],
    python_requires='>=3.6',
)