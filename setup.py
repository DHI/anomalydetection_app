import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="anomalydetection_app",
    version='0.0.1',
    install_requires=["dash>=1.18.1", "plotly>=4.14.3", "pandas>=1.2.0", "dash-bootstrap-components>=0.11.1",
                      "anomalydetection", "numpy>=1.19.5", "python_dateutil>=2.8.1"],
    dependency_links=["https://github.com/DHI/anomalydetection.git"],
    extras_require={
        "dev": ["pytest>=6.2.1"],
        "test": ["pytest>=6.2.1"],
    },
    author="Laura Froelich",
    author_email="lafr@dhigroup.com",
    description="Time series anomaly detection experimentation and visualization.",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DHI/anomalydetection_app",
    packages=setuptools.find_packages(),
    include_package_data=True,
)
