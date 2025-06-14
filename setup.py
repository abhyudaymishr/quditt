from setuptools import setup, find_packages

with open("readme.md", "r") as fh:
    long_description = fh.read()

setup(
    name="qudit",
    version="0.0.1",
    description="Quantum computing with qudits",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="plutoniumm",
    url="https://github.com/plutoniumm/qudit",
    license="MIT",
    package_dir={"": "."},
    packages=find_packages(where="."),
    python_requires=">=3.10",
    install_requires=["numpy", "scipy"],
    extras_require={
        "dev": ["pytest", "twine", "wheel"],
    },
)
