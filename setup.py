from setuptools import setup, find_packages
from pathlib import Path


def parse_requirements(filename: str) -> list[str]:
    path = Path(filename)
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith(("#", "-"))
    ]


def read_readme() -> str:
    try:
        return Path("README.md").read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


setup(
    name="vihos",
    version="0.1.0",
    description="Vietnamese Hate Speech Detection using PhoBERT",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    author="Academic Project",
    license="MIT",
    python_requires=">=3.8",

    packages=find_packages(where="src"),
    package_dir={"": "src"},

    install_requires=parse_requirements("requirements.txt"),

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Natural Language :: Vietnamese",
    ],

    include_package_data=True,
)
