from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_desc = f.read()

setup(
    name="rl",
    version="0.0.1",
    author="Matthew Feng",
    author_email="matt@hypertext.sh",
    description="Implementations of various reinforcement learning algorithms.",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url="https://github.com/hypertextresearch/rl",
    packages=find_packages(),
    python_requires=">=3.6"
)