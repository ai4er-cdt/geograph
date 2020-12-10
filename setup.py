from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.1",
    author="Biodiversity Team",
    author_email="author@example.com",
    description="Group Team Challenge 2021; Biodiversity Team",
    url="url-to-github-page",
    packages=find_packages(),
    test_suite="src.tests.test_all.suite",
)
