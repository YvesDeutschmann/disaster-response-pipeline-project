import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="disaster-response", 
    version="1.0.0",
    author="Yves Deutschmann",
    author_email="yves.deutschmann@gmail.com",
    description="Udacity's Disaster Response Pipeline project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/YvesDeutschmann/disaster-response-pipeline-project",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)