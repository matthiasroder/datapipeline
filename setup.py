from setuptools import setup

setup(
    name="datapipeline",
    version="0.1.0",
    description="Document Processing Pipeline",
    author="Matthias",
    py_modules=["pipeline"],
    install_requires=[
        "pdfplumber>=0.7.0",
        "python-docx>=0.8.11",
        "beautifulsoup4>=4.11.1",
        "markdownify>=0.11.0",
        "pandas>=1.5.0",
        "openai>=1.0.0",
        "requests>=2.28.0"
    ],
    entry_points={
        "console_scripts": [
            "pipeline=pipeline:main",
        ],
    },
)