from setuptools import find_packages, setup

with open("VERSION", "r") as f:
    version = f.read().strip()

with open("llmexport/version.py", "w") as f:
    f.write(f'__version__ = "{version}"\n')

setup(
    name="llmexport",
    version=version,
    description="llmexport: A toolkit to export llm to onnx or mnn.",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/wangzhaode/llm-export",
    author="wangzhaode",
    author_email="hi@zhaode.wang",
    project_urls={
        "Bug Tracker": "https://github.com/wangzhaode/llm-export/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    license="MIT",
    install_requires=["yaspin", "torch", "numpy", "transformers", "sentencepiece", "onnx", "onnxslim", "onnxruntime", "MNN"],
    packages=find_packages(exclude=("tests", "tests.*")),
    entry_points={"console_scripts": ["llmexport=llmexport:main"]},
    zip_safe=True,
    python_requires=">=3.6",
)