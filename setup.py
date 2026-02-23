from setuptools import find_packages, setup


setup(
    name="trade_assist",
    version="0.1.0",
    description="Technical analysis and configurable policy backtesting",
    packages=find_packages(),
    package_data={"trade_assist": ["schemas/*.json"]},
    include_package_data=True,
    install_requires=["yfinance", "pandas", "numpy", "matplotlib", "jsonschema"],
    extras_require={
        "dev": ["pytest>=8.0", "black>=24.0", "mypy>=1.8", "flake8>=7.0"],
    },
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "trade-assist=trade_assist.cli:main",
        ]
    },
)
