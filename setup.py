from setuptools import find_packages, setup


setup(
    name="trade_assist",
    version="0.1.0",
    description="Technical analysis and configurable policy backtesting",
    packages=find_packages(),
    package_data={"trade_assist": ["schemas/*.json"]},
    include_package_data=True,
    install_requires=["yfinance", "pandas", "numpy", "matplotlib", "jsonschema"],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "trade-assist=trade_assist.cli:main",
        ]
    },
)
