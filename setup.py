from setuptools import setup, find_packages

setup(
    name = 'cloneus',
    version='0.0.1',
    author='Rypo',
    keywords="LLM, multi-user RP finetuning, discord, AI, chatbot",
    package_dir={"": "src"},
    packages = find_packages(where="src"),
    url='https://github.com/Rypo/CloneUs',
    license='MIT',
)