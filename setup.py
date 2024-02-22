from setuptools import setup, find_packages

setup(
    name = 'cloneus',
    version='0.0.1',
    author='Rypo',
    keywords="LLM, multi-user RP finetuning, discord, AI, chatbot",
    package_dir={"": "src"},
    packages = find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        'python-dotenv',
        'omegaconf',
        'more-itertools',
        'ujson',
        'ipython',
        'matplotlib',
        'google-api-python-client',
        'pandas',
        'bitsandbytes',
        'transformers',
        'datasets',
        'peft',
        'accelerate',
        'optimum',
        'safetensors',
        'trl',
        'wandb',
        'flash-attn',
        'unsloth[conda] @ git+https://github.com/unslothai/unsloth.git',
    ],
    extras_require={
        'discord': [ 
            'discord.py',
            # for image generation
            'diffusers', # do not use diffusers[torch] or it will try to install torch/triton=2.1.2
            'DeepCache'
        ],
        'quants': [ # this is fine with pytorch 2.2.0
            'auto-gptq',
            'autoawq',
            'exllamav2',
        ],
    },
    url='https://github.com/Rypo/CloneUs',
    license='MIT',
)