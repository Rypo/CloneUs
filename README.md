# CloneUs

CloneUs provides as set of tools for cloning your Discord guildmates.

Specifically, it demonstrates LoRA finetuning for a multi-role, multi-turn chatbots and provides an example `server` built with [discord.py](https://github.com/Rapptz/discord.py) to unleash your creation.

## Features
- LoRA finetuning with [transformers](https://github.com/huggingface/transformers) + [peft](https://github.com/huggingface/peft)
- Optimized training and inference with [unsloth](https://github.com/unslothai/unsloth)
- Utilities for finetuning and running inference on open-source foundation, instruct-tuned, and chat-tuned models
- Customizable data chunking and chat windowing to help make the most of your data
- YouTube metadata parsing for richer chat context

## Getting Started
### System Requirements
- Nvidia GPU, Ampere or later (30xx, 40xx) 
- Linux or WSL for Windows  

### Prerequisites
- (required) [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) for downloading your server chat.
- (recommended) A [YouTube API key](https://developers.google.com/youtube/v3/getting-started) to enable YouTube link parsing.
- (optional) A [Discord bot account](https://discordpy.readthedocs.io/en/stable/discord.html) if you'd like to use your model on a Discord server.

### Setup
1. Create an environment and install base requirements. (Use `mamba` in place of `conda` if installed)
```bash
conda create -n cloneus python=3.10 packaging ninja -c conda-forge
conda activate cloneus
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install xformers -c xformers
```

2. Clone repo, install dependencies and package.
```bash
git clone https://github.com/Rypo/CloneUs.git
cd CloneUs
pip install -e ".[discord]" # For no server, `pip install -e .`. For quant methods (awq, gptq) `pip install -e ".[discord, quants]"`
```

3. Export discord chat with DiscordChatExporter. Select JSON format. Output path to `CloneUs/data/discord`. This will take a while for large chats. See [the docs](https://github.com/Tyrrrz/DiscordChatExporter/tree/master/.docs) for more information.
4. Run `python scripts/build.py data/discord/<YOUR_EXPORT_FILE>.json`
5. Update `.env` with your YouTube API key if you want YouTube links to be encoded into metadata for richer interactions.

### Training
1. edit `config/train_config.yaml` 
   - Update `chatlog_csv` to point to `data/discord/<YOUR_CHAT_EXPORT>.csv` created by `build.py`
   - If you want your bot to know first names, change `author_tag` to include "fname" (e.g. `"{author} ({fname})"`) and make sure you've included first names in your `config/users.json` 
2. Run `python scripts/train.py`

### Running the Discord Bot
- Update the `.env` with your `BOT_TOKEN`, `GUILDS_ID`, and `CHANNEL_ID` (and/or the `DEV_*` equivalents for a testing server) if you haven't already.
- Several commands depend on `authorInitial` being populated in `config/users.json`. Add an uncased unique character or char+digit(s) for each user. Avoid reserved chars {i,m,p,x} unless including a digit.
  - If you do not set these, they will assigned for you using first names if set, otherwise display names.

1. Add your trained model to `server/config/models.json`. Use the relative path to your trained checkpoint (e.g. `mistral.../chunk.../cnk8192-.../checkpoint-500`)
2. Run `python server/run.py` or `TESTING=1 python server/run.py` to run the testing server.


## Roadmap
- Generalize for applications beyond Discord
- Optional server hosting for trained models/discord bot
- RAG for YouTube links
- Dedicated per person LoRA training