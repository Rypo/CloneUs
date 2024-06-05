# CloneUs

CloneUs provides a framework for efficiently creating chat clones of your friends, colleagues, guildmates, enemies.. whomever you have sufficient chat data for, really.

Specifically, it assists with finetuning LLMs for multi-role, multi-turn conversations on consumer-grade hardware.

## Features
- LoRA finetuning with [transformers](https://github.com/huggingface/transformers) + [peft](https://github.com/huggingface/peft)
- Optimized training and inference with [unsloth](https://github.com/unslothai/unsloth)
- Supports the latest open source models like Llama3, Phi-3, Mistral v0.3, etc.
- Utilities for finetuning and running inference on open-source foundation, instruct-tuned, and chat-tuned models
- Customizable data chunking and chat windowing to help make the most of your data
- Support for [text-generation-webui](https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab#parameters-description) sampling techniques such as `mirostat`, `dynamic_temperature`, among others
- YouTube metadata parsing for richer chat context.
- A fully functional discord `server` bot built with [discord.py](https://github.com/Rapptz/discord.py) to serve up your clones

### System Requirements
- Nvidia GPU, Ampere or later (30xx, 40xx) 
- Linux or WSL for Windows
  
## Setup
Create an environment and install base requirements. (Use `mamba` in place of `conda` if available)
```bash
conda create -n cloneus python=3.10 packaging ninja -c conda-forge
conda activate cloneus
conda install pytorch-cuda=12.1 pytorch xformers -c pytorch -c nvidia -c xformers
```
Clone repo, install dependencies and package.
```bash
git clone https://github.com/Rypo/CloneUs.git
cd CloneUs
pip install -e ".[discord]" # For no server, `pip install -e .`. For quant methods (awq, gptq) `pip install -e ".[discord, quants]"`
```


### Zero Start (no data, no training)
  <details>
  <summary>You can try out generation without first finetuning, though this is not recommended.</summary>
  
  ```python
  # Expect generic, low-quality responses with this method
  from cloneus import Cloneus
  model_id = 'NousResearch/Hermes-2-Theta-Llama-3-8B'
  author_names = ['JohnnyFish','Balbo23','sstaaph','ItonightI']

  clo = Cloneus.from_model_id(model_id, author_names=author_names)

  chat_history = [
    ('JohnnyFish', 'blub blub'),
    ('Balbo23', 'yeah, I was gonna say fundamentally the same thing'),]

  for nt in clo.stream_generate(chat_history,'JohnnyFish'):
      print(nt, end='')
  ```
  </details>


## Quickstart
Export your chat data into a `.csv` with one of two formats below and place in `data/chat/`.

**Non-Discord Data**
Required columns: `username`, `text`, `timestamp`*

*`timestamp` _is technically optional, though some functionality will disabled if it is missing_.

**Discord Data**
Required columns: `AuthorID`, `Author`, `Date`, `Content`
> see [Discord Setup](#discord-setup) for more on how to obtain this data from a Discord server.

### Usage
To train using the default [training config](config/train/train_config.yaml), its as simple as:
```bash
# train with defaults
python scripts/train.py -d data/chat/YOUR_CHAT_LOG.csv
```
While your model trains, checkpoints are periodically saved to `runs/full/.../..checkpoint-xxx`. Once your done training, select one and give your chatbot a spin!
```python
from cloneus import Cloneus

checkpoint = runs/full/.../..checkpoint-xxx
clo = Cloneus.from_pretrained(checkpoint_path=checkpoint)

# Context to feed the model. Pairs of (author,message) where author is a user in your data, message is anything.
chat_history = [
  ('aboardLlama3', 'whatdup'),
  ('BillyTheAdult', 'Same old same old'),]

# The author you'd like to generate a response to the chat for 
next_author = 'nextinLinus'

# stream out the response
for nt in clo.stream_generate(chat_history, next_author):
    print(nt, end='')
```
<!-- If anyone ever sees this I need you to know that "aboardLlama3" was the first randomly generated username I rolled...Llama3...seriously -->

## Better Start (preferred method)

### Building the user index
The user index is a `.json` file containing all the users in your data. It is referenced often throughout the library, so getting it in shape early pays off later.

```bash
python scripts/build.py data/chat/YOUR_CHAT_LOG.csv # or .json for Discord, but more on that later.
```

This script lets you add more information about your users that can't be extracted automatically, such as first names, existing bots in the data, and users that should be excluded from training. See the `--help` for options.

After you've gone through the setup, double-check your `config/users.json` and make sure it's to your liking.  

### Configuring .env
If you are not using the Discord bot, there's only 1 important item in the `.env` file: `YOUTUBE_API_KEY`

Optionally, fill with a [YouTube API key](https://developers.google.com/youtube/v3/getting-started) to enable youtube link parsing:
- YouTube urls are converted into text metadata during training and inference to help generate better responses.
- All youtube links generated by the bot will be valid video links.
- A cache of youtube video data is saved to `data/youtube/` to avoid exhausting your daily free-tier limit on repeat calls.

This is highly recommended if your chat frequently sends youtube links. The daily free tier quota is 10,000 credits which is enough to parse 50k links (batched) or generate 100 links.

If you do plan to use the discord server bot, see [Discord Setup](#discord-setup) for other required `.env` values.

## Training Config
You may directly modify the default [training config](config/train/train_config.yaml) or duplicate and edit the copy. To train with a different config path, use the `-c` argument. e.g.
```bash
python scripts/train.py -c config/train/my_custom_config.yaml
```
The training config is the bread and butter of the library. With few exceptions, all roads lead from here. The layout takes inspiration from [axolotl](https://github.com/OpenAccess-AI-Collective/axolotl) and with aims to be as flexible as possible, there are a similarly large number of options. That said, most are "set and forget" with only a small handful changing between runs.

These are the most important options:

* Model
  ```yaml
  model_id: NousResearch/Hermes-2-Theta-Llama-3-8B # huggingface repo
  flashattn_lib: unsloth # huggingface, unsloth
  # - Most, but not all models are supported by unsloth. Switch to "huggingface" if unsupported.
  ```

* Training
  ```yaml
  chunk_size: 8192 # Upper token limit on conversation length. If a conversation exceeds this limit, it will be split into new groups.
  batch_size: 2
  gradient_accumulation_steps: 4
  ```
  - If you're spilling over too much into RAM, or are just running out all types of memory, these are the knobs to tune. Prefer matching `chunk_size` to the model context size whenever possible, however.

* Dataset
  ```yaml
  chatlog_csv: 'data/chat/YOUR_CHAT_FILE.csv' # change this to avoid having to call -d data/chat/my_data.csv every time.
  hours_between_sessions: 4 # [1, 3, 5]
  # hours of silence before we start a new chat chunk. Can be a list to use multiple cutoffs.
  min_session_length: 3
  # Minimum number of messages in a chat chunk. Any chunks smaller will be forcibly regrouped.
  ```
  - `hours_between_sessions` serves are pseudo end of topic marker. Conversation boundaries are drawn between periods of inactivity. This is the option that controls those boundaries. **Tweak this** until you find the sweet spot for your data.
  - `min_session_length` is the lower bound on conversation length. After all, a conversation with 1 message isn't much of a conversation.

* Message Format
  ```yaml
  tag_placement: replace_role # tag_only, content_prefix, replace_role
  # - replace_role: Replaces user/assistant with formatted author_tag
  # - content_prefix: Inserts author_tag before text content
  # - tag_only: Ignores any existing template and creates one as {author_tag}{tag_sep}CONTENT{postfix}.

  author_tag: "{author}" # -- used for all tag_placement values
  # - For replace_role format, "{author} ({fname})" works well if you added firstNames
  # - fstring options: author=displayName, fname=firstName, lauthor=lower(displayName)

  tag_sep: # "\n" # Ignored when tag_placement="replace_role", otherwise "\n" or " : " work well
  # Separator between author_tag and start of text

  postfix: # "</s>" # ONLY used when tag_placement="tag_only"
  # Separator between end of text and the next author_tag

  custom_chat_template: "{{bos_token}}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>' }}{% endif %}"
  # For role_replace: Be sure to remove any mention of "assistant" from your custom_chat_template as shown
  # - or use shorthand "chatml" to use above AND add tokens (<|im_start|>, <|im_end|>, ..) to the model/tokenizer if not already present.
  ```
  - These depend heavily on the model you are finetuning. As a general rule of thumb:
    - Use `replace_role`: if user/assistant are strings (chatml format, llama-3), not standalone tokens
    - Use `content_prefix`: if user/assistant are tokens (phi-3) or have forced order pattern (Mistral)
    - Use `tag_only`: if you are finetuning a foundation model, not an instruct/chat model.

* Prompt
  ```yaml
  # replace_role example
  template: |-
    Uncensored Discord server chat.
    Topic: offbeat.
    Participants: {name_mapping}
  append_msg: # blank for replace_role
  name_mapping: null # Will be filled during scripts/train.py
  ```
  ```yaml
  # content_prefix example
  template: |-
    Simulate a conversation between members of a group chat. 
    Each turn, you will respond as one of the members. Begin your response with the name tag of who you are speaking as.
    name tags options: {name_mapping}.
    ---
    If you understand the objective, say "{append_msg}", then we will begin.
  append_msg: "OK" # For content_prefix with system-less models. If set, system prompt will comprise 2 messages, a user "system" message and an assistant follow-up containing `append_msg`, for example "OK" 
  name_mapping: null # Will be filled during scripts/train.py
  ```
  - This is your system prompt. It is only used by instruct/chat tuned models. Since we are finetuning, it does not need to be extremely elaborate, but smaller datasets especially can benefit from the added guidance.
  - For `content_prefix`, if the model does not support a 'system' role, it will be inserted as the first 'user' message. If set, `append_msg` will be the first 'assistant' message.

## Discord Setup
Read this section if you either want to use Discord as data source or want to user the Discord server bot.

### Prerequisites
- [DiscordChatExporter](https://github.com/Tyrrrz/DiscordChatExporter) for downloading discord server chat.
- A [Discord bot account](https://discordpy.readthedocs.io/en/stable/discord.html) if you'd like to use your model on a Discord server.

### Extracting Discord Data
Required columns: `AuthorID`, `Author`, `Date`, `Content`

There are two ways to get a `.csv` with the required columns using DiscordChatExporter.
1. Choose the export format as `CSV`. 
2. Choose export format as `JSON` and run `scripts/build.py`. This is the preferred method.

In DiscordChatExporter
- Enter your token
- Select the channel to export
- set Output path: `data/chat/`
- set Export format: JSON
- set Format markdown if not already

See [the docs](https://github.com/Tyrrrz/DiscordChatExporter/tree/master/.docs) for a more detailed guide and additional information.

Once the export is complete, build the user index and convert to a csv:
```bash
python scripts/build.py data/chat/YOUR_DISCORD_EXPORT_FILE.json
```

During the conversion, you will see `GUILDS_ID` and `CHANNEL_ID` printed in your console. Add these to your `.env` as either 
- `GUILDS_ID` and `CHANNEL_ID` or 
- `DEV_GUILDS_ID` and `DEV_CHANNEL_ID`

Optionally, fill `YOUTUBE_API_KEY` as well with a [YouTube API key](https://developers.google.com/youtube/v3/getting-started) if you haven't already.

Finally, [edit your training config](#training-config) to have `dataset.chatlog_csv` set to the newly created `data/chat/YOUR_DISCORD_EXPORT_FILE.csv`, not the `.json`.

and train:
```python
python scripts/train.py -c my_custom_config.yaml # or to use training_config.yaml: python scripts/train.py
```


### Running the Discord Bot
- Update the `.env` with your `BOT_TOKEN`, `GUILDS_ID`, and `CHANNEL_ID` (and/or the `DEV_*` equivalents for a testing server) if you haven't already.
  - The discord.py docs have a [nice overview](https://discordpy.readthedocs.io/en/stable/discord.html) of how to set up a bot account and get your `BOT_TOKEN`.

1. Select your favorite checkpoint from training and add the path to "BEST" in `server/config/models.json`
    ```jsonc
    {
    "name": "model1",
    "desc": "A user-friendly memorable description", // < 100 chars
    "ckpt": "Hermes-2-Theta-Llama-3-8B/chunkh4/2024..._vqdoguk/checkpoint-5500" // your path here
    },
    ```
2. Run `python server/run.py` or `TESTING=1 python server/run.py` to run the testing server.
            

## Roadmap
- [x] Allow training on non-Discord data sources
- [ ] Dedicated per person LoRA training
- [ ] RAG for YouTube links
- [ ] Optional server hosting for trained models/discord bot