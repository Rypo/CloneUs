import random
import discord
from discord import app_commands
from discord.ext import commands

from omegaconf import OmegaConf

from cloneus.data import roles


import config.settings as settings
from cmds import (
    flags as cmd_flags, 
    transformers as cmd_tfms, 
    choices as cmd_choices
)

from managers.txtman import CloneusManager
from managers.msgman import MessageManager



class ConfigManager:
    def __init__(self, cfg):
        self.cfg = cfg

STATE_CONFIG = OmegaConf.create(dict(
    streaming_mode = False,
    tts_mode = False,
    autonomous = False,
    auto_reply_method = '',

))


class SetConfig:#(commands.HybridGroup): # name='set', description='Change something about the bot'):
    clomgr: CloneusManager
    msgmgr: MessageManager
    streaming_mode: bool = False
    tts_mode: bool = False
    auto_reply_mode: str = ''
    auto_reply_candidates: list = None
    auto_reply_enabled_users: set[discord.User] = set()
    _display_initials: str = ','.join(roles.initial_to_author.keys())
    # discord.utils.MISSING 
    #self.cfgmgr = ConfigManager(STATE_CONFIG)
    default_system_msg: str = None
    model_randomize_proba: float = None

    def check_state(self):
        # TODO: check streaming mode/num beams
        pass

    def argsettings(self):
        return [
            ('Streaming', self.streaming_mode,''),
            ('TTS mode', self.tts_mode,''),
            ('Auto reply mode', self.auto_reply_mode if self.auto_reply_mode else False, (f'({self._display_initials})') if self.auto_reply_mode else ''),
            #('Fully autonomous', self.autonomous if self.autonomous else False,'')
         ]
    
    @commands.command(name='fullauto', hidden=True)
    @commands.is_owner()
    async def fullauto(self, ctx: commands.Context, autonomous: bool = None):
        self.msgmgr.autonomous = autonomous if autonomous is not None else (not self.msgmgr.autonomous)
        await self.autoreply(ctx, auto_mode = 'rbest')
        await ctx.send(f'full auto: {self.msgmgr.autonomous}', ephemeral=True)

    @commands.hybrid_group(name='set')#, fallback='arg')#, description='Quick set a value', aliases=[])
    async def setarg(self, ctx: commands.Context):
        '''(GROUP). call `!help set` to see sub-commands.'''
        if ctx.invoked_subcommand is None:
            await ctx.send(f"{ctx.subcommand_passed} does not belong to genmode")
    
    @setarg.command(name='streaming')
    async def set_streaming(self, ctx: commands.Context, enabled:bool = True):
        '''Turn streaming text output on/off.

        If enabled, bot will update messages in place via edits
          each time it predicts a new word..

        Args:
            enabled: Enable/disable streaming mode
        '''
        msgs = []
        if enabled and self.clomgr.gen_config.num_beams > 1:
            self.clomgr.gen_config.num_beams = 1
            msgs.append('Streaming incompatible with num_beams>1. Setting num_beams=1')

        self.streaming_mode = enabled
        msgs.append(f'Streaming mode: {"en" if self.streaming_mode else "dis"}abled.')
        
        await ctx.send('\n'.join(msgs))
    
    @setarg.command(name='speaking')
    async def set_speaking(self, ctx: commands.Context, enabled:bool = True):
        '''Turn text-to-speech on/off.

        If enabled, bot will speak messages. 
        Notes: 
            - Only works for commands called with bang prefix `!cmd`.
            - Not compatible with streaming mode.
            - limited to ??? characters. 

        Args:
            enabled: Enable/disable tts mode
        '''
        self.tts_mode = enabled
        await ctx.send(f'TTS mode: {"en" if self.tts_mode else "dis"}abled.')

    @setarg.command(name='youtube')
    async def set_youtube(self, ctx: commands.Context, enabled:bool = None):
        '''Turn youtube link parsing on/off.

        Args:
            enabled: Toggle enable/disable youtube link parsing
        '''
        if enabled is None:
            enabled = not self.clomgr.ytm.enabled
        self.clomgr.ytm.enabled = enabled
        await ctx.send(f'YouTube link parsing: {"en" if enabled else "dis"}abled.')

    @setarg.command(name='ctxlimit')
    async def ctxlimit(self, ctx: commands.Context, limit: int):
        """Set the maximum number of messages the bot can store in context. (Default: 31)
        
        Setting limit > Default will increase the *capacity* but will NOT add
            any messages to the context. 
        Setting limit < Default will reduce the capacity and WILL remove the
            earliest messages from the context. 

        Args:
            limit: maximum number of context messages to use for predictions.
        """
        prv_limit = self.msgmgr.message_cache_limit
        self.msgmgr.message_cache_limit = limit
        await ctx.send(f"Updated limit: ({prv_limit} -> {self.msgmgr.message_cache_limit})")


    
    @setarg.command(name='autoreply')
    @app_commands.choices(auto_mode=cmd_choices.AUTO_MODES)
    async def autoreply(self, ctx: commands.Context, auto_mode: str = 'rbest', 
                        author_initials: app_commands.Transform[str, cmd_tfms.AuthorInitialsTransformer] = None,):
        """When set, bot will automatically respond after *every* (non-command) user message.
        
        Args:
            auto_mode: Method for automatically choosing the author (default: rbest)
                rbest  = Random weighted selection (p)
                irbest = Random inverse weighted selection (1-p)
                urand  = Uniform random selection (p= 1/5)
                top    = Most probable pick (p=1)

            author_initials: Unordered sequence of author initials (no spaces). Restricts selection to those authors.

        """
        self.auto_reply_mode = auto_mode.lower()

        self.auto_reply_candidates=None
        self._display_initials=','.join(roles.initial_to_author.keys())

        self.auto_reply_enabled_users.add(ctx.author)
        
        if author_initials:
            if len(author_initials)==1:
                self.auto_reply_mode = author_initials[0]+'bot'
                return await ctx.send(f"Auto reply mode enabled. Auto {self.auto_reply_mode} rollin' out")
                        
            self.auto_reply_candidates = [roles.initial_to_author[i] for i in author_initials]
            self._display_initials = ','.join(author_initials)

        
        if self.auto_reply_mode == 'off':
            self.auto_reply_enabled_users.remove(ctx.author)
            self.auto_reply_mode = ''
            self.msgmgr.autonomous = False
            return await ctx.send("Auto reply mode disabled")
            
        
        modedesc={'rbest':'weighted random selection', 
                  'irbest':'inverted-weight random selection', 
                  'urand':'uniform random selection', 
                  'top':'top probability author'}

        msg=f'Auto reply mode enabled. Using {modedesc[self.auto_reply_mode]}: {self.auto_reply_mode} ∈ ({self._display_initials})'
        await ctx.send(msg)


    
    @setarg.command(name='model')
    @app_commands.choices(version=cmd_choices.MODEL_GENERATIONS)
    async def set_model(self, ctx: commands.Context, version: app_commands.Choice[str]):
        """Pick your favorite past model by its wildest lines
        
        Args:
            version: The model that said that line
        """

        genmap = {m['name']: settings.RUNS_DIR/m['ckpt'].split('runs/full/')[-1] for m in settings.BEST_MODELS} 
        await ctx.defer()
        await self.clomgr.load(genmap[version.value], gen_config='best_generation_config.json')
        await ctx.send(f'switched to {version.value.title()} model. May the odds be ever in your favor.')
    
    @setarg.command(name='randomode')
    async def set_randomode(self, ctx: commands.Context, change_rate: int = 5):
        """Enabled random swapping between trained models mid conversation.
        
        Args:
            change_rate: Average number of messages between model swaps. Set=0 to disable.
        """
        if change_rate == 0:
            self.model_randomize_proba = None
            self.clomgr.model_randomize_proba = self.model_randomize_proba
            return await ctx.send('Took meds. Sticking to 1 personality.')
        self.model_randomize_proba = 1/change_rate
        self.clomgr.model_randomize_proba = self.model_randomize_proba

        return await ctx.send(f'Brain swap on average every {change_rate} messages.')

    @setarg.command(name='era')
    @app_commands.choices(period=cmd_choices.MODEL_YEARS)
    async def set_era(self, ctx: commands.Context, period: app_commands.Choice[str]):
        """Wind back the clock and talk to our past selves.
        
        Args:
            period: The years the model has been trained on.
        """
        
        yearmap = {m['years']: settings.RUNS_DIR/m['ckpt'].split('runs/full/')[-1] for m in settings.YEAR_MODELS}
        if period.value == 'random':
            rperiod=random.choice(list(yearmap))
            model_path = yearmap[rperiod]
            period_msg = f'||{rperiod}||'
        else: 
            model_path = yearmap[period.value] #settings.RUNS_DIR/period.value['ckpt'].split('runs/full/')[-1]
            period_msg = period.name
        await ctx.defer()
        await self.clomgr.load(model_path, gen_config=None)
        _ = self.clomgr.update_genconfig({"top_k": 80, "top_p": 0.9, "repetition_penalty": 1.1, "temperature": 1.2})
        await ctx.send(f"Traveled back to {period_msg}. _Paradoxes will void warranty_")



    @commands.hybrid_command(name='gc')
    async def genconfig(self, ctx: commands.Context, *, flags: cmd_flags.GenerationFlags):
        """Set Generation Configuration values."""
        msgs = []
        if flags.num_beams is not None and flags.num_beams>1 and self.streaming_mode:
            self.streaming_mode = False
            msgs.append('Streaming incompatible with num_beams>1. Streaming disabled.')

        flag_config = dict(flags)#{name: fval for name, fval in flags}
         # filter out Nones since define special behavor in clo.set_genconf for Nones
        config_updates = {name: val for name, val in flag_config.items() if val is not None}
        update_message = self.clomgr.update_genconfig(config_updates)
        msgs.append(update_message)

        await ctx.send('\n'.join(msgs))
    
    @setarg.command(name='wordrules')
    async def wordrules(self, ctx: commands.Context, *, flags: cmd_flags.WordRuleFlags):
        """Set word rules.
        
        Note: incompatible with streaming mode.
        Note: works better with `num_beams` > 1

        """
        banwords = flags.banned_words
        weightwords = flags.weighted_words

        msgs = []

        if banwords == ['']:
            banwords = 'CLEAR'
            msgs += ['Banned word list cleared.']
        elif banwords is not None:
            msgs += [f'Banned words: `{banwords}`']
                
        
        if weightwords == ['']:
            weightwords = 'CLEAR'
            msgs += ['Weighted word list cleared.']
        elif weightwords is not None:
            msgs += [f'Weighted words: `{banwords}`']
            
        update_msg = self.clomgr.update_wordlists(banwords, weightwords)
        print(update_msg)
        #msgs.append(update_msg)
        
        await ctx.send('\n'.join(msgs))

        
        #await ctx.send(f'Banned words: `{flags.banned_words}`')
        #await ctx.send(f"Weighted words: `{flags.weighted_words}`")
        #await ctx.send(outstr)

    @setarg.command(name='sysmsg')
    async def sysmsg(self, ctx: commands.Context, system_msg: str=None):
        """Set the default system message for calls to /ask and /chat.
        
        Args:
            system_msg: The default system_msg used for ALL `/ask` `/chat` calls unless locally overriden.
        """

        await ctx.send(f"Default System Message: ({self.default_system_msg} ⇒ {system_msg!r})")
        self.default_system_msg = system_msg
