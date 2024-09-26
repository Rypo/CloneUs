# This example requires the 'message_content' privileged intent to function.
import io
import time
import json
import yaml
import copy
import logging
import asyncio
import tempfile
import random
import datetime
import itertools
from functools import cached_property
import traceback
import typing
from pathlib import Path
from contextlib import asynccontextmanager, contextmanager

import discord
from discord.enums import TextStyle
from discord.ext import commands,tasks
from discord.utils import MISSING

from PIL import Image

from utils import image as imgutil
from cloneus.utils import common as comutil

def _local_logging_setup(level=logging.DEBUG):
    # TODO: incorporate into server logging setup
    from colorama import Fore, Back, Style
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    cfmt = discord.utils._ColourFormatter()
    fncol = Back.BLACK+Fore.GREEN+Style.DIM #Fore.MAGENTA
    fmtstr = ('[' + '{colour}' + '%(levelname)-5s' + Style.RESET_ALL
              + Fore.BLACK+Style.BRIGHT + ' @ ' + '%(asctime)s' + Style.RESET_ALL + ']'
              + '(' + fncol + '%(funcName)s' + Style.RESET_ALL + ')' + ' %(message)s')
    
    cfmt.FORMATS = {level: logging.Formatter(
        fmtstr.format(colour=colour), ('%H:%M:%S' if level < logging.WARNING else '%Y-%m-%d %H:%M:%S'))
        for level, colour in cfmt.LEVEL_COLOURS}
    handler.setFormatter(cfmt)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

logger = _local_logging_setup(logging.DEBUG)

@contextmanager
def button_disabled(button: discord.ui.Button, emoji_enabled:str, emoji_disabled:str = '⏳', exit_disabled:bool=False):
    button.disabled = True
    button.emoji = emoji_disabled
    try:
        yield
    finally:
        button.emoji = emoji_enabled
        button.disabled = exit_disabled


class ConfirmActionView(discord.ui.View):
    def __init__(self, *, timeout: float = 30):
        super().__init__(timeout=timeout)
        self.value = None
    
    def disable_and_clear(self):
        for item in self.children:
            item.disabled = True
        self.clear_items()
    
    async def on_timeout(self) -> None:
        self.value = True
        self.disable_and_clear()
    

    @discord.ui.button(label='Accept', style=discord.ButtonStyle.green)
    async def accept(self, interaction: discord.Interaction, button: discord.ui.Button):
        #await interaction.response.send_message('Confirming', ephemeral=True)
        self.value = True
        self.disable_and_clear()
        self.stop()

    @discord.ui.button(label='Reject', style=discord.ButtonStyle.grey)
    async def reject(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.value = False
        self.disable_and_clear()
        self.stop()

    @discord.ui.button(label='cancel', style=discord.ButtonStyle.red)
    async def cancel(self, interaction: discord.Interaction, button: discord.ui.Button):
        self.value = '<CANCEL>'
        self.disable_and_clear()
        self.stop()


# NOTE: self.message.edit is **NOT** in place. 
# > Changed in version 2.0: The edit is no longer in-place, instead the newly edited message is returned. 
# > https://discordpy.readthedocs.io/en/stable/api.html?highlight=webhook#discord.WebhookMessage.edit
class ImagePrimaryView(discord.ui.View):
    def __init__(self, *, timeout=None):
        super().__init__(timeout=timeout)
        
        self.call_ctx: commands.Context# = None
        #self.bot: commands.Bot
        self.message: discord.Message #= None
        
        self.images: list[Image.Image] # = []
        self.thumbs: list[Image.Image] # = []
        self._file_cache: list[Image.Image] # = []

        self.img_count:int # = 0

        self.grid_view: DynamicImageGridView # = None
        self.grid_display_active:bool # = False
        
        logger.debug(f'Init: {self}')
        self.t_init = time.monotonic()
        self._time_last_refresh: float
        self._timeout_warning_visible = False
        self.timeout_embed = discord.Embed()

    def image_attrs(self, start:int=0, stop:int|None=None):
        raise NotImplementedError('Requires override')
    
    def thumb_attrs(self, start:int=0, stop:int|None=None):
        raise NotImplementedError('Requires override')
    
    def image_files(self, start:int=0, stop:int|None=None, ext='WebP', **kwargs):
        return [imgutil.to_bfile(**iattrs, ext=ext, **kwargs) for iattrs in self.image_attrs(start, stop)]
    
    async def _populate_cache(self, start:int=0, stop:int|None=None, ext='WebP', **kwargs):
        self._file_cache[start:stop] = self.image_files(start, stop, ext=ext, **kwargs)

    def cached_files(self, start:int=0, stop:int|None=None, ext='WebP', **kwargs):
        files = self._file_cache[start:stop]
        asyncio.create_task(self._populate_cache(start, stop, ext=ext))
        return files
    
    @property
    def time_remaining(self):
        return self.timeout - (time.monotonic()-self._time_last_refresh)

    async def reaction_toggle(self, on:bool=None, emoji:str='⏰'):
        has_reaction = emoji in str(self.message.reactions)
        # emoji in [r.emoji for r in self.message.reactions]:
        # (react:= discord.utils.find(lambda r: r.emoji == emoji, self.message.reactions)):
        if on is not None:
            try:
                if on and not has_reaction:
                    await self.message.add_reaction('⏰')
                    print(f'added react: {str(self.message.reactions)}')
                    has_reaction = True
            
                if not on and has_reaction:
                    await self.message.remove_reaction(emoji, self.call_ctx.bot.user)
                    print(f'removed react: {str(self.message.reactions)}')
                    has_reaction = False

            except Exception as e:
                print(e)
        
        return has_reaction

    @tasks.loop(seconds=90)
    async def timeout_warning(self, warn_begin:float = (5*60)):
        time_remaining = self.time_remaining
        
        #if time_remaining <= warn_at: 
        #    await self.reaction_toggle(on=True)
        
        if time_remaining <= warn_begin:
            if time_remaining > 90:
                if self.timeout_warning.seconds < 90:
                    self.timeout_warning.change_interval(seconds=90)
                timeout_time = (datetime.datetime.now() + datetime.timedelta(seconds=time_remaining)).time().strftime('%I:%M:%S %p')
                #mins,secs = divmod(round(time_remaining), 60)
                timeout_msg = "Timeout at {0} (— {1}m {2:02d}s )".format(timeout_time, *divmod(round(time_remaining), 60))
            else:
                if self.timeout_warning.seconds > 45:
                    self.timeout_warning.change_interval(seconds=45)
                timeout_msg = f"⏰ Timeout in {time_remaining:0.1f}s"
            
            self.timeout_embed = self.timeout_embed.set_footer(text=f'{timeout_msg}')
            
            if self.img_count > 1:
                self.message = await self.message.edit(embed=self.timeout_embed) # , suppress=False
            
            logger.debug(f'Timeout in {time_remaining:0.2f}s : {self} : (prompt={self.call_ctx.kwargs["prompt"]!r})')

        
        if time_remaining <= (-3*60):
            logger.warning(f'LATE TIMEOUT: {self} (prompt={self.call_ctx.kwargs["prompt"]!r})')
            await self.on_timeout()
                

    async def on_timeout(self) -> None:
        logger.info(f'{self} Timeout ({time.monotonic()-self.t_init:0.2f}/{self.timeout}s)')
        
        self.lock_ui()
        self.clear_items()
        try:
            self.message = await self.message.edit(embed=None, view=self)
        except discord.NotFound:
            logger.debug(f'Message(id={self.message.id}) not found, no view to clear')


        self.timeout_warning.cancel()

        if self.grid_view:
            await self.grid_view.on_timeout()
            self.grid_view.stop()
        
        #await self.reaction_toggle(on=False)
            
        del self.images,self.thumbs
        self.stop()
       

    async def reset_timeout(self):
        #print('timeouts reset')
        
        self.timeout = self.timeout
        self._time_last_refresh = time.monotonic()
        #await self.reaction_toggle(on=False)
        if self.message.embeds:
            self.message = await self.message.edit(embed=None)
        
    
    def lock_ui(self):
        for item in self.children:
            item.disabled = True

    def set_view_state(self, grid_view:bool):
        if grid_view:
            if self.grid_view is None:
                self.grid_view = DynamicImageGridView(parent_view=self, orphaned=False, timeout=None)
        
        self.grid_display_active = grid_view
    
    async def send(self, ctx: commands.Context, *args, **kwargs):
        raise NotImplementedError('Requires override')
    
    async def refresh(self, *args, **kwargs):
        raise NotImplementedError('Requires override')

    async def refresh_view(self, grid_display:bool=None):
        if grid_display is not None:
            self.set_view_state(grid_display)
        
        await self.reset_timeout()
        
        if self.grid_display_active:
            self.message = await self.grid_view.refresh()
        else:
            self.message = await self.refresh()
        
        return self.message
    
    

    
class DrawUIView(ImagePrimaryView):#discord.ui.View):
    def __init__(self, *, timeout=None):
        super().__init__(timeout=timeout)

        self._initial_timeout = timeout
        
        self.call_ctx: commands.Context = None
        self.message:discord.Message = None
        
        #self.upsample_thread = None
        self.upsample_message = None

        self.kwarg_sets: list[dict] = []
        self.local_paths: list[Path] = []

        self.images: list[Image.Image] = []
        self.thumbs: list[Image.Image] = []
        self._file_cache: list[Image.Image] = []

        self.img_count = 0
        
        self.cur_filename = None
        self.cur_imgnum = 0
        

        self.is_redraw: bool = None
        self.imgen: commands.Cog = None
        self.call_fn: typing.Callable=None
        self.init_flags = None
        
        self.in_queue=0
        self.queue_message=None

        self.grid_view: DynamicImageGridView = None
        self.grid_display_active:bool = False
        
        #self.img_channel = None
        #self.remote_attachments:list[discord.Attachment] = []
        #self.remote_sender: asyncio.Task = None
        #self.remote_senders: list[asyncio.Task] = []
        
        
    def image_attrs(self, start:int=0, stop:int|None=None):
        return [{'image':img, 'filestem':fpth.stem, 'description':kwg['prompt']} 
                for img,fpth,kwg in itertools.islice(zip(self.images, self.local_paths, self.kwarg_sets), start, stop)]
    
    def thumb_attrs(self, start:int=0, stop:int|None=None):
        return [{'image':img, 'filestem': f'{fpth.stem}_thb', 'description':kwg['prompt']} 
                for img,fpth,kwg in itertools.islice(zip(self.thumbs, self.local_paths, self.kwarg_sets), start, stop)]
    
    async def send(self, ctx: commands.Context, n_init_images:int): #, kwargs:dict=None, image: Image.Image|None=None): #image: Image.Image|None, fpath: str|None):
        self.call_ctx = ctx
        #self.bot: commands.Bot = self.call_ctx.bot
        logger.debug(f'args, kwargs: {ctx.args} {ctx.kwargs}')

        self.imgen = ctx.cog
        self.cmd_name = ctx.command.name
        self.call_fn = self.imgen.redraw if self.cmd_name=='redraw' else self.imgen.draw
        self.init_flags = self.call_ctx.kwargs['flags']
        
        self.message: discord.WebhookMessage = await ctx.send(view=self)
        self._time_last_refresh = time.monotonic()
        self.message: discord.Message = await self.call_ctx.fetch_message(self.message.id) # NOTE: failure to do this will result in Webhook errors after 15minutes
        
        
        self.timeout_warning.start()#warn_begin = (7*60))

        await self.update_queue(n_init_images)
        self.set_view_state(grid_view = n_init_images>1)
        
        return self.message
    
    async def refresh(self):
        self.update_buttons()
        index = self.cur_imgnum-1
        img_file = imgutil.to_bfile(self.images[index], filestem=self.local_paths[index].stem, description=self.kwarg_sets[index]['prompt'], ext='PNG')#ext='WebP', lossless = True)
        self.message = await self.message.edit(attachments=[img_file], view=self)
        return self.message
        
    def update_buttons(self):
        #print('before mod:', self.cur_imgnum)
        if self.cur_imgnum < 1:
            self.cur_imgnum = self.img_count
        elif self.cur_imgnum > self.img_count:
            self.cur_imgnum = 1
        
        all_disabled = self.img_count == 0
        nav_disabled = self.img_count < 2

        self.prev_button.disabled = nav_disabled # (self.cur_imgnum <= 1)
        self.next_button.disabled = nav_disabled # (self.cur_imgnum >= self.n_images)
        self.explode_button.disabled = nav_disabled

        self.redo_button.label = f'({self.cur_imgnum} / {self.img_count})'
        
        self.redo_button.disabled = all_disabled
        self.config_button.disabled = all_disabled
        self.upsample_button.disabled = all_disabled


    async def add_images(self, images: list[Image.Image], call_kwargs:list[dict]):
        n = len(images)
        self.img_count += n
        self.cur_imgnum  = self.img_count
        self.images.extend(images)

        
        fast_files = []
        for i,image in enumerate(images):
            kwgs = call_kwargs[i]
            kwgs.update(fast=self.init_flags.fast)
            self.kwarg_sets.append(kwgs)
            prompt = kwgs['prompt']
            
            fpath = imgutil.prompt_to_fpath(prompt, ext='WebP', bidx=(i if n>1 else None))
            
            self.local_paths.append(fpath)
            self.thumbs.append(imgutil.to_thumb(image, max_size=(256, 256)))
            
            if self.grid_display_active and self.img_count <= 10:
                fast_files.append(imgutil.to_bfile(image, filestem=fpath.stem, description=prompt, ext='WebP'))

            self._file_cache.append(imgutil.to_bfile(image, filestem=fpath.stem, description=prompt, ext='WebP'))

        if fast_files:
            self.message = await self.message.add_files(*fast_files)

        await self.update_queue(-n)

        self.message = await self.refresh_view()
        
        
 


    async def update_queue(self, d:int):
        self.in_queue += d
        msg = f'Queue: {self.in_queue}' if self.in_queue != 0 else ''#'\u200b' # if it goes negative, I wanna know '​'
        self.message = await self.message.edit(content = msg)#, view=self)#embed=embed)

    async def redo(self, interaction:discord.Interaction, kwargs:dict):
        await interaction.response.defer()
        await self.update_queue(1)
        #kwargs = self.kwargs.copy()
        
        kwargs.pop('refine_strength',None)
        # need to reset refine strength in case was changed in modal.
        # otherwise would be refining on every single redo
        
        
        # NOTE: I bet you append these to a list and pop them. Then you'd be able to cancel queued jobs
        
        #await self.call_ctx.invoke(self.call_fn, *self.call_ctx.args, **kwargs, _view=self)
        #task = asyncio.create_task(self.call_fn(self.call_ctx, *self.call_ctx.args, **kwargs, _view=self))
        await asyncio.create_task(self.call_ctx.invoke(self.call_fn, *self.call_ctx.args, **kwargs, _view=self))        


    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.primary, disabled=True, emoji='⬅️', row=1)
    async def prev_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        self.prev_button.disabled = True
        await interaction.response.defer()
        self.cur_imgnum -= 1
        await self.refresh_view()
    
    @discord.ui.button(label='(0 / 0)', style=discord.ButtonStyle.secondary, disabled=True, emoji='🔄', row=1) # ▶️ 🔄 ♻️
    async def redo_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        kwargs = self.kwarg_sets[self.cur_imgnum-1].copy()
        kwargs['seed'] = random.randint(1e9, 1e10-1)
        
        return await self.redo(interaction, kwargs)
    
    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.primary, disabled=True, emoji='➡️', row=1)
    async def next_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        self.next_button.disabled = True
        await interaction.response.defer()
        self.cur_imgnum += 1
        await self.refresh_view()

    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.secondary, disabled=True, emoji='⚙️', row=1) # 🎛️ ⚙️ 🎚️
    async def config_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        #await interaction.response.defer()
        
        kwargs = self.kwarg_sets[self.cur_imgnum-1].copy()

        #kwargs.pop('aspect') # remove since we have dropdown
        kwargs.pop('fast', None) # not worth having redraw scroll, just use whatever is passed on cmd call
        image_url = kwargs.pop('imgurl', kwargs.pop('imgfile', None))

        #image_url = kwargs.pop('imgfile', None)
        if isinstance(image_url, discord.Attachment):
            image_url = image_url.url
        
        # hd_config = {'refine_steps': kwargs.pop('refine_steps'), 'refine_strength':kwargs.pop('refine_strength')}
        if (ref_strength := kwargs.pop('refine_strength', None)) is None:
            ref_strength = 0.30
        
        hd_config = {'refine_strength': ref_strength}
        
        cm = ConfigModal(vals={
            'prompt': kwargs.pop('prompt'),
            'negative_prompt': kwargs.pop('negative_prompt'),
            'image_url': image_url,
            'config': yaml.dump(kwargs, sort_keys=False), # json.dumps(kwargs, indent=1)
            'hd_config': yaml.dump(hd_config, sort_keys=False),
        })
        
        await interaction.response.send_modal(cm) 
        await cm.wait()

        config: dict = yaml.load(cm.vals['config'], yaml.SafeLoader)
        hd_config: dict = yaml.load(cm.vals['hd_config'], yaml.SafeLoader)
        
        # Allow abbr for aspect select
        if aspect:=config.get('aspect'):
            opts = ['auto','square', 'portrait', 'landscape']
            aspect = [o for o in opts if o.startswith(aspect.lower())][0]
            if aspect=='auto':
                aspect = None
            config['aspect'] = aspect
        
        new_kwargs = self.kwarg_sets[self.cur_imgnum-1].copy()
        
        new_kwargs.update({
            'prompt':cm.vals['prompt'],
            'negative_prompt':cm.vals['negative_prompt'],
            **config,
            **hd_config
        })
        if image_url: #if self.is_redraw:
            #new_kwargs.update({'imgfile': cm.vals['image_url']})
            new_kwargs.update({'imgurl': cm.vals['image_url']})
        
        #self.kwarg_sets.append(new_kwargs)
        # NOTE: if decide to do this, remove the defer from on_submit
        await self.redo(cm.submit_interaction, new_kwargs)
    

    @discord.ui.button(label='HD', style=discord.ButtonStyle.green, disabled=True, emoji='⬇️', row=1) # 🆙
    async def upsample_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        # idea: Maybe just make it a toggle? like -- red: HD (off), green: HD (on)
        await interaction.response.defer(thinking=True)
        await self.reset_timeout() # somehow the timeout resets without this, but the timeout_warning task still goes negative
        index = self.cur_imgnum-1
        kwargs = self.kwarg_sets[index].copy()
        
        kwargs['imgurl'] = self.message.attachments[0].url
        #kwargs['imgfile'] = self.message.attachments[0]
        
        if (ref_strength := kwargs.get('refine_strength')) is None:
            ref_strength = 0.30
        
        

        hd_kwargs = {
            'imgurl': self.message.attachments[0].url,
            'prompt': kwargs.get('prompt'),
            'imgfile': self.images[index], # bypass the download
            'refine_strength': ref_strength,
            'steps': kwargs.get('steps'),
            'negative_prompt':kwargs.get('negative_prompt'),
            'guidance_scale':kwargs.get('guidance_scale'),
            'detail_weight':kwargs.get('detail_weight'),
            'seed':kwargs.get('seed'),
        }
        
        image = await asyncio.create_task(self.call_ctx.invoke(self.imgen.hd_upsample, **hd_kwargs)) 
        
        #await interaction.followup.send(file=to_bytes_file(image_file, kwargs['prompt']))
        fpath = imgutil.save_image_prompt(image, kwargs['prompt'], ext='WebP', lossless=True)
        ifile = imgutil.impath_to_file(fpath, description=kwargs['prompt'])
        # ifile = imgutil.to_bfile(image, filestem=fpath.stem, description=kwargs['prompt'], ext='WebP', lossless=True)
        
        # if self.upsample_thread is None:
        #    self.upsample_thread = await self.call_ctx.channel.create_thread(name='Upsampled Images', message=self.message)
        
        # await self.upsample_thread.send(file=ifile)
        #await interaction.followup.send(file=ifile, wait=True)
        
        # TODO: this would break if > 10 upsamples
        if self.upsample_message is None:
            self.upsample_message = await interaction.followup.send(file=ifile, wait=True)
            self.upsample_message = await self.upsample_message.channel.fetch_message(self.upsample_message.id)
            #self.upsample_message = await self.message.reply(files=[ifile])
        else:
            #self.upsample_message = await self.call_ctx.fetch_message(self.upsample_message.id)
            self.upsample_message = await self.upsample_message.add_files(ifile)
            await interaction.delete_original_response()

        
    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.secondary, disabled=True, emoji='↗️', row=0) # row=4 # 💢 ⛶  label='🗗︎', 
    async def explode_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        self.lock_ui()
        with button_disabled(button, emoji_enabled='↗️', emoji_disabled='⏳'):
            await interaction.response.edit_message(view=self)
        self.message = await self.refresh_view(grid_display=True)
 
    
    # @discord.ui.button(label='\u200b', style=discord.ButtonStyle.green, disabled=False, emoji='🛑', row=0) # row=4 # 💢 ⛶  label='🗗︎', 
    # async def halt_button(self, interaction:discord.Interaction, button: discord.ui.Button):
    #     await self.on_timeout()

class GifUIView(ImagePrimaryView):
    def __init__(self,  images:list[Image.Image], *, timeout:float=None):
        super().__init__(timeout=timeout)
        
        self.images = images
        self.thumbs: list[Image.Image] = [imgutil.to_thumb(image, max_size=(256, 256)) for image in self.images]
        self.img_count:int = len(images)
        
        #self.message: discord.Message
        self.gif_filepath: Path
        self.prompt: str
        
        self.grid_view: DynamicImageGridView = None

        self.gif_bytes:int
        self._bytes_limit = discord.utils.DEFAULT_FILE_SIZE_LIMIT_BYTES
        
        self._gif_bfile = None
        self.gif_file_cacher: asyncio.Task = None
        self.optimized_images = None

        self.ext: typing.Literal['GIF','MP4'] = 'GIF'
    


    def image_attrs(self, start:int=0, stop:int|None=None):
        return [{'image':img, 'filestem':self.gif_filepath.stem, 'description':self.prompt} for img in self.images[start:stop]] 
    
    def thumb_attrs(self, start:int=0, stop:int|None=None):
        return [{'image':img, 'filestem': f'{self.gif_filepath.stem}_thb', 'description':self.prompt} for img in self.thumbs[start:stop]] 
    
    async def _set_gif_file(self):
        self._gif_bfile = imgutil.animation_to_bfile(image_frames=self.images, filestem=self.gif_filepath.stem, description=self.prompt, ext=self.ext)
        #return self._gif_bfile
    
    def gif_bfile(self):
        t0=time.monotonic()
        if self._gif_bfile is None:
            self._gif_bfile = imgutil.animation_to_bfile(image_frames=self.images, filestem=self.gif_filepath.stem, description=self.prompt, ext=self.ext)
        gbfile = copy.deepcopy(self._gif_bfile)
        logger.debug(f'deep copy gif: {time.monotonic() - t0:0.2f}s')
        return gbfile
    
        

    async def send(self, ctx: commands.Context, message:discord.WebhookMessage, out_gifpath:Path, prompt: str):
        self.call_ctx = ctx
        self.message = message
        self.message = await self.call_ctx.fetch_message(self.message.id)

        self.gif_filepath = Path(out_gifpath)
        self.prompt = prompt
        #self.filenames = [self.out_gifpath.with_stem(self.out_gifpath.stem + f'_{i}').with_suffix('.webp').name for i in range(len(self.images))]

        self._file_cache = self.image_files(0, None, ext='WebP')
        # If too big:
        # √ - just send a .mp4, by far the easiest and best option
        # x - halve the number of frames
        # x - reduce size (w,h) of gif
        # x - use catbox hosting
        # x - return exploded view with the collapse button disabled. 
        self.gif_bytes = len(self.gif_bfile().fp.read()) 
        logger.info(f'Animation Size - file: {self.gif_bytes/1e6:0.3f} MB | disk: {self.gif_filepath.stat().st_size/1e6:0.3f} MB')
        
        if self.message.guild is not None:
            self._bytes_limit = self.message.guild.filesize_limit  # 25*(1024**2)#2.5e7 # 25MB for non nitro
        
        #over_limit = 
        
        #self.grid_view = DynamicImageGridView(self, orphaned=False, timeout=None)
        
        if self.gif_bytes > self._bytes_limit:
            self.message = await self.message.edit(content="Hwoo boy, that's a lotta bytes you got there.. Guess we doing MP4 now.", view=self) # Must send view=self for on_timeout to engage
            self.ext = 'MP4'

            #self.message = await self.refresh_view(grid_display=True)
            
        #else:
        self.gif_file_cacher = asyncio.create_task(asyncio.gather(asyncio.sleep(0.1), self._set_gif_file()))
        #asyncio.to_thread(imgutil.gif_to_bfile, image_frames=self.images, filestem=self.gif_filepath.stem, description=self.prompt))
        #tsk=asyncio.create_task(self.send_thb())
        f0 = imgutil.to_bfile(self.images[0], filestem=self.gif_filepath.stem, description=self.prompt, ext='WebP')
        self.message = await self.message.edit(content='', attachments=[f0], view=self) # set content to '' to clear "seasoning..."
        #self.message = await self.refresh_view(show_grid=False)
        
        #embed = discord.Embed(type='gifv').set_image(url = f'attachment://{file.filename}')
        # set content to '' to clear "seasoning..."
        #self.message = await self.message.edit(content='', attachments=[discord.File(fp=self.gif_filepath, filename=self.gif_filepath.name, description=prompt)], view=self) #embed=self.embed, 
        self.message = await self.refresh_view(grid_display=False)
        #self.gif_bfile = asyncio.create_task(asyncio.to_thread(imgutil.gif_to_bfile, image_frames=self.images, filestem=self.gif_filepath.stem, description=self.prompt))
            
        self.timeout_warning.start()
        #self.message.edit(content='', attachments=[discord.File(fp=out_gifpath, filename=out_gifpath.name, description=prompt)], view=self)
        return self.message

    async def refresh(self):
        #tsk=asyncio.create_task(self.send_thb())
        # asyncio.to_thread(
        #msg_tsk = asyncio.create_task(self.message.edit(attachments=[imgutil.to_bfile(self.images[0], filestem=self.gif_filepath.stem, description=self.prompt, ext='WebP')], view=self))
        
        img_file = self.gif_bfile()
        #img_file = await self.agif_bfile()
        
        #self.message = await msg_tsk
        #image_frames = self.optimized_images if self.optimized_images is not None else self.images
        #img_file = await asyncio.to_thread(imgutil.gif_to_bfile, image_frames=image_frames, filestem=self.gif_filepath.stem, description=self.prompt)
        #embed = discord.Embed(type='gifv').set_image(url=f'attachment://{img_file.filename}')
        self.message = await self.message.edit(attachments=[img_file], view=self)#, embed=embed,) # set content to '' to clear "seasoning..."
        #self.gif_bfile = asyncio.create_task(asyncio.to_thread(imgutil.gif_to_bfile, image_frames=self.images, filestem=self.gif_filepath.stem, description=self.prompt))
        #self.gif_file_cacher = asyncio.create_task(asyncio.to_thread(imgutil.gif_to_bfile, image_frames=self.images, filestem=self.gif_filepath.stem, description=self.prompt))
        #msg = await self.img_channel.send(files=files)
        #self.message = await imgutil.try_send_gif(msg=self.message, gif_filepath=self.gif_filepath, prompt=self.prompt, view=self)
        #discord.Embed(u)
        #self.reset_timeouts()
        return self.message
       
    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.secondary, disabled=False, emoji='↗️', row=0) # row=4 # 💢 ⛶  label='🗗︎', 
    async def explode_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        was_disabled = button.disabled
        self.lock_ui()
        with button_disabled(button, emoji_enabled='↗️', emoji_disabled='⏳', exit_disabled=was_disabled):
            await interaction.response.edit_message(view=self)
        self.message = await self.refresh_view(grid_display=True)

class DynamicImageGridView(discord.ui.View):
    def __init__(self, parent_view:DrawUIView, *, orphaned:bool=False, max_grid_images:int=10, timeout=None):
        super().__init__(timeout=timeout)
        self.parent_view = parent_view
        self.orphaned = orphaned
        self.max_grid_images = max_grid_images
        
        self.message = self.parent_view.message
        self.cur_batchnum = 1
        self.display_timer = None
        self.page_update_delay = 1.5 # seconds
        self.has_nav: bool = True

        self._init_buttons()
        logger.debug(f'Init: {self}')
        self.t_init = time.monotonic()

        #self._image_attrs:list[dict] = []
        #self._thumb_attrs:list[dict] = []
        self._visible_index:int = None 
        self._last_img_count:int = None 
    
    def _init_buttons(self):
        if self.orphaned:
            self.collapse_button.disabled = True
        
        if self.n_batches < 2:
            self.prev_button.disabled = True # (self.cur_imgnum <= 1)
            self.next_button.disabled = True # (self.cur_imgnum >= self.n_images)
            self.remove_item(self.counter_button).remove_item(self.prev_button).remove_item(self.next_button)
            self.has_nav = False
        self.update_buttons()

    @property
    def n_batches(self):
        n_batches,_rem = divmod(self.parent_view.img_count, self.max_grid_images)
        #print(f'init| image_count: {self.parent_view.img_count}, ~n_batches: {n_batches}, rem: {_rem}')
        n_batches += min(_rem, 1) # 0 or 1
        #print('n_batches:', n_batches)
        self._last_img_count = self.parent_view.img_count
        return n_batches
    

    def interrupt_refresh(self):
        if self.display_timer:
           self.display_timer.cancel()
    
    async def halt_refresh(self):
        if self.display_timer:
           self.display_timer.cancel()
           await self.display_timer

    def lock_ui(self):
        for item in self.children:
            item.disabled = True
            
    
    async def on_timeout(self) -> None:
        logger.info(f'{self} Timeout ({time.monotonic()-self.t_init:0.2f}/{self.timeout}s)')
        # https://discordpy.readthedocs.io/en/stable/faq.html#how-can-i-disable-all-items-on-timeout
        self.interrupt_refresh()
        self.lock_ui()
        self.clear_items()
        try:
            self.message = await self.message.edit(embed=None, view=self)
        except discord.NotFound:
            logger.debug(f'Message(id={self.message.id}) not found, no view to clear')
        #self.message = await self.message.edit(view=self)
        #self.stop()
        #del self.image_attrs_batches
            
    def update_buttons(self):
        n_batches = self.n_batches
        if self.cur_batchnum < 1:
            self.cur_batchnum = n_batches
        elif self.cur_batchnum > n_batches:
            self.cur_batchnum = 1
        
        #print(self.has_nav, self.children)
        self.prev_button.disabled = (n_batches < 2)
        self.next_button.disabled = (n_batches < 2)
        self.collapse_button.disabled = self.orphaned

        if not self.has_nav and n_batches > 1:
            self.prev_button.disabled = False
            self.next_button.disabled = False
            self.add_item(self.counter_button).add_item(self.prev_button).add_item(self.next_button)

            self.has_nav = True
        
        self.counter_button.label = f'({self.cur_batchnum} / {n_batches})'
    
    def attr_batch(self, thumbs:bool, index:int=None):
        if index is None:
            index = self.cur_batchnum-1
        
        offs = index*self.max_grid_images
        stop = offs+self.max_grid_images
        
        if thumbs:
            return self.parent_view.thumb_attrs(offs, stop=stop)

        return self.parent_view.image_attrs(offs, stop=stop)
        
       
    async def async_file_batch(self, img_attr_batch:list[dict], ext:str, **kwargs):
        for img_attrs in img_attr_batch:
            yield imgutil.to_bfile(**img_attrs, ext=ext, **kwargs)
    
    async def async_filecache_batch(self, index:int,):
        offs = index*self.max_grid_images
        stop = offs+self.max_grid_images
        for file in self.parent_view.cached_files(offs, stop):
            yield file
    
    async def async_batch(self, thumbs:bool, index:int, ext:str, **kwargs):
        offs = index*self.max_grid_images
        stop = offs+self.max_grid_images
        if thumbs:
            for img_attrs in self.parent_view.thumb_attrs(offs, stop=stop):
                yield imgutil.to_bfile(**img_attrs, ext=ext, **kwargs)
        else: 
            for file in self.parent_view.cached_files(offs, stop, ext=ext):
                yield file

    
    async def render_sync(self, thumbs:bool, index:int = None):
        if index is None:
            index = self.cur_batchnum-1

        props = dict(ext='JPEG', optimize=False, quality=30) if thumbs else dict(ext='WebP')

        self.message = await self.message.edit(
            attachments=[imgutil.to_discord_file(**img_attrs, **props) for img_attrs in self.attr_batch(thumbs=thumbs, index=index)], view=self)
        
        self._visible_index = index
        
    async def render_delayed(self, thumbs:bool, delay:float, index:int = None):
        if index is None:
            index = self.cur_batchnum-1
        
        props = dict(ext='JPEG', optimize=False, quality=30) if thumbs else dict(ext='WebP', lossless=True, plugin="opencv",) #  lossless=True, quality=0
        
        
        #img_files = [file async for file in self.async_file_batch(self.attr_batch(thumbs=thumbs, index=index), **props)]
        await asyncio.sleep(delay/2)
        #t0=time.monotonic()
        
        img_files = [file async for file in self.async_batch(thumbs=thumbs, index=index, **props)]

        #print(f'time to_files (thumbs={thumbs}): {time.monotonic()-t0:0.2f}s')
        await asyncio.sleep(delay/2)
        
        self.message = await self.message.edit(attachments=img_files, view=self)
        self._visible_index = index

    async def static_refresh(self):
        if len(self.parent_view.message.attachments) != self.parent_view.img_count:
            return await self.parent_view.message.edit(attachments=self.parent_view.image_files(0, stop=self.max_grid_images), view=self)
        return await self.parent_view.message.edit(view=self)

    async def refresh(self, interaction:discord.Interaction=None):
        has_new_images = self._last_img_count != self.parent_view.img_count # this will update on self.n_batches, so call ahead
        
        if self.n_batches < 2:
            return await self.static_refresh()
        #self.interrupt_refresh()
        self.update_buttons()
        index = self.cur_batchnum-1 # always after update_buttons
        
        #if interaction:
        #    self.disable_all()
        #    await interaction.response.defer()
        #     self.display_timer = asyncio.create_task(self.render_delayed(thumbs=True, delay=0., index=index))
        #    await interaction.response.edit_message(view=self)
        
        await self.parent_view.reset_timeout()
        thumb_timer = None
        # prevent needlessly refreshing images whenever parent_view calls refresh
        if index != self._visible_index or (has_new_images and len(self.message.attachments)<self.max_grid_images):
            thumb_timer = asyncio.create_task(self.render_delayed(thumbs=True, delay=0., index=index))
            self.display_timer = asyncio.create_task(self.render_delayed(thumbs=False, delay=self.page_update_delay, index=index))
            # self.display_timer = asyncio.gather(
            #     asyncio.create_task(self.render_delayed(thumbs=True, delay=0., index=index)),
            #     asyncio.sleep(0.1),
            #     asyncio.create_task(self.render_delayed(thumbs=False, delay=self.page_update_delay, index=index))
            #     )
                        
        else:
            self.message = await self.message.edit(view=self) 
        #    print('no change')
        #if thumb_timer:
        #    await thumb_timer
        #self.update_buttons()
        #self.message = await self.message.edit(view=self) 
        
        
        return self.message
        

        
    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.secondary, disabled=False, emoji='↙️', row=1) # row=4 # 💢 \u200b ↩️ ↵ label='↩', 
    async def collapse_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        was_disabled = button.disabled
        self.interrupt_refresh()
        self.lock_ui() # Need this or clicking an arrow will make the view comeback

        with button_disabled(button, emoji_enabled='↙️', emoji_disabled='⏳', exit_disabled=was_disabled):
           await interaction.response.edit_message(view=self)
           self.message = await self.parent_view.refresh_view(grid_display=False)
           self._visible_index = None
        
        
        

    
    @discord.ui.button(label='(0 / 0)', style=discord.ButtonStyle.grey, disabled=True,  row=1) # ▶️ 🔄 ♻️ emoji='🔄',
    async def counter_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()

    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.primary, disabled=False, emoji='⬅️', row=1)
    async def prev_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        #await interaction.response.defer()
        self.interrupt_refresh()
        self.lock_ui()
        await interaction.response.edit_message(view=self)

        self.cur_batchnum -= 1
        await self.refresh(interaction)
        
    
    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.primary, disabled=False, emoji='➡️', row=1)
    async def next_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        #await interaction.response.defer()
        self.interrupt_refresh()
        self.lock_ui()
        await interaction.response.edit_message(view=self)
        self.cur_batchnum += 1
        await self.refresh(interaction)


        
class ConfigModal(discord.ui.Modal, title='Tweaker Menu'):
    prompt = discord.ui.TextInput(label='Prompt', style=TextStyle.paragraph, required=True, min_length=1, max_length=1000)
    negative_prompt = discord.ui.TextInput(label='Negative Prompt', style=TextStyle.short, required=False, max_length=1000)
    config = discord.ui.TextInput(label='Config', style=TextStyle.paragraph, required=False, min_length=1, max_length=200,) # style=TextStyle.short,
    hd_config = discord.ui.TextInput(label='⬇️ HD Config (Upsampling)', style=TextStyle.paragraph, required=False, max_length=100) # style=TextStyle.short,
    #image_url =  # style=TextStyle.short,

    def __init__(self, vals=None, *args, **kwargs):
        self.vals = vals
        super().__init__(*args, **kwargs)
        self.image_url = discord.ui.TextInput(label='Image URL', style=TextStyle.short, required=False)
        if vals.get('image_url') is not None: 
            self.add_item(self.image_url)
        
        self._set_default_values()
        self.submit_interaction = None
        
        #self.children[0]

    def _set_default_values(self, vals=None):
        if vals is not None:
            self.vals = vals

        self.prompt.default = self.vals.get('prompt')
        self.negative_prompt.default = self.vals.get('negative_prompt')
        self.config.default = self.vals.get('config')
        self.hd_config.default = self.vals.get('hd_config')
        self.image_url.default = self.vals.get('image_url')
        
        
    async def on_submit(self, interaction: discord.Interaction):
        self.vals['prompt'] = self.prompt.value
        self.vals['negative_prompt'] = self.negative_prompt.value
        self.vals['config'] = self.config.value
        self.vals['hd_config'] = self.hd_config.value
        self.vals['image_url'] = self.image_url.value
        
        self.submit_interaction = interaction
        #await interaction.response.defer()
        self.stop()
        
    
    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        await interaction.response.send_message('Oops! Done goofed.', ephemeral=True)
        traceback.print_exception(type(error), error, error.__traceback__)