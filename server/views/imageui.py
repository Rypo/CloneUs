# This example requires the 'message_content' privileged intent to function.
import io
import time
import json
import yaml
import asyncio
import tempfile
import random
import datetime
import traceback
import typing
from pathlib import Path

import discord
from discord.enums import TextStyle
from discord.ext import commands
from discord.utils import MISSING

from PIL import Image

from utils import image as imgutil
from cloneus.utils import common as comutil

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

class DrawUIView(discord.ui.View):
    def __init__(self, *, timeout=None):
        super().__init__(timeout=timeout)
        
        self.call_ctx = None
        self.image_url=None
        self.message = None
        self.upsample_message = None
        #self.upsample_thread = None

        self.kwarg_sets = []
        self.local_paths = []
        self.images = []

        self.cur_filename = None
        self.cur_imgnum = 0
        self.img_count = 0

        self.is_redraw: bool = None
        self.imgen: commands.Cog = None
        self.call_fn: typing.Callable=None
        self.init_flags = None
        
        self.in_queue=0
        self.queue_message=None
    
    @property
    def kwargs(self):
        return self.kwarg_sets[self.cur_imgnum-1]

    async def on_timeout(self) -> None:
        print(f'{self.__class__.__name__} Timeout:', datetime.datetime.now())
        for item in self.children:
            item.disabled = True
        del self.images, self.local_paths
        self.clear_items()
        self.message = await self.message.edit(view=self)
        if self.queue_message:
            self.queue_message = await self.queue_message.delete()


    def add_kwargs(self, call_kwargs:list[dict]):

        if self.is_redraw is None: # first run
            kwargs = call_kwargs[-1]
            self.is_redraw = 'imgurl' in kwargs or 'imgfile' in kwargs
            self.imgen = self.call_ctx.bot.get_cog(self.call_ctx.cog.qualified_name)
            self.call_fn = self.imgen.redraw if self.is_redraw else self.imgen.draw
            self.init_flags = self.call_ctx.kwargs['flags']
        
        #seed = kwargs.pop('seed', None)#random.randint(1e9, 1e10-1))
        for kset in call_kwargs:
            kset.update(fast=self.init_flags.fast)
        
        self.kwarg_sets.extend(call_kwargs)

    async def send(self, ctx: commands.Context, n_init_images:int, start_as_grid:bool = False): #, kwargs:dict=None, image: Image.Image|None=None): #image: Image.Image|None, fpath: str|None):
        self.call_ctx = ctx
        self.message = await ctx.send(view=self)
        await self.update_queue(n_init_images)
        # if self.mcounter is None:
        #     self.in_queue = n_init_images
        #     #self.mcounter = await interaction.followup.send(content = f'Queue: {self.in_queue}', wait=True, silent=True)
        #     self.mcounter = await self.message.reply(content = f'Queue: {self.in_queue}', silent=True)
        return self.message

    def update_buttons(self):
        #print('before mod:', self.cur_imgnum)
        if self.cur_imgnum < 1:
            self.cur_imgnum = self.img_count
        elif self.cur_imgnum > self.img_count:
            self.cur_imgnum = 1
        
        self.prev_button.disabled = self.img_count < 2 # (self.cur_imgnum <= 1)
        self.next_button.disabled = self.img_count < 2 # (self.cur_imgnum >= self.n_images)
        self.explode_button.disabled = self.img_count < 2

        self.redo_button.label = f'({self.cur_imgnum} / {self.img_count})'
        
        if self.img_count != 0:
            self.redo_button.disabled = False
            self.config_button.disabled = False
            self.upsample_button.disabled = False

    async def update_view(self, img_file: discord.File = None):
        self.update_buttons()
        #print(f'index: {self.cur_imgnum}, items: {self.n_images}')
        if img_file is None:
            self.message = await self.message.edit(view=self)
        else:
            self.message = await self.message.edit(attachments=[img_file], view=self)
        
        #return self.message.attachments[0]

    # async def add_image(self, image: Image.Image, call_kwargs:dict):
    #     self.img_count += 1
    #     self.cur_imgnum  = self.img_count
    #     self.add_kwargs([call_kwargs])
    #     #self.kwarg_sets += self.proc_kwargs(kwargs)

    #     prompt = call_kwargs['prompt']
        
    #     fpath = imgutil.save_image_prompt(image, prompt=prompt, ext='PNG', bidx=None)
    #     self.images.append(image)
    #     self.local_paths.append(fpath)
        
    #     await self.refresh()
    #     #bfile = imgutil.impath_to_file(fpath, description=prompt)
    #     #bfile = imgutil.to_bytes_file(image, self.kwargs['prompt'])
    #     #cur_attach = await self.update_view(bfile)
    #     #self.files.append(bfile)
    #     await self.update_queue(-1)
        
    
    async def add_images(self, images: list[Image.Image], call_kwargs:list[dict]):
        n = len(images)
        self.img_count += n
        self.cur_imgnum  = self.img_count
        self.add_kwargs(call_kwargs)
        self.images.extend(images)
        
        fpaths = [imgutil.save_image_prompt(images[i], call_kwargs[i]['prompt'], ext='PNG', bidx=i) for i in range(n)]
        self.local_paths.extend(fpaths)
        #fp_tasks = asyncio.gather(*[asyncio.to_thread(imgutil.save_image_prompt, images[i], prompt, ext='PNG', bidx=i) for i in range(n)])


        #fpaths = await fp_tasks#asyncio.gather(*fp_tasks)
        #for fpath in fpaths:
        #    self.local_paths.append(fpath)
            #bfile = imgutil.impath_to_file(fpath, prompt)
            
        await self.refresh()
        await self.update_queue(-n)
 

    async def refresh(self):
        self.update_buttons()
        index = self.cur_imgnum-1
        #print(f'Index: {index} | local_paths[{index}]: {self.local_paths[index].name} | kwarg_sets[{index}]: {self.kwarg_sets[index]}') #[p.name for p in self.local_paths]
        #print(f'prompts:', set([k['prompt'] for k in self.kwarg_sets]))

        cur_file = imgutil.to_bfile(self.images[index], filestem=self.local_paths[index].stem, description=self.kwarg_sets[index]['prompt'])
        await self.update_view(cur_file)
        #cur_attach = await self.update_view(cur_file)

        return self.message

    async def update_queue(self, d:int, interaction:discord.Interaction=None):
        # if self.mcounter is None:
        #     self.in_queue = 0
        #     #self.mcounter = await interaction.followup.send(content = f'Queue: {self.in_queue}', wait=True, silent=True)
        #     self.mcounter = await self.message.reply(content = f'Queue: {self.in_queue}', silent=True)
        
        self.in_queue += d

        msg = f'Queue: {self.in_queue}' if self.in_queue != 0 else '\u200b' # if it goes negative, I wanna know '​'

        #self.mcounter = await self.mcounter.edit(content = msg)
        self.message = await self.message.edit(content = msg)

    async def redo(self, interaction:discord.Interaction, kwargs:dict):
        await interaction.response.defer()
        await self.update_queue(1, interaction)
        #kwargs = self.kwargs.copy()
        
        kwargs.pop('refine_strength',None)
        # need to reset refine strength in case was changed in modal.
        # otherwise would be refining on every single redo
        
        
        # NOTE: I bet you append these to a list and pop them. Then you'd be able to cancel queued jobs
        
        #await self.call_ctx.invoke(self.call_fn, *self.call_ctx.args, **kwargs, _view=self)
        #task = asyncio.create_task(self.call_fn(self.call_ctx, *self.call_ctx.args, **kwargs, _view=self))
        await asyncio.create_task(self.call_ctx.invoke(self.call_fn, *self.call_ctx.args, **kwargs, _view=self))        


    # @discord.ui.select(cls=discord.ui.Select, placeholder='image layout/aspect ratio', row=0)
    # async def aspect_select(self, interaction: discord.Interaction, select: discord.ui.Select):
    #     selected_value = select.values[0]

    #     self.kwargs['aspect'] = None if selected_value == 'auto' else selected_value
    #     for opt in select.options:
    #         opt.default = (opt.value == selected_value)
    #     return await self.redo(interaction)

    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.primary, disabled=True, emoji='⬅️', row=1)
    async def prev_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        self.prev_button.disabled = True
        await interaction.response.defer()
        self.cur_imgnum -= 1
        await self.refresh()
    
    @discord.ui.button(label='(0 / 0)', style=discord.ButtonStyle.secondary, disabled=True, emoji='🔄', row=1) # ▶️ 🔄 ♻️
    async def redo_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        kwargs = self.kwargs.copy()
        kwargs['seed'] = random.randint(1e9, 1e10-1)
        
        return await self.redo(interaction, kwargs)
    
    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.primary, disabled=True, emoji='➡️', row=1)
    async def next_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        self.next_button.disabled = True
        await interaction.response.defer()
        self.cur_imgnum += 1
        await self.refresh()

    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.secondary, disabled=True, emoji='⚙️', row=1) # 🎛️ ⚙️ 🎚️
    async def config_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        #await interaction.response.defer()
        
        kwargs = self.kwargs.copy()

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
        
        new_kwargs = self.kwargs.copy()
        
        new_kwargs.update({
            'prompt':cm.vals['prompt'],
            'negative_prompt':cm.vals['negative_prompt'],
            **config,
            **hd_config
        })
        if self.is_redraw:
            #new_kwargs.update({'imgfile': cm.vals['image_url']})
            new_kwargs.update({'imgurl': cm.vals['image_url']})
        
        #self.kwarg_sets.append(new_kwargs)
        # NOTE: if decide to do this, remove the defer from on_submit
        await self.redo(cm.submit_interaction, new_kwargs)
    

    @discord.ui.button(label='HD', style=discord.ButtonStyle.green, disabled=True, emoji='⬇️', row=1) # 🆙
    async def upsample_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        # TODO: Maybe just make it a toggle? like -- red: HD (off), green: HD (on)
        await interaction.response.defer(thinking=True)
        # print(interaction.message.attachments)
        kwargs = self.kwargs.copy()
        #imgen = self.call_ctx.bot.get_cog(self.call_ctx.cog.qualified_name)
        kwargs['imgurl'] = self.message.attachments[0].url
        #kwargs['imgfile'] = self.message.attachments[0]
        
        if (ref_strength := kwargs.get('refine_strength')) is None:
            ref_strength = 0.30
        
        index = self.cur_imgnum-1

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
        #kwargs['strength'] = 0 # set to 0 to avoid strong redraw.
        #kwargs['refine_steps'] = max(kwargs['refine_steps'], 1)
        #kwargs['denoise_blend'] = kwargs.get('denoise_blend', None),
        
        # Pop aspect so that dimensions are preserved
        #kwargs.pop('aspect', None)
        print(self.call_ctx.args)
        #image, fpath = await self.call_ctx.invoke(imgen.hd_upsample, *self.call_ctx.args, **kwargs)
        image, fpath = await self.call_ctx.invoke(self.imgen.hd_upsample, *self.call_ctx.args, **hd_kwargs)
        
        #await interaction.followup.send(file=to_bytes_file(image_file, kwargs['prompt']))
        ifile = imgutil.to_bytes_file(image, kwargs['prompt'], ext='PNG')
        
        # if self.upsample_thread is None:
        #    self.upsample_thread = await self.call_ctx.channel.create_thread(name='Upsampled Images', message=self.message)
        
        # await self.upsample_thread.send(file=ifile)

        # TODO: this would break if > 10 upsamples
        if self.upsample_message is None:
            self.upsample_message = await self.message.reply(file=ifile)
        else:
            self.upsample_message = await self.upsample_message.add_files(ifile)

        await interaction.delete_original_response()
    
    # @discord.ui.button(label='\u200b', style=discord.ButtonStyle.secondary, disabled=False, emoji='❌', row=0) # row=4
    # async def delete_button(self, interaction:discord.Interaction, button: discord.ui.Button):
    #     await interaction.response.defer()
    #     self.stop()
    #     self.clear_items()
    #     await self.message.edit(view=self)
    
    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.secondary, disabled=True, emoji='↗️', row=0) # row=4 # 💢 ⛶  label='🗗︎', 
    async def explode_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        button.disabled = True
        button.emoji = '⏳'
        await interaction.response.edit_message(view=self)
        # button.disabled = False # already handled in button_update
        button.emoji = '↗️'        
        
        if self.img_count <= 10:
            files = [imgutil.to_bfile(img, filestem=fpath.stem, description=kwargs['prompt']) for img,fpath,kwargs in zip(self.images, self.local_paths, self.kwarg_sets)]
            #self.message = await self.message.edit(content='',attachments=files, view=ImageGridView(self))
            self.message = await self.message.edit(attachments=files, view=ImageGridView(self))
        else:
            filestems, descriptions = [], []
            for fpath, kwargs in zip(self.local_paths, self.kwarg_sets):
                filestems.append(fpath.stem)
                descriptions.append(kwargs['prompt'])

            #list(zip(*[(fpath.stem, kwargs['prompt']) for fpath,kwargs in zip(self.local_paths, self.kwarg_sets)]))

            view = PagedImageGridView(self, images=self.images, filestems=filestems, descriptions=descriptions)
            self.message = await view.send(message=self.message)


class GifUIView(discord.ui.View):
    def __init__(self,  images:list[Image.Image], *, timeout:float=None):
        super().__init__(timeout=timeout)
        self.images = images
        self.message: discord.Message
        self.gif_filepath: Path
        self.prompt: str

        self.optimized_images = None
    
    async def on_timeout(self) -> None:
        print(f'{self.__class__.__name__} Timeout:', datetime.datetime.now())
        for item in self.children:
            item.disabled = True
        self.clear_items()
        self.message = await self.message.edit(view=self)
        self.stop()
    
    async def send(self, message: discord.Message, out_gifpath:Path, prompt: str):
        self.message = message
        self.gif_filepath = Path(out_gifpath)
        self.prompt = prompt
        #self.filenames = [self.out_gifpath.with_stem(self.out_gifpath.stem + f'_{i}').with_suffix('.webp').name for i in range(len(self.images))]
        
        # TODO: if too big, etiher:
        # - halve the number of frames
        # - reduce size (w,h) of gif
        # - use catbox hosting
        # - return exploded view with the collapse button disabled. 
        #   - FIXME: Currently will error out if you try to collapse

        message, self.optimized_images = await imgutil.try_send_gif(self.message, out_gifpath, prompt, view=self)
        if message is not None:
            self.message = message
        else:
            # failed to send, twice.
            # return exploded view with the collapse button disabled.
            view = PagedImageGridView(prev_view=None, images=self.images, filestems=[str(self.gif_filepath.stem)]*len(self.images), descriptions=None, timeout=self.timeout)
            self.message = await self.message.edit(content=f"Still too big, maybe tone it down a bit next time.", view=view)
            
            self.message = await view.send(self.message)

        #self.message.edit(content='', attachments=[discord.File(fp=out_gifpath, filename=out_gifpath.name, description=prompt)], view=self)
        return self.message
    
    async def refresh(self):
        # asyncio.to_thread(
        image_frames = self.optimized_images if self.optimized_images is not None else self.images
        img_file = await asyncio.to_thread(imgutil.gif_to_bfile, image_frames=image_frames, filestem=self.gif_filepath.stem, description=self.prompt)
        self.message = await self.message.edit(content='', attachments=[img_file], view=self)
        
        #self.message = await imgutil.try_send_gif(msg=self.message, gif_filepath=self.gif_filepath, prompt=self.prompt, view=self)
        
        return self.message


    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.secondary, disabled=False, emoji='↗️', row=0) # row=4 # 💢 ⛶  label='🗗︎', 
    async def explode_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        button.disabled = True
        button.emoji = '⏳'
        await interaction.response.edit_message(view=self)
        button.disabled = False
        button.emoji = '↗️'
        
        view = PagedImageGridView(self, images=self.images, filestems=[str(self.gif_filepath.stem)]*len(self.images), descriptions=None, timeout=self.timeout)
        self.message = await view.send(self.message)
        


class ImageGridView(discord.ui.View):
    def __init__(self, prev_view:DrawUIView, *, timeout=None):
        timeout = timeout if timeout is not None else prev_view.timeout
        super().__init__(timeout=timeout)
        self.prev_view = prev_view
    
    async def on_timeout(self) -> None:
        print(f'{self.__class__.__name__} Timeout:', datetime.datetime.now())
        for item in self.children:
            item.disabled = True
        self.clear_items().stop()
        #await self.draw_view.message.edit(view=self)
    
    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.secondary, disabled=False, emoji='↙️', row=0) # row=4 # 💢 \u200b ↩️ ↵ label='↩', 
    async def collapse_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        return await self.prev_view.refresh()
        #self.message = await self.message.edit(content='',attachments=files, view=self.draw_view)


class PagedImageGridView(discord.ui.View):
    def __init__(self, prev_view:GifUIView|DrawUIView|None, images:list[Image.Image], filestems: list[str], descriptions: list[str] = None, *, max_grid_images:int=10, timeout=None):
        super().__init__(timeout=(timeout if timeout is not None else prev_view.timeout))
        self.prev_view = prev_view
        self.max_grid_images = max_grid_images
        self.image_attrs_batches, self.thumb_attrs_batches = self._init_batches(images, filestems, descriptions)
        self.message: discord.Message

        
        self.n_batches = len(self.image_attrs_batches)
        self.cur_batchnum = 1
        self.display_timer = None
        self.page_update_delay = 2 #1 #0.75 # seconds

        self._init_buttons()

    async def send(self, message: discord.Message):
        self.message = message
        await self.send_fullsize()
        return self.message
    
    def _init_buttons(self):
        if self.prev_view is None:
            self.collapse_button.disabled = True
        
        self.prev_button.disabled = self.n_batches < 2 # (self.cur_imgnum <= 1)
        self.next_button.disabled = self.n_batches < 2 # (self.cur_imgnum >= self.n_images)
        self.update_buttons()
    
    def _init_batches(self, images:list[Image.Image], filestems: list[str], descriptions: list[str]):
        n_images = len(images)
        assert len(filestems) == n_images, 'All images must a filestem name'
        
        if descriptions is None:
            descriptions = [None]*n_images

        image_attrs = [{'image':img, 'fstem':stem, 'desc':desc} for img,stem,desc in zip(images, filestems, descriptions)]
        image_attrs_batches = list(comutil.batched(image_attrs, n = self.max_grid_images))

        thumb_attrs = [{'image':img, 'fstem':stem, 'desc':desc} for img,stem,desc in zip(
            imgutil.to_thumbnails(images, max_size=(256,256)), 
            [f'{fs}_thb_{i}' for i,fs in enumerate(filestems)], 
            descriptions)]

        thumb_attrs_batches = list(comutil.batched(thumb_attrs, n = self.max_grid_images))

        return image_attrs_batches, thumb_attrs_batches



    def interrupt_refresh(self):
        if self.display_timer:
           self.display_timer.cancel()

    def freeze(self):
        self.interrupt_refresh()

        for item in self.children:
            item.disabled = True
        

    async def on_timeout(self) -> None:
        print(f'{self.__class__.__name__} Timeout:', datetime.datetime.now())
        # https://discordpy.readthedocs.io/en/stable/faq.html#how-can-i-disable-all-items-on-timeout
        self.freeze()
        self.clear_items()
        await self.message.edit(view=self)
        #self.message = await self.message.edit(view=self)
        #self.stop()
        #del self.image_attrs_batches
        
    
    def update_buttons(self):
        if self.cur_batchnum < 1:
            self.cur_batchnum = self.n_batches
        elif self.cur_batchnum > self.n_batches:
            self.cur_batchnum = 1
        
        #self.collapse_button.disabled = False
        self.counter_button.label = f'({self.cur_batchnum} / {self.n_batches})'
    
    async def file_coro(self, img_attr:dict, ext:str, **kwargs):
        return imgutil.to_discord_file(image=img_attr['image'], filestem=img_attr['fstem'], description=img_attr['desc'], ext=ext, **kwargs)
    
    async def async_file_batch(self, img_thumbnails:bool, ext:str, **kwargs):
        index = self.cur_batchnum-1
        img_batch = self.thumb_attrs_batches[index] if img_thumbnails else self.image_attrs_batches[index]
        #try:
        for img_attr in img_batch:
            #task=asyncio.create_task(asyncio.to_thread(imgutil.to_discord_file, image=img_attr['image'], filestem=img_attr['fstem'], description=img_attr['desc'], ext=ext, **kwargs))
            
            #yield asyncio.to_thread(imgutil.to_discord_file, image=img_attr['image'], filestem=img_attr['fstem'], description=img_attr['desc'], ext=ext, **kwargs)
            yield imgutil.to_discord_file(image=img_attr['image'], filestem=img_attr['fstem'], description=img_attr['desc'], ext=ext, **kwargs)

    
    
    async def send_thumbnails(self):
        #index = self.cur_batchnum-1
        #thumb_batch = self.thumb_attrs_batches[index]
        #img_files = [imgutil.to_discord_file(thb_attr['image'], thb_attr['fstem'], thb_attr['desc'], ext='JPEG', optimize=True, quality=50) for thb_attr in thumb_batch]
        #img_files = await asyncio.to_thread(self.file_batch, img_batch = thumb_batch, ext='JPEG', optimize=False, quality=30)
        
       
        #img_files = [await file async for file in self.async_file_batch(img_thumbnails=True, ext='JPEG', optimize=False, quality=30)]
        img_files = [file async for file in self.async_file_batch(img_thumbnails=True, ext='JPEG', optimize=False, quality=30)]
        
        #async for img_attr in self.async_img_batch(thumbnails=True):
        #    img_files.append(imgutil.to_discord_file(image=img_attr['image'], filestem=img_attr['fstem'], description=img_attr['desc'], ext='JPEG', optimize=False, quality=30))
        #img_files = await self.file_batch( img_batch = thumb_batch, ext='JPEG', optimize=False, quality=30) #asyncio.to_thread(self.file_batch, img_batch = thumb_batch, ext='JPEG', optimize=False, quality=30)

        self.message = await self.message.edit(attachments=img_files, view=self)
        
    

    async def send_fullsize(self):
        #print('sending full size')
        #index = self.cur_batchnum-1
        #img_batch = self.image_attrs_batches[index]
        ##self.message = await self.message.edit(attachments=[], view=self)
        #img_files = [imgutil.to_discord_file(img_attr['image'], img_attr['fstem'], img_attr['desc'], ext='WebP') for img_attr in img_batch] #  lossless=True, quality=0
        
        #img_files = [await file async for file in self.async_file_batch(img_thumbnails=False, ext='WebP')] #  lossless=True, quality=0
        img_files = [file async for file in self.async_file_batch(img_thumbnails=False, ext='WebP')] #  lossless=True, quality=0
        
        #img_files = [imgutil.to_discord_file(image=img_attr['image'], filestem=img_attr['fstem'], description=img_attr['desc'], ext='WebP') async for img_attr in self.async_img_batch(False)]
        #img_files = await self.file_batch(img_batch = img_batch, ext='WebP')
        #img_files = await asyncio.to_thread(self.file_batch, img_batch = img_batch, ext='WebP')
        self.message = await self.message.edit(attachments=img_files, view=self)
        

    async def update_ui(self, interaction:discord.Interaction):
        self.interrupt_refresh()
        await interaction.response.defer()
        self.update_buttons()
        # img_files = [imgutil.to_bfile(img, fname, desc) for img,fname,desc in self.thumb_attrs_batches[self.cur_batchnum-1]]
        # img_files = [imgutil.to_discord_file(img, None, None) for img,fname,desc in self.thumb_attrs_batches[self.cur_batchnum-1]]
        await self.send_thumbnails()
        #await interaction.response.edit_message(view=self)
        
        #self.message = await self.message.edit(view=self)
        #self.display_timer = asyncio.get_event_loop().call_later(self.page_update_delay,self.refresh)
        self.display_timer = asyncio.create_task(self.refresh())
        

    async def refresh(self, delay:float = None):
        if delay is None:
            delay = self.page_update_delay
                
        await asyncio.sleep(delay)
        
        await self.send_fullsize()
        return self.message
        #return self.message
        #return await self.refresh(files)
        
    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.secondary, disabled=False, emoji='↙️', row=1) # row=4 # 💢 \u200b ↩️ ↵ label='↩', 
    async def collapse_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        self.freeze() # Need this or clicking an arrow will make the view comeback
        button.disabled = True
        button.emoji = '⏳'
        await interaction.response.edit_message(view=self)
        #await interaction.response.defer()
        #self.message = await self.message.edit(view=self)
        #await interaction.response.edit_message(view=self)
        
        #await interaction.response.defer()
        #self.message = await self.previous_view.refresh()
        
        #return await self.prev_view.refresh()
        return await self.prev_view.refresh()
        #self.message = await self.prev_view.refresh()
        #button.disabled = False
        #button.emoji = '↙️'
        #await self.on_timeout()
        #await interaction.delete_original_response()
        #self.message = await self.previous_view.refresh()
        #return
        #try_send_gif(self.message, self.out_gifpath, self.previous_view.prompt, view=self.previous_view)
         #await self.message.edit(view=self.previous_view)
        #return await self.previous_view.refresh()
    
    @discord.ui.button(label='(0 / 0)', style=discord.ButtonStyle.grey, disabled=True,  row=1) # ▶️ 🔄 ♻️ emoji='🔄',
    async def counter_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()

    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.primary, disabled=True, emoji='⬅️', row=1)
    async def prev_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        #await interaction.response.defer()
        self.cur_batchnum -= 1
        await self.update_ui(interaction)
        
    
    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.primary, disabled=True, emoji='➡️', row=1)
    async def next_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        #await interaction.response.defer()
        self.cur_batchnum += 1
        await self.update_ui(interaction)
        
class ConfigModal(discord.ui.Modal, title='Tweaker Menu'):
    prompt = discord.ui.TextInput(label='Prompt', style=TextStyle.paragraph, required=True, min_length=1, max_length=1000)
    negative_prompt = discord.ui.TextInput(label='Negative Prompt', style=TextStyle.short, required=False, max_length=300)
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