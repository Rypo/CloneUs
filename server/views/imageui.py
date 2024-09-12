# This example requires the 'message_content' privileged intent to function.
import io
import time
import json
import yaml
import asyncio
import tempfile
import random
import traceback
import typing
from pathlib import Path

import discord
from discord.enums import TextStyle
from discord.ext import commands
from discord.utils import MISSING

from PIL import Image

from utils import image as imgutil

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
    def __init__(self, kwargs:dict, *, timeout=None):
        super().__init__(timeout=timeout)
        
        self.call_ctx = None
        self.image_url=None
        self.message = None
        self.upsample_message = None
        #self.upsample_thread = None
        #self.kwargs = kwargs
        kwargs['seed'] = kwargs.get('seed', random.randint(1e9, 1e10-1))
        self.kwarg_sets = [kwargs]

        self.attachments: list[discord.Attachment] = []
        self.local_paths = []
        self.images = []
        #self.files: list[discord.File] = []
        #self.already_upsampled = []
        self.cur_filename = None
        self.cur_imgnum = 0
        self.n_images = 0
        self.is_redraw: bool = None
        self.in_queue=None
        self.mcounter=None
        self.call_fn:typing.Callable=None
    
    @property
    def kwargs(self):
        return self.kwarg_sets[self.cur_imgnum-1]

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True
        del self.images, self.attachments, self.local_paths
        self.clear_items()
        self.message = await self.message.edit(view=self)
        if self.mcounter:
            await self.mcounter.delete()

    async def send(self, ctx: commands.Context, image: Image.Image, fpath: str):
        self.call_ctx = ctx
        #call_kwargs = {**self.call_ctx.kwargs} # prompt, imgfile
        #flags = call_kwargs.pop('flags')
        kwargs = self.kwarg_sets[-1]
        # kwargs = {
        #     **call_kwargs,
        #     **kwargs, 
        #     'fast': flags.fast,
        # }
        print('MERGED KWARGS:', kwargs)
        self.kwarg_sets=[kwargs]
        self.is_redraw = 'imgurl' in kwargs or 'imgfile' in kwargs
        imgen = self.call_ctx.bot.get_cog(self.call_ctx.cog.qualified_name)
        # default_auto = (self.is_redraw and flags.aspect is None)
        self.call_fn = imgen.redraw if self.is_redraw else imgen.draw
        # if self.is_redraw:
        #     self.aspect_select.add_option(label='auto', description=f'Auto (best match)', emoji='💠', default=default_auto)
        
        # for soption in ['🔲 square (1:1)', '📱 portrait (13:19)', '🖼️ landscape (19:13)']:
        #     emoji, label, ar = soption.split()
        #     is_default = not default_auto and self.kwargs['aspect']==label
        #     self.aspect_select.add_option(label=label, description=f'{label.title()} {ar}', emoji=emoji, default=is_default)

        self.message = await ctx.send(view=self)
        #self.upsample_message = await self.message.reply('<Embiggened Images>')
        #self.upsample_message = await ctx.send('<Embiggened Images>')
        #self.upsample_thread = await ctx.channel.create_thread(name='<Embiggened Images>', message=self.message)
        # https://github.com/Rapptz/discord.py/issues/9008#issuecomment-1299999500
        # print(self.call_ctx.command)
        # print(self.call_ctx.author)
        await self.add_image(image, fpath)

        return self.message

    def update_buttons(self):
        #print('before mod:', self.cur_imgnum)
        if self.cur_imgnum < 1:
            self.cur_imgnum = self.n_images
        elif self.cur_imgnum > self.n_images:
            self.cur_imgnum = 1
        
        self.prev_button.disabled = self.n_images < 2 # (self.cur_imgnum <= 1)
        self.next_button.disabled = self.n_images < 2 # (self.cur_imgnum >= self.n_images)
        self.explode_button.disabled = self.n_images < 2

        self.redo_button.label = f'({self.cur_imgnum} / {self.n_images})'
        

    async def update_view(self, img_file: discord.File = None):
        self.update_buttons()
        #print(f'index: {self.cur_imgnum}, items: {self.n_images}')
        if img_file is None:
            self.message = await self.message.edit(view=self)
        else:
            self.message = await self.message.edit(attachments=[img_file], view=self)
        
        return self.message.attachments[0]

    async def add_image(self, image: Image.Image, fpath:str):
        self.n_images += 1
        self.cur_imgnum  = self.n_images
        
        bfile = imgutil.to_bytes_file(image, self.kwargs['prompt'])
        cur_attach = await self.update_view(bfile)
        #self.files.append(bfile)
        self.attachments.append(cur_attach)
        self.images.append(image)
        self.local_paths.append(fpath)
        

    async def refresh(self):
        self.update_buttons()
        index = self.cur_imgnum-1
        print(f'Index: {index} | Local Paths: {[p.name for p in self.local_paths]}')
        print(f'kwarg_sets:', [k['prompt'] for k in self.kwarg_sets])
        print(f'kwarg_sets[{index}]:', self.kwarg_sets[index])
        # time is comparable for both methods. 0.5 - 1.0 seconds
        # try:
        #    cur_file = await self.attachments[index].to_file(use_cached=True)
        #    await self.update_view(cur_file)
        # except discord.errors.NotFound:
        #     print('Content expired. Falling back to local file.')
        
        expired_attach = self.attachments[index]
        #cur_file = discord.File(self.local_paths[index], filename=expired_attach.filename, description=expired_attach.description)
        cur_file = imgutil.to_bfile(self.images[index], filestem=expired_attach.filename, description=expired_attach.description)
        
        cur_attach = await self.update_view(cur_file)
        self.attachments[index] = cur_attach #.insert(self.cur_imgnum, cur_attach)
        return self.message

    async def update_queue(self, d:int, interaction:discord.Interaction=None):
        if self.mcounter is None:
            self.in_queue = 0
            self.mcounter = await interaction.followup.send(content = f'Queue: {self.in_queue}', wait=True, silent=True)
        
        self.in_queue += d
        # if self.in_queue <= 0:
        #     await self.mcounter.delete()
        #     self.mcounter = None
        # else:
        self.mcounter = await self.mcounter.edit(content = f'Queue: {self.in_queue}',)

    async def redo(self, interaction:discord.Interaction, kwargs:dict):
        # await interaction.response.defer(thinking=True)
        await interaction.response.defer()
        await self.update_queue(1, interaction)
        #kwargs = self.kwargs.copy()

        # need to reset refine strength in case was changed in modal.
        # otherwise would be refining on every single redo
        kwargs.pop('refine_strength',None)
        
        image, fpath = await asyncio.create_task(self.call_ctx.invoke(self.call_fn, *self.call_ctx.args, **kwargs))
        # NOTE: I bet you append these to a list and pop them. Then you'd be able to cancel queued jobs
        # if self.is_redraw:
        #     image, fpath = await self.call_ctx.invoke(imgen.redraw, *self.call_ctx.args, **kwargs)
        # else:
        #     #kwargs.pop('imgfile',None) # Still need to pop in case it's passed as None 
        #     image, fpath = await self.call_ctx.invoke(imgen.draw, *self.call_ctx.args, **kwargs)
        
        self.kwarg_sets.append(kwargs)
        await self.add_image(image, fpath)
        await self.update_queue(-1, interaction)
        #await interaction.delete_original_response()


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
    
    @discord.ui.button(label='(0 / 0)', style=discord.ButtonStyle.secondary, disabled=False, emoji='🔄', row=1) # ▶️ 🔄 ♻️
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

    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.secondary, disabled=False, emoji='⚙️', row=1) # 🎛️ ⚙️ 🎚️
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
            new_kwargs.update({'imgfile': cm.vals['image_url']})
        
        #self.kwarg_sets.append(new_kwargs)
        # NOTE: if decide to do this, remove the defer from on_submit
        await self.redo(cm.submit_interaction, new_kwargs)
    

    @discord.ui.button(label='HD', style=discord.ButtonStyle.green, disabled=False, emoji='⬇️', row=1) # 🆙
    async def upsample_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        # TODO: Maybe just make it a toggle? like -- red: HD (off), green: HD (on)
        await interaction.response.defer(thinking=True)
        # print(interaction.message.attachments)
        kwargs = self.kwargs.copy()
        imgen = self.call_ctx.bot.get_cog(self.call_ctx.cog.qualified_name)
        kwargs['imgurl'] = self.message.attachments[0].url
        kwargs['imgfile'] = self.message.attachments[0]
        
        if (ref_strength := kwargs.get('refine_strength')) is None:
            ref_strength = 0.30
        
        
        hd_kwargs = {
            'imgurl': self.message.attachments[0].url,
            'prompt': kwargs.get('prompt'),
            'imgfile': self.message.attachments[0],
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
        image, fpath = await self.call_ctx.invoke(imgen.hd_upsample, *self.call_ctx.args, **hd_kwargs)
        
        #await interaction.followup.send(file=to_bytes_file(image_file, kwargs['prompt']))
        ifile = imgutil.to_bytes_file(image, kwargs['prompt'], ext='PNG')
        
        # if self.upsample_thread is None:
        #    self.upsample_thread = await self.call_ctx.channel.create_thread(name='Upsampled Images', message=self.message)
        
        # await self.upsample_thread.send(file=ifile)

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
        
        if self.n_images <= 10:
            files = [imgutil.to_bfile(img, att.filename, att.description) for img,att in zip(self.images,self.attachments)]#attach.to_file() for attach in self.attachments]
            self.message = await self.message.edit(content='',attachments=files, view=ImageGridView(self))
        else:
            filestems, descriptions = list(zip(*[(Path(at.filename).stem, at.description) for at in self.attachments]))
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
        
        view = PagedImageGridView(self, images=self.images, filestems=[str(self.gif_filepath.stem)]*len(self.images), descriptions=None)
        self.message = await view.send(self.message)
        


class ImageGridView(discord.ui.View):
    def __init__(self, prev_view:DrawUIView, *, timeout=None):
        timeout = timeout if timeout is not None else prev_view.timeout
        super().__init__(timeout=timeout)
        self.prev_view = prev_view
    
    async def on_timeout(self) -> None:
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
        self.message = await self.send_fullsize()
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
        image_attrs_batches = list(imgutil.batched(image_attrs, n = self.max_grid_images))

        thumb_attrs = [{'image':img, 'fstem':stem, 'desc':desc} for img,stem,desc in zip(
            imgutil.to_thumbnails(images, max_size=(256,256)), 
            [f'{fs}_thb_{i}' for i,fs in enumerate(filestems)], 
            descriptions)]

        thumb_attrs_batches = list(imgutil.batched(thumb_attrs, n = self.max_grid_images))

        return image_attrs_batches, thumb_attrs_batches



    def interrupt_refresh(self):
        if self.display_timer:
           self.display_timer.cancel()

    def freeze(self):
        self.interrupt_refresh()

        for item in self.children:
            item.disabled = True
        

    async def on_timeout(self) -> None:
        # https://discordpy.readthedocs.io/en/stable/faq.html#how-can-i-disable-all-items-on-timeout
        self.freeze()
        self.clear_items()
        self.message = await self.message.edit(view=self)
        self.stop()
        #del self.image_attrs_batches
        
    
    def update_buttons(self):
        if self.cur_batchnum < 1:
            self.cur_batchnum = self.n_batches
        elif self.cur_batchnum > self.n_batches:
            self.cur_batchnum = 1
        
        #self.collapse_button.disabled = False
        self.counter_button.label = f'({self.cur_batchnum} / {self.n_batches})'
    
    def file_batch(self, img_batch, ext, **kwargs):
        # https://docs.python.org/3/library/asyncio-task.html#running-in-threads
        return [imgutil.to_discord_file(img_attr['image'], img_attr['fstem'], img_attr['desc'], ext=ext, **kwargs) for img_attr in img_batch]
        #return [await asyncio.to_thread(imgutil.to_discord_file, image=img_attr['image'], filestem=img_attr['fstem'], description=img_attr['desc'], ext=ext, **kwargs) for img_attr in img_batch]
    
    async def send_thumbnails(self):
        index = self.cur_batchnum-1
        thumb_batch = self.thumb_attrs_batches[index]
        #img_files = [imgutil.to_discord_file(thb_attr['image'], thb_attr['fstem'], thb_attr['desc'], ext='JPEG', optimize=True, quality=50) for thb_attr in thumb_batch]
        img_files = await asyncio.to_thread(self.file_batch, img_batch = thumb_batch, ext='JPEG', optimize=False, quality=30)
        return await self.message.edit(attachments=img_files, view=self)
    

    async def send_fullsize(self):
        index = self.cur_batchnum-1
        img_batch = self.image_attrs_batches[index]
        ##self.message = await self.message.edit(attachments=[], view=self)
        #img_files = [imgutil.to_discord_file(img_attr['image'], img_attr['fstem'], img_attr['desc'], ext='WebP') for img_attr in img_batch] #  lossless=True, quality=0

        img_files = await asyncio.to_thread(self.file_batch, img_batch = img_batch, ext='WebP')
        return await self.message.edit(attachments=img_files, view=self)

    async def update_ui(self, interaction:discord.Interaction):
        self.interrupt_refresh()
        await interaction.response.defer()
        self.update_buttons()
        # img_files = [imgutil.to_bfile(img, fname, desc) for img,fname,desc in self.thumb_attrs_batches[self.cur_batchnum-1]]
        # img_files = [imgutil.to_discord_file(img, None, None) for img,fname,desc in self.thumb_attrs_batches[self.cur_batchnum-1]]
        self.message = await self.send_thumbnails()
        #await interaction.response.edit_message(view=self)
        
        #self.message = await self.message.edit(view=self)
        
        self.display_timer = asyncio.create_task(self.refresh())

    async def refresh(self, delay:float = None):
        if delay is None:
            delay = self.page_update_delay
                
        await asyncio.sleep(delay)
        
        self.message = await self.send_fullsize()
        #return self.message
        #return await self.refresh(files)
        
    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.secondary, disabled=False, emoji='↙️', row=1) # row=4 # 💢 \u200b ↩️ ↵ label='↩', 
    async def collapse_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        self.freeze() # Need this or clicking an arrow will make the view comeback
        button.disabled = True
        button.emoji = '⏳'
        await interaction.response.defer()
        self.message = await self.message.edit(view=self)
        #await interaction.response.edit_message(view=self)
        
        #await interaction.response.defer()
        #self.message = await self.previous_view.refresh()
        
        #return await self.prev_view.refresh()
        self.message = await self.prev_view.refresh()
        
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