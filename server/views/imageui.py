# This example requires the 'message_content' privileged intent to function.
import io
import time
import json
import yaml
import random
import traceback
import typing

import discord
from discord.enums import TextStyle
from discord.ext import commands
from discord.utils import MISSING

from PIL import Image


def to_bytes_file(image:Image.Image, prompt: str):
    filename = f'image_{hash(image.tobytes())}.png' 
    with io.BytesIO() as imgbin:
        image.save(imgbin, 'PNG')
        imgbin.seek(0)
        
        return discord.File(fp=imgbin, filename=filename, description=prompt)

def to_bfile(image:Image.Image, filename: str, description:str, ):
    #if filename is None:
    #    filename = f'image_{hash(image.tobytes())}.png' 
    with io.BytesIO() as imgbin:
        image.save(imgbin, 'JPEG')
        imgbin.seek(0)
        
        return discord.File(fp=imgbin, filename=filename, description=description)
    
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
    
    @property
    def kwargs(self):
        return self.kwarg_sets[self.cur_imgnum-1]

    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True
        del self.images, self.attachments, self.local_paths
        self.clear_items()
        await self.message.edit(view=self)

    async def send(self, ctx: commands.Context, image: Image.Image, fpath: str, ephemeral=False):
        self.call_ctx = ctx
        call_kwargs = {**self.call_ctx.kwargs} # prompt, imgfile
        flags = call_kwargs.pop('flags')
        kwargs = self.kwarg_sets[-1]
        kwargs = {
            **call_kwargs,
            **kwargs, 
            'fast': flags.fast,
        }
        print('MERGED KWARGS:', kwargs)
        self.kwarg_sets=[kwargs]
        self.is_redraw = 'imgurl' in kwargs or 'imgfile' in kwargs
        
        # default_auto = (self.is_redraw and flags.aspect is None)
        
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
        
        bfile = to_bytes_file(image, self.kwargs['prompt'])
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
        cur_file = to_bfile(self.images[index], filename=expired_attach.filename, description=expired_attach.description)
        
        cur_attach = await self.update_view(cur_file)
        self.attachments[index] = cur_attach #.insert(self.cur_imgnum, cur_attach)
        
    async def redo(self, interaction:discord.Interaction, kwargs:dict):
        await interaction.response.defer(thinking=True)
        #kwargs = self.kwargs.copy()

        # TODO: Should refine steps be ignored here? for now, yes.
        # since they are seperated in config Modal
        #kwargs['refine_steps'] = 0
        #self.kwarg_sets.append(kwargs)
        
        imgen = self.call_ctx.bot.get_cog(self.call_ctx.cog.qualified_name)
        #if kwargs.get('imgfile'):
        if self.is_redraw:
            image, fpath = await self.call_ctx.invoke(imgen.redraw, *self.call_ctx.args, **kwargs)
        else:
            #kwargs.pop('imgfile',None) # Still need to pop in case it's passed as None 
            image, fpath = await self.call_ctx.invoke(imgen.draw, *self.call_ctx.args, **kwargs)
        
        self.kwarg_sets.append(kwargs)
        await self.add_image(image, fpath)
        await interaction.delete_original_response()


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
        
        hd_config = {'refine_steps': kwargs.pop('refine_steps'), 'refine_strength':kwargs.pop('refine_strength')}
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
        kwargs['strength'] = 0 # set to 0 to avoid strong redraw.
        kwargs['refine_steps'] = max(kwargs['refine_steps'], 1)
        #kwargs['denoise_blend'] = kwargs.get('denoise_blend', None),
        
        # Pop aspect so that dimensions are preserved
        kwargs.pop('aspect', None)
        image, fpath = await self.call_ctx.invoke(imgen.redraw, *self.call_ctx.args, **kwargs)
        
        #await interaction.followup.send(file=to_bytes_file(image_file, kwargs['prompt']))
        ifile = to_bytes_file(image, kwargs['prompt'])
        
        # if self.upsample_thread is None:
        #    self.upsample_thread = await self.call_ctx.channel.create_thread(name='Upsampled Images', message=self.message)
        
        # await self.upsample_thread.send(file=ifile)
        
        # if self.upsample_message.content:
        #     self.upsample_message = await self.upsample_message.edit(content=None, attachments=[ifile])
        # else:
        #     self.upsample_message = await self.upsample_message.add_files(ifile)

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
        await interaction.response.defer()
        print('Exploded Image View')
        # 10 is max image attachments in a message
        image_attachs = zip(self.images,self.attachments)
        if self.n_images > 10:
            msg = await interaction.followup.send('Only 10 images can be displayed at once. Showing last 10.', ephemeral=True, wait=True)
            image_attachs = zip(self.images[-10:],self.attachments[-10:])
            await msg.delete(delay=2.0)
        files = [to_bfile(img, att.filename, att.description) for img,att in image_attachs]#attach.to_file() for attach in self.attachments]
        # self.message = await self.message.edit(content='',attachments=files, view=self)
        self.message = await self.message.edit(content='',attachments=files, view=GridUIView(self))
        
        #self.message = await self.message.edit(content='', attachments=self.attachments, view=self)
        #self.message = await self.message.edit(content='',attachments=files, view=self)
        return
        
class GridUIView(discord.ui.View):
    def __init__(self, draw_view:DrawUIView, *, timeout=None):
        timeout = timeout if timeout is not None else draw_view.timeout
        super().__init__(timeout=timeout)
        self.draw_view = draw_view
    
    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True
        self.clear_items()
        await self.draw_view.message.edit(view=self)
    
    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.secondary, disabled=False, emoji='↙️', row=0) # row=4 # 💢 \u200b ↩️ ↵ label='↩', 
    async def collapse_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        print('Collapsed Explode View')
        await interaction.response.defer()
        return await self.draw_view.refresh()
        #self.message = await self.message.edit(content='',attachments=files, view=self.draw_view)


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