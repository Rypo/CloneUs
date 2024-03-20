# This example requires the 'message_content' privileged intent to function.
import io
import json
import yaml
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

class DrawUIView(discord.ui.View):
    def __init__(self, kwargs:dict, *, timeout=None):
        super().__init__(timeout=timeout)
        
        self.call_ctx = None
        self.image_url=None
        self.message = None
        self.kwargs = kwargs

        self.attachments: list[discord.Attachment] = []
        self.attach_urls = []
        self.already_upsampled = []
        self.cur_filename = None
        self.cur_imgnum = 0
        self.n_images = 0
        
    async def on_timeout(self) -> None:
        for item in self.children:
            item.disabled = True
        
        self.clear_items()
        await self.message.edit(view=self)

    async def send(self, ctx: commands.Context, image: Image.Image, ephemeral=False):
        self.call_ctx = ctx
        self.call_kwargs = {**self.call_ctx.kwargs} # prompt, imgfile
        flags = self.call_kwargs.pop('flags')
        
        self.kwargs = {
            **self.call_kwargs,
            #'prompt': self.call_kwargs['prompt'], 
            **self.kwargs, 
            'fast': flags.fast,
        }
        print('MERGED KWARGS:', self.kwargs)

        self.message = await ctx.send(view=self)
        await self.add_image(image)

        return self.message

    def update_buttons(self):
        self.prev_button.disabled = (self.cur_imgnum <= 1)
        self.next_button.disabled = (self.cur_imgnum >= self.n_images)

        self.page_button.label = f'{self.cur_imgnum} / {self.n_images}'

        self.upsample_button.disabled = self.cur_filename in self.already_upsampled
        print(f'index: {self.cur_imgnum}, items: {self.n_images}')
        print(self.already_upsampled)

    async def update_message(self, img_file: discord.File = None):
        if img_file is not None:
            self.message = await self.message.edit(attachments=[img_file], view=self)
        else:
            self.message = await self.message.edit(view=self)

    async def add_image(self, image_file, is_upsample=False):
        self.n_images += 1
        
        if not is_upsample:
            # put REDO images at the end of the list
            self.cur_imgnum  = self.n_images
            
        bfile = to_bytes_file(image_file, self.kwargs['prompt'])
        self.cur_filename = bfile.filename
        
        if is_upsample:
            self.already_upsampled.append(bfile.filename)
        
        self.update_buttons()
        await self.update_message(bfile)
        cur_attach = self.message.attachments[0]

        #print('Add '+ ("upsample" if is_upsample else ""), 'curFilename:', self.cur_filename)
        
        print(f'Inserting {cur_attach.filename} at index: {self.cur_imgnum}')
        self.attachments.insert(self.cur_imgnum, cur_attach)
        if is_upsample:
            self.cur_imgnum+=1
            self.update_buttons()
            await self.update_message()
            # self.message = await self.message.edit(view=self)

    async def refresh(self):
        # TODO: handle CDN expired
        # discord.errors.NotFound: 404 Not Found (error code: 0): asset not found
        cur_file = await self.attachments[self.cur_imgnum-1].to_file(use_cached=False)#.read(use_cached=True)
        self.cur_filename = cur_file.filename
        
        self.upsample_button.disabled = (self.cur_filename in self.already_upsampled)
        await self.update_message(cur_file)
        
        #self.update_buttons()
        print('Attach_file:', self.cur_filename)


    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.secondary,  disabled=False, emoji='🆙', row=1)
    async def upsample_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(thinking=True)
        
        kwargs = self.kwargs.copy()
        imgen = self.call_ctx.bot.get_cog(self.call_ctx.cog.qualified_name)

        image_file = await self.call_ctx.invoke(
            imgen.redraw,
            imgfile = self.message.attachments[0],#.url, 
            prompt = kwargs['prompt'], 
            steps = kwargs['steps'], 
            strength = kwargs.get('refine_strength', kwargs.get('strength', 30)), # 30
            negative_prompt = kwargs['negative_prompt'],
            guidance_scale = kwargs['guidance_scale'],
            refine_steps = max(kwargs['refine_steps'], 1),
            denoise_blend = kwargs.get('denoise_blend', None),
            fast = kwargs['fast']
        )
        
        await self.add_image(image_file, True)
        await interaction.delete_original_response()

        # print(interaction.message.attachments)
        # print(self.call_ctx.command)
        # print(self.call_ctx.author)
        
    
    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.secondary, disabled=False, emoji='🔄', row=1)
    async def redo_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(thinking=True)
        kwargs = self.kwargs.copy()

        imgen = self.call_ctx.bot.get_cog(self.call_ctx.cog.qualified_name)
        if kwargs.get('imgfile'):
            image_file = await self.call_ctx.invoke(imgen.redraw, *self.call_ctx.args, **kwargs)
        else:
            kwargs.pop('imgfile',None) # Still need to pop in case it's passed as None 
            image_file = await self.call_ctx.invoke(imgen.draw, *self.call_ctx.args, **kwargs)

        await self.add_image(image_file, is_upsample=False)

        await interaction.delete_original_response()

        
    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.secondary, disabled=False, emoji='⚙️', row=1) # 🎛️ ⚙️ 🎚️
    async def config_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        #await interaction.response.defer()
        kwargs = self.kwargs.copy()
        image_url =  kwargs.pop('imgfile', None)

        if isinstance(image_url, discord.Attachment):
            image_url = image_url.url
        
        cm = ConfigModal(vals={
            'prompt': kwargs.pop('prompt'),
            'negative_prompt': kwargs.pop('negative_prompt'),
            'image_url': image_url,
            'config': yaml.dump(kwargs, sort_keys=False)
        })
        #cm = ConfigModal({'config': json.dumps(self.call_ctx.kwargs, indent=1)})
        
        await interaction.response.send_modal(cm) 
        await cm.wait()

        config = yaml.load(cm.vals['config'], yaml.SafeLoader)
        
        #image_url = cm.vals['image_url']
        self.kwargs.update({
            'prompt':cm.vals['prompt'],
            'negative_prompt':cm.vals['negative_prompt'],
            'imgfile': cm.vals['image_url'],
            **config
        })


    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.secondary, disabled=False, emoji='❌', row=1) # row=4
    async def delete_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        self.stop()
        self.clear_items()
        await self.message.edit(view=self)
        
    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.primary, disabled=True, emoji='⬅️', row=2)
    async def prev_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        if self.cur_imgnum > 1:
            self.cur_imgnum -= 1
        self.update_buttons()
        await self.refresh()
        #await self.update_message(self.current_page_data)
    
    @discord.ui.button(label='0 / 0', style=discord.ButtonStyle.secondary, disabled=True, row=2)
    async def page_button(self, interaction:discord.Interaction, button: discord.ui.Button):

        self.update_buttons()
        #await self.refresh()

    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.primary, disabled=True, emoji='➡️', row=2)
    async def next_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        if self.cur_imgnum < self.n_images:
            self.cur_imgnum += 1
        self.update_buttons()
        await self.refresh()
        #await self.update_message(self.current_page_data)

    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.primary, disabled=False, emoji='⬇️', row=2)
    async def down_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        #await interaction.response.defer()
        active_file = await self.message.attachments[0].to_file()
        await interaction.response.send_message(file=active_file)
        msg = await interaction.original_response()
        
        #self.cur_imgnum += 1
        #await self.refresh()
class ConfigModal(discord.ui.Modal, title='Config Tweaker'):
    prompt = discord.ui.TextInput(label='Prompt', style=TextStyle.paragraph, required=True, min_length=1, max_length=500)
    negative_prompt = discord.ui.TextInput(label='Negative Prompt', style=TextStyle.short, required=False, max_length=300)
    config = discord.ui.TextInput(label='Config', style=TextStyle.paragraph, required=False, min_length=1, max_length=1000,) # style=TextStyle.short,
    image_url = discord.ui.TextInput(label='Image URL', style=TextStyle.short, required=False) # style=TextStyle.short,

    def __init__(self, vals=None, *args, **kwargs):
        self.vals = vals
        super().__init__(*args, **kwargs)
        self._set_default_values()

    def _set_default_values(self, vals=None):
        if vals is not None:
            self.vals = vals

        self.prompt.default = self.vals.get('prompt')
        self.negative_prompt.default = self.vals.get('negative_prompt')
        self.config.default = self.vals.get('config')
        self.image_url.default = self.vals.get('image_url')
        
        
    async def on_submit(self, interaction: discord.Interaction):
        self.vals['prompt'] = self.prompt.value
        self.vals['negative_prompt'] = self.negative_prompt.value
        self.vals['config'] = self.config.value
        self.vals['image_url'] = self.image_url.value
        
        await interaction.response.defer()
        self.stop()
        
    
    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        await interaction.response.send_message('Oops! Done goofed.', ephemeral=True)
        traceback.print_exception(type(error), error, error.__traceback__)