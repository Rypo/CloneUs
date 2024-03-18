# This example requires the 'message_content' privileged intent to function.
import io
import json
import yaml
import traceback
import typing
from typing import Optional
from collections import OrderedDict

import discord
from discord.enums import TextStyle
from discord.ext import commands
from discord.utils import MISSING

from PIL import Image

param_defaults = {
    'sdxl': {
        'prompt':None,
        'neg_prompt':None,
        'steps': 50,
        'strength': 55,
        'guidance': 10,
        'stage_mix': None,
        'refine_strength':30,
    },
    'sdxl_turbo': {
        'prompt':None,
        'neg_prompt':None,
        'steps': 4,
        'strength': 55,
        'guidance': 0,
        'stage_mix': None,
        'refine_strength':None,
    }
}
def to_bytes_file(image:Image.Image, prompt: str):
    filename = f'image_{hash(image.tobytes())}.png' 
    with io.BytesIO() as imgbin:
        image.save(imgbin, 'PNG')
        imgbin.seek(0)
        
        return discord.File(fp=imgbin, filename=filename, description=prompt)
class DrawUIView(discord.ui.View):
    def __init__(self, kwargs, *, timeout=None):
        super().__init__(timeout=timeout)
        
        self.call_ctx = None
        self.image_url=None
        self.message = None
        self.kwargs = kwargs
        #self.image_file = image_file
        #self.images = [image]
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
        self.call_kwargs = self.call_ctx.kwargs.copy()
        
        self.kwargs = {
            'prompt': self.call_kwargs['prompt'], 
            **self.kwargs, 
            'fast': self.call_kwargs['fast']
        }
        self.kwargs['neg_prompt'] = self.kwargs.pop('negative_prompt')
        self.kwargs['guidance'] = self.kwargs.pop('guidance_scale')
        
        if 'denoise_blend' in self.call_kwargs:
            self.kwargs['stage_mix'] = self.kwargs.pop('denoise_blend', None)
        
        if 'image_url' in self.call_kwargs:
            self.kwargs['image_url'] = self.call_kwargs['image_url']
            
        print(self.kwargs)
        #await self.add_image(self.images[self.cur_index])
        #bfile = to_bytes_file(image, self.call_ctx.kwargs['prompt'])
        self.message = await ctx.send(view=self)
        await self.add_image(image)
        #self.message = await ctx.send(file=bfile, view=self, ephemeral=ephemeral, )
        #self.n_images += 1
        #self.cur_imgnum += 1
        #self.image_url=image_url
        #await self.update_message(self.data, image_url=image_url)
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
            # Insert UPSAMPLE image in place
        

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
        cur_file = await self.attachments[self.cur_imgnum-1].to_file(use_cached=True)#.read(use_cached=True)
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
            image_url = self.message.attachments[0].url, 
            prompt = kwargs['prompt'], 
            steps = kwargs['steps'], 
            strength = 30,
            neg_prompt = kwargs['neg_prompt'],
            guidance = kwargs['guidance'],
            stage_mix = kwargs.get('stage_mix',None),
            #refine_strength = 30.0, 
            fast = kwargs['fast']
        )
        
        #self.already_upsampled.append(self.cur_imgnum+1) # +1 because it's incremented before the view is updated in add_image
        await self.add_image(image_file, True)
        
        await interaction.delete_original_response()
        #await (await interaction.followup.send('\u200b', wait=True)).delete()
        # print(interaction.message.attachments)
        # print(self.call_ctx.command)
        # print(self.call_ctx.author)
        
    
    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.secondary, disabled=False, emoji='🔄', row=1)
    async def redo_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer(thinking=True)
        kwargs = self.kwargs.copy()

        # This needs to be generic because can be /draw OR /redraw
        image_file = await self.call_ctx.invoke(self.call_ctx.command, *self.call_ctx.args, **kwargs)
        await self.add_image(image_file, is_upsample=False)

        await interaction.delete_original_response()

        
    @discord.ui.button(label='\u200b', style=discord.ButtonStyle.secondary, disabled=False, emoji='⚙️', row=1) # 🎛️ ⚙️ 🎚️
    async def config_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        #await interaction.response.defer()
        kwargs = self.kwargs.copy()
        
        cm = ConfigModal(vals={
            'prompt': kwargs.pop('prompt'),
            'neg_prompt': kwargs.pop('neg_prompt'),
            'image_url': kwargs.pop('image_url', None),
            'config': yaml.dump(kwargs, sort_keys=False)
        })
        #cm = ConfigModal({'config': json.dumps(self.call_ctx.kwargs, indent=1)})
        
        await interaction.response.send_modal(cm) 
        await cm.wait()

        config = yaml.load(cm.vals['config'], yaml.SafeLoader)
        
        self.kwargs.update({
            'prompt':cm.vals['prompt'],
            'neg_prompt':cm.vals['neg_prompt'],
            'image_url':cm.vals['image_url'],
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
    neg_prompt = discord.ui.TextInput(label='Negative Prompt', style=TextStyle.short, required=False, max_length=300)
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
        self.neg_prompt.default = self.vals.get('neg_prompt')
        self.config.default = self.vals.get('config')
        self.image_url.default = self.vals.get('image_url')
        
        
    async def on_submit(self, interaction: discord.Interaction):
        self.vals['prompt'] = self.prompt.value
        self.vals['neg_prompt'] = self.neg_prompt.value
        self.vals['config'] = self.config.value
        self.vals['image_url'] = self.image_url.value
        
        await interaction.response.defer()
        self.stop()
        
    
    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        await interaction.response.send_message('Oops! Done goofed.', ephemeral=True)
        traceback.print_exception(type(error), error, error.__traceback__)

class RedrawUIView(discord.ui.View):
    def __init__(self, model_name='sdxl', *, timeout=None):
        super().__init__(timeout=timeout)
        
        # self.field_map = OrderedDict({
        #     'prompt': ('Prompt', False, 'primary'),
        #     'steps': ('Steps', True, 'primary'),
        #     'strength': ('Strength', True, 'primary'),
        #     'neg_prompt': ('Negative Prompt', False, 'secondary'),
        #     'guidance': ('Guidance Scale', True, 'secondary'),
        #     'stage_mix': ('Stage Mix', True, 'secondary'),
        #     'refine_strength': ('Refine Strength', True, 'secondary'),
        # })
        self.field_map = OrderedDict({
            'prompt': {'name': 'Prompt', 'inline': False, 'option': 'primary'},
            'steps': {'name': 'Steps', 'inline': True, 'option': 'primary'},
            'strength': {'name': 'Strength', 'inline': True, 'option': 'primary'},
            'neg_prompt': {'name': 'Negative Prompt', 'inline': False, 'option': 'secondary'},
            'guidance': {'name': 'Guidance Scale', 'inline': True, 'option': 'secondary'},
            'stage_mix': {'name': 'Stage Mix', 'inline': True, 'option': 'secondary'},
            'refine_strength': {'name': 'Refine Strength', 'inline': True, 'option': 'secondary'},
        })
        self.data = param_defaults[model_name]
        if model_name == 'sdxl_turbo':
            self.secondary_button.style=discord.ButtonStyle.gray
            self.secondary_button.disabled = True
            self.secondary_button.label = 'SDXL Only'
        
        self.model_name = model_name
        self.image_url=None
        self.embed = None

    async def send(self, ctx: commands.Context, image_url=None, image_filename=None, ephemeral=False):
        
        self.message = await ctx.send(view=self, ephemeral=ephemeral)
        if image_filename is not None:
            image_url=f"attachment://{image_filename}"

        self.image_url=image_url
        await self.update_message(self.data, image_url=image_url)
        return self.message
        
    
    async def update_message(self, data, image_url=None):
        self.update_buttons()
        await self.message.edit(embed=self.create_embed(data, image_url=image_url), view=self)

    def update_buttons(self):
        if self.data.get('prompt') is not None:
            self.primary_button.style = discord.ButtonStyle.success
            self.go_button.disabled = False
            self.go_button.style = discord.ButtonStyle.success
            

    @discord.ui.button(label="Primary Options", style=discord.ButtonStyle.primary,  disabled=False)
    async def primary_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        #await interaction.response.defer()
        pm = PrimaryOptionsModal(vals=self.data)
        #pm.set_vals(self.data)
        await interaction.response.send_modal(pm) # title='Test feed title'
        await pm.wait()
        
        self.data.update(pm.vals)
        await self.update_message(self.data)
    
    @discord.ui.button(label="Secondary Options", style=discord.ButtonStyle.secondary,  disabled=False)
    async def secondary_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        
        sm = SecondaryOptionsModal(self.data)
        
        await interaction.response.send_modal(sm) 
        await sm.wait()
        self.data.update(sm.vals)
        await self.update_message(self.data)

    @discord.ui.button(label="Remix!", style=discord.ButtonStyle.gray,  disabled=True, emoji="🎨")
    async def go_button(self, interaction:discord.Interaction, button: discord.ui.Button):        
        print(self.data)
        self.data['steps'] = int(self.data['steps'])
        self.data['strength'] = float(self.data['strength'])

        if self.data['neg_prompt'] == '':
            self.data['neg_prompt'] = None

        if self.data['guidance']:
            self.data['guidance'] = float(self.data['guidance'])
        
        if self.data['stage_mix']:
            self.data['stage_mix'] = float(self.data['stage_mix'])
        elif self.data['stage_mix'] == '':
            self.data['stage_mix'] = None

        if self.data['refine_strength']:
            self.data['refine_strength'] = float(self.data['refine_strength'])
        elif self.data['refine_strength'] == '':
            self.data['refine_strength'] = None
        
        await interaction.response.defer(thinking=True)
        #interaction.followup.send()
        self.defered_interaction = interaction
        #await interaction.response.defer()
        self.stop()
        
        #await interaction.response.send_message('Heeeeere we gooooo!', ephemeral=True, delete_after=1)
    
    @discord.ui.button(label="X", style=discord.ButtonStyle.danger, disabled=False, row=4)
    async def delete_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        
        await self.message.delete()
        self.clear_items()

    def create_embed(self, data:dict, image_url=None):
        if self.embed is None:
            desc = ("> Click 'Primary Options' button to open the menu.\n"
                    "> Use a square image for better results.\n"
                    "> For best results use 1024x1024 (512x512 Turbo).\n")
                    
            args_desc = (
                    "> **Prompt**: Description for desired new image.\n"
                    "> **Steps**: Iters to run. More= ⬆Quality & ⬆Time.\n"
                    "> **Strength**: Image change amount. 0=None, 100=Everything.\n") # How much to change the initial image.
            sargs_desc = (
                    "> **Neg. Prompt**: Whatever you don't want in the new image.\n"
                    "> **Guidance**: More= ⬆Prompt Adhere & ⬇Quality, ⬇Creative.\n" # Importance of Prompt.
                    "> **Stage Mix**: Base/Refine blend factor. ↓Quality & ⬇Run Time.\n" # ↓ ⇣ ⬇ ⇓
                    "> **Refine Str.**: Image refine amount. 0=None, 100=Too much.\n") # How much to refine img.
            
            
            if self.model_name == 'sdxl':
                desc += "> Click 'Secondary Options' for more options.\n"
                args_desc += sargs_desc
            
            desc += "\n"+args_desc+"\n"

            self.embed = discord.Embed(title=f"Image Redraw Arguments", color=discord.Color.dark_gold(), description=desc,)
            
            for argname, vals in self.field_map.items():
                if self.model_name == 'sdxl' or vals['option']=='primary':
                    self.embed.add_field(name=vals['name'], value=data.get(argname, self.data.get(argname)), inline=vals['inline'])

                    
        else:
            for i,(argname, vals) in enumerate(self.field_map.items()):
                if self.model_name == 'sdxl' or vals['option']=='primary':
                    self.embed.set_field_at(i, name=vals['name'], value=data.get(argname, self.data.get(argname)), inline=vals['inline'])
            
        
        if image_url:
            #file = discord.File("path/to/my/image.png", filename="image.png")
            #self.embed.set_image(url=image_url)
            self.embed.set_thumbnail(url=image_url)

        return self.embed



class PrimaryOptionsModal(discord.ui.Modal, title='Primary Options (SDXL+Turbo)'):    
    prompt = discord.ui.TextInput(label='Prompt', style=TextStyle.paragraph, required=True, 
                                placeholder='Enter your text prompt here...', min_length=1, max_length=500,)
    steps = discord.ui.TextInput(label='Steps', style=TextStyle.short, required=False, default=None)#vals['steps'])
    strength = discord.ui.TextInput(label='Strength', style=TextStyle.short, required=False, min_length=1, max_length=3)#, default=vals['strength']

    def __init__(self, vals=None, *args, **kwargs):
        self.vals = vals
        super().__init__(*args, **kwargs)
        self._set_default_values()

    
    def _set_default_values(self, vals=None):
        if vals is not None:
            self.vals = vals

        self.prompt.default = self.vals.get('prompt')
        self.steps.default = self.vals.get('steps', self.steps.default)
        self.strength.default = self.vals.get('strength', self.strength.default)
        
        
    async def on_submit(self, interaction: discord.Interaction):
        self.vals['prompt'] = self.prompt.value
        self.vals['steps']=self.steps.value
        self.vals['strength']=self.strength.value
        
        await interaction.response.defer()
        self.stop()
        
    
    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        await interaction.response.send_message('Oops! Done goofed.', ephemeral=True)
        traceback.print_exception(type(error), error, error.__traceback__)

class SecondaryOptionsModal(discord.ui.Modal, title='Secondary Options (Only SDXL)'):
    neg_prompt = discord.ui.TextInput(label='Negative Prompt', style=TextStyle.long, required=False, 
                                        placeholder="Enter anything you don't want in the image...", max_length=300)
    guidance = discord.ui.TextInput(label='Guidance', style=TextStyle.short, required=False, min_length=1, max_length=4,)# default=vals['guidance'])
    stage_mix = discord.ui.TextInput(label='Stage Mix', style=TextStyle.short, required=False, max_length=3,)# default=vals['stage_mix'])
    refine_strength = discord.ui.TextInput(label='Refine Strength', style=TextStyle.short, required=False, max_length=3,)# default=vals['refine_strength'])
    
    def __init__(self, vals=None, *args, **kwargs):
        self.vals = vals
        super().__init__(*args, **kwargs)
        self._set_default_values()


    def _set_default_values(self, vals:dict=None):
        if vals is not None:
            self.vals = vals
        self.neg_prompt.default = self.vals.get('neg_prompt', self.neg_prompt.default)
        self.guidance.default = self.vals.get('guidance', self.guidance.default)
        self.stage_mix.default = self.vals.get('stage_mix', self.stage_mix.default)
        self.refine_strength.default = self.vals.get('refine_strength', self.refine_strength.default)
    
    # def set_vals(self, vals:dict):
    #     self.neg_prompt.default=vals.get('neg_prompt', self.neg_prompt.default)
    #     self.guidance.default=vals.get('guidance', self.guidance.default)
    #     self.stage_mix.default=vals.get('stage_mix', self.stage_mix.default)
    #     self.refine_strength.default=vals.get('refine_strength', self.refine_strength.default)
        
    #     self.vals = vals
    
    async def on_submit(self, interaction: discord.Interaction):
        self.vals['neg_prompt'] = self.neg_prompt.value
        self.vals['guidance'] = self.guidance.value
        self.vals['stage_mix'] = self.stage_mix.value
        self.vals['refine_strength'] = self.refine_strength.value
        await interaction.response.defer()

        self.stop()
        
    
    async def on_error(self, interaction: discord.Interaction, error: Exception) -> None:
        await interaction.response.send_message('Oops! Done goofed.', ephemeral=True)
        traceback.print_exception(type(error), error, error.__traceback__)




# src: https://github.com/Rapptz/discord.py/blob/master/examples/views/tic_tac_toe.py
# Defines a custom button that contains the logic of the game.
# The ['TicTacToe'] bit is for type hinting purposes to tell your IDE or linter
# what the type of `self.view` is. It is not required.
class TicTacToeButton(discord.ui.Button['TicTacToe']):
    def __init__(self, x: int, y: int):
        # A label is required, but we don't need one so a zero-width space is used
        # The row parameter tells the View which row to place the button under.
        # A View can only contain up to 5 rows -- each row can only have 5 buttons.
        # Since a Tic Tac Toe grid is 3x3 that means we have 3 rows and 3 columns.
        super().__init__(style=discord.ButtonStyle.secondary, label='\u200b', row=y)
        self.x = x
        self.y = y

    # This function is called whenever this particular button is pressed
    # This is part of the "meat" of the game logic
    async def callback(self, interaction: discord.Interaction):
        assert self.view is not None
        view: TicTacToe = self.view
        state = view.board[self.y][self.x]
        if state in (view.X, view.O):
            return

        if view.current_player == view.X:
            self.style = discord.ButtonStyle.danger
            self.label = 'X'
            self.disabled = True
            view.board[self.y][self.x] = view.X
            view.current_player = view.O
            content = "It is now O's turn"
        else:
            self.style = discord.ButtonStyle.success
            self.label = 'O'
            self.disabled = True
            view.board[self.y][self.x] = view.O
            view.current_player = view.X
            content = "It is now X's turn"

        winner = view.check_board_winner()
        if winner is not None:
            if winner == view.X:
                content = 'X won!'
            elif winner == view.O:
                content = 'O won!'
            else:
                content = "It's a tie!"

            for child in view.children:
                child.disabled = True

            view.stop()

        await interaction.response.edit_message(content=content, view=view)


# This is our actual board View
class TicTacToe(discord.ui.View):
    # This tells the IDE or linter that all our children will be TicTacToeButtons
    # This is not required
    children: typing.List[TicTacToeButton]
    X = -1
    O = 1
    Tie = 2

    def __init__(self):
        super().__init__()
        self.current_player = self.X
        self.board = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]

        # Our board is made up of 3 by 3 TicTacToeButtons
        # The TicTacToeButton maintains the callbacks and helps steer
        # the actual game.
        for x in range(3):
            for y in range(3):
                self.add_item(TicTacToeButton(x, y))

    # This method checks for the board winner -- it is used by the TicTacToeButton
    def check_board_winner(self):
        for across in self.board:
            value = sum(across)
            if value == 3:
                return self.O
            elif value == -3:
                return self.X

        # Check vertical
        for line in range(3):
            value = self.board[0][line] + self.board[1][line] + self.board[2][line]
            if value == 3:
                return self.O
            elif value == -3:
                return self.X

        # Check diagonals
        diag = self.board[0][2] + self.board[1][1] + self.board[2][0]
        if diag == 3:
            return self.O
        elif diag == -3:
            return self.X

        diag = self.board[0][0] + self.board[1][1] + self.board[2][2]
        if diag == 3:
            return self.O
        elif diag == -3:
            return self.X

        # If we're here, we need to check if a tie was made
        if all(i != 0 for row in self.board for i in row):
            return self.Tie

        return None




