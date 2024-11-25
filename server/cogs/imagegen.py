import re
import io

import typing
import asyncio
import logging
import datetime
import functools
from pathlib import Path

import discord
from discord import app_commands
from discord.ext import commands

from diffusers.utils import load_image, make_image_grid

from PIL import Image
import imageio.v3 as iio # pip install -U "imageio[ffmpeg, pyav]" # ffmpeg: mp4, pyav: trans_png

from managers import imgman
import config.settings as settings

from cmds import transformers as cmd_tfms, choices as cmd_choices, flags as cmd_flags
from utils.command import check_up
from utils.reporting import StatusItem
from utils.globthread import stop_global_executors, wrap_async_executor, async_gen
from utils import image as imgutil
from views import imageui
from run import BotUs

cmds_logger = logging.getLogger('server.cmds')
event_logger = logging.getLogger('server.event')
color_logger = logging.getLogger('pconsole') 
# Resource: https://github.com/CyberTimon/Stable-Diffusion-Discord-Bot/blob/main/bot.py

GUI_TIMEOUT = 30*60 # NOTE: limit is 15 minutes unless webhook message is converted to message


class PromptCanceled(Exception): 
    '''Raised when user cancels command after prompt spellcheck.'''

class Autocorrect:
    def __init__(self, model_relpath:str='extras/models/jamspell/en.bin') -> None:
        import jamspell
        from cloneus import cpaths
        self.corrector = jamspell.TSpellCorrector()
        self.corrector.LoadLangModel(str(cpaths.ROOT_DIR/model_relpath))
        self.ignored_texts = set()
        self.ignored_words = set()

    def __call__(self, text:str) -> tuple[str, bool]:
        return self.filtered_correction(text=text)
        #fixed_text = self.corrector.FixFragment(text)
        #return fixed_text, fixed_text!=text

    def correct(self, text:str):
        return self.corrector.FixFragment(text)
    
    def check(self, text:str):
        return self.corrector.FixFragment(text) != text
    
    def filtered_correction(self, text:str):
        # whole phrase ignored
        if text in self.ignored_texts:
            return text, False
        
        new_text = self.correct(text)
        
        # no corrections
        if new_text == text:
            return text, False
        
        corrections,nfilt = self.get_corrections(text, new_text, filter_ignored=True)
        
        # all corrections filtered
        if not corrections:
            return text, False
        
        # no correction filtered
        if nfilt == 0:
            return new_text, True
        
        # some filtered, some not
        new_text = text
        for o,n in corrections:
            new_text = new_text.replace(o, n)
        
        return new_text, True


    def ban_texts(self, old:str, new:str):
        self.ignored_texts.add(old)

        corrected,_ = self.get_corrections(old, new)
        for o,_ in corrected:
            self.ignored_words.add(o)
        
    def get_corrections(self, old:str, new:str, filter_ignored:bool = True):
        corrections = [(o,n) for o,n in zip(old.split(), new.split()) if o != n]
        n_chg = len(corrections)
        if filter_ignored:
            corrections = list(filter(lambda o_: o_[0] not in self.ignored_words, corrections))
        n_filtered = n_chg - len(corrections)
        return corrections, n_filtered

    @staticmethod
    def show_diff(old:str, new:str):
        diffed = []
        for oword,nword in zip(old.split(), new.split()):
            if oword != nword:
                diffed.append(f'[[[ ~~{oword}~~ **{nword}** ]]]')
            else:
                diffed.append(oword)

        return ' '.join(diffed)


class SetImageConfig:
    bot: BotUs
    igen: imgman.BaseSDXLManager|imgman.BaseFluxManager|imgman.BaseSD3Manager

    def argsettings(self):
        return [
            #('Image Seed', self.igen.global_seed,''),
         ]
    
    @commands.hybrid_group(name='iset')#, fallback='arg')#, description='Quick set a value', aliases=[])
    async def isetarg(self, ctx: commands.Context):
        '''(GROUP). Image Settings. call `!help set` to see sub-commands.'''
        if ctx.invoked_subcommand is None:
            await ctx.send(f"{ctx.subcommand_passed} does not belong to iset")
    
    @isetarg.command(name='model', aliases=['artist',])
    @app_commands.choices(version=cmd_choices.IMAGE_MODELS)
    async def iset_model(self, ctx: commands.Context, version: str, offload: bool=False):
        '''Loads in the image generation model
        
        Args:
            version: Image model name
            offload: If True, model will be moved off GPU when not in use to save vRAM (Default=False) 
        '''
        # NOTE: this will make precheck error messages non-ephemeral.  
        await self.view_check_defer(ctx)
        disp_name = imgman.AVAILABLE_MODELS[version]['desc']
        if self.igen.model_name != version or self.igen.offload != offload:
            if self.igen.is_ready:
                await self.igen.unload_pipeline()
            
            self.igen = imgman.AVAILABLE_MODELS[version]['manager'](offload=offload)
            return await self.imgup(ctx)
        elif not self.igen.is_ready:
            return await self.imgup(ctx)
        else:
            return await ctx.send(f'{disp_name} already up')


    async def sampler_autocomplete(self, interaction: discord.Interaction, current: str,) -> list[app_commands.Choice[str]]:
        if not self.igen.is_ready:
            return []
        compat_schedulers = self.igen.available_schedulers(return_aliases=True)
        return [app_commands.Choice(name=sched, value=sched) for sched in compat_schedulers if current.lower() in sched.lower()]

    @isetarg.command(name='scheduler')
    @app_commands.autocomplete(sampler=sampler_autocomplete)
    async def iset_scheduler(self, ctx: commands.Context, sampler: str, schedtype: typing.Literal['Karras','sgm_uniform','exponential','beta', 'ts_leading']=None):
        '''Change the image generation scheduler
        
        Args:
            sampler: The sampler's nickname
            schedtype: The scheduler type nickname
        '''
        if not self.igen.is_ready:
            return await ctx.send(f'Image model not loaded! Call `!imgup` first.')
                    
        
        scheduler_clsname = self.igen.set_scheduler(sampler_alias=sampler, schedtype_alias=schedtype)
        
        scheduler_name = f'{sampler} - {schedtype}' if schedtype is not None else sampler
        return await ctx.send(f'Scheduler set: {scheduler_name} ({scheduler_clsname})')
    
    @isetarg.command(name='pag')
    async def iset_pag(self, ctx: commands.Context, scale: float=3.0, layers: str='mid', disable:bool=False):
        '''Configure PAG (Perturbed-Attention Guidance) for model if possible
        
        Args:
            scale: PAG scale. Similar to guidance. Can improve quality+adherence, too high over saturates. Default=3.
            layers: Network layers to apply PAG. Can use comma for list. Default='mid'.
            disable: If true, disable PAG completely. Default=Fasle.
        '''
        if not self.igen.is_ready:
            return await ctx.send(f'Image model not loaded! Call `!imgup` first.')
        
        try:
            scale = 0.0 if disable else scale
            pag_config = self.igen.set_pag_config(scale, layers.split(','), disable_on_zero=disable)
            return await ctx.send(f'```json\nPAG {"Enabled" if scale else "Disabled"}: {pag_config}\n```')
        except NotImplementedError:
            return await ctx.send('PAG not implemented for this model type. Try an SDXL based model.')

    
class ImageGen(commands.Cog, SetImageConfig): #commands.GroupCog, group_name='img'
    '''Suite of tools for generating images.'''
    
    def __init__(self, bot: BotUs):
        self.bot = bot
        #self.igen = imgman.ColorfulXLLightningManager(offload=False)
        # self.igen = imgman.FluxSchnevManager(offload=False)
        self.igen = imgman.JuggernautXIManager(offload=False)
        #self.igen = imgman.SD35MediumManager(offload=False)
        self.spell_check = Autocorrect()
        self.msg_views: dict[int, discord.ui.View] = {}

        self._log_extra = {'cog_name': self.qualified_name}
        self._load_lock = asyncio.Lock()
        self._model_load_commands = set(['▶ Replay CMD']+['draw', 'redraw','hd','animate', 'reanimate', 'optimize'])

    async def cog_unload(self):
        await self.bot.wait_until_ready()
        await self.igen.unload_pipeline()
        #self.bot.tree.remove_command(self.ctx_menu.name, type=self.ctx_menu.type)
        for view in self.msg_views.values():
            if not view.is_finished():
                await view.on_timeout()
        stop_global_executors()
    
    async def cog_before_invoke(self, ctx: commands.Context) -> None:
        if not self.igen.is_ready:
            if ctx.command.name in self._model_load_commands:
                _ = await self.view_check_defer(ctx)
                msg = await self.imgup(ctx.channel)
            

    async def cog_after_invoke(self, ctx: commands.Context) -> None:
        cmd_status = 'FAIL' if ctx.command_failed else 'PASS'
        pos_args = [a for a in ctx.args[2:] if a]
        
        cmds_logger.info('[{stat}] {a.display_name}({a.name}) command "{c.prefix}{c.invoked_with} ({c.command.qualified_name})" args:({args}, {c.kwargs})'.format(
            stat=cmd_status, a=ctx.author, c=ctx, args=pos_args), extra=self._log_extra)
    

    @commands.Cog.listener('on_reaction_add')
    async def on_reaction_add(self, reaction: discord.Reaction, user: discord.User):
        if reaction.message.author == self.bot.user:
            if str(reaction.emoji) == '🔞':
                
                view = self.msg_views.get(reaction.message.id, discord.utils.MISSING)
                if view:
                    reaction.message = await view.set_blur(True) # so, that works for forcing an update. good to know.
                
                    #print(view, [a.is_spoiler() for a in reaction.message.attachments])
                    #await view.refresh()
                else:
                    #print('NO VIEW HERE ADD')
                    #attachs = [await attach.to_file(spoiler=True) for attach in reaction.message.attachments]
                    attachs = asyncio.gather(*[attach.to_file(spoiler=True) for attach in reaction.message.attachments])
                    # reaction.message = await reaction.message.edit(attachments=attachs, view=view)
                    
                    msg = await reaction.message.edit(attachments=[], view=view)
                    msg = await msg.edit(attachments=await attachs, view=view)
                
            #event_logger.info(f'[REACT] {user.display_name}({user.name}) ADD "{reaction.emoji}" TO '
            #                '{a.display_name}({a.name}) message {reaction.message.content!r}'.format(a=reaction.message.author, reaction=reaction), extra=self._log_extra)
    
    @commands.Cog.listener('on_reaction_remove')
    async def on_reaction_remove(self, reaction: discord.Reaction, user: discord.User):
        
        if reaction.message.author == self.bot.user:
            if str(reaction.emoji) == '🔞':
                view = self.msg_views.get(reaction.message.id, discord.utils.MISSING)
                if view:
                    reaction.message = await view.set_blur(False)
                    
                    #print(view, 'is_spoliers:', [a.is_spoiler() for a in reaction.message.attachments])
                    #await view.refresh()
                else:
                    
                    #attachs = [await attach.to_file(spoiler=True) for attach in reaction.message.attachments]
                    attachs = asyncio.gather(*[attach.to_file(spoiler=False) for attach in reaction.message.attachments])
                    # reaction.message = await reaction.message.edit(attachments=attachs, view=view)
                    
                    msg = await reaction.message.edit(attachments=[], view=view)
                    msg = await msg.edit(attachments=await attachs, view=view)


    @commands.Cog.listener('on_message_delete')
    async def on_message_delete(self, message: discord.Message):
        if message.id in self.msg_views:
            view = self.msg_views[message.id]
            
            if not view.is_finished():
                await view.on_timeout()
            self.msg_views.pop(message.id)
            #print(f'Message Deleted: {message.author} - "{message.content}"')
            event_logger.info('[DELETE] {a.display_name}({a.name}) message {message.content!r} view {view}'.format(
                a=message.author, message=message, view=view), extra=self._log_extra)
        else:
            event_logger.debug('Message without view deleted.', extra=self._log_extra)

    async def view_check_defer(self, ctx: commands.Context, view=None):
        needs_view = (view is None or view.message is None)
        # interact = ctx if isinstance(ctx, discord.Interaction) else ctx.interaction
        
        if ctx.interaction and not ctx.interaction.response.is_done():
            await ctx.defer()
            await asyncio.sleep(1)
            
        
        return needs_view
    
    def fix_prompt(self, prompt:str, check_spelling:bool = True):
        was_corrected = False

        if prompt is None:
            return '', False
        
        if check_spelling:
            prompt, was_corrected = self.spell_check(prompt)
        
        if len(prompt) > 1000:
            prompt = prompt[:1000]+'...' # Will error out if >1024 chars.

        return prompt, was_corrected
    
    async def validate_prompt(self, ctx: commands.Context, prompt:str):
        prompt_new, was_sp_corrected = self.fix_prompt(prompt)
        
        if was_sp_corrected:
            #diff_old, diff_new = self.spell_check.show_diff(prompt, prompt_new)
            diffed = self.spell_check.show_diff(prompt, prompt_new)
            view = imageui.ConfirmActionView(timeout=30)
            diffstr = (
                f'>>> {diffed}'
            )

            msg = await ctx.channel.send(f'Accept changes?\n{diffstr}', view=view)#, ephemeral=True)
            await view.wait()
            if view.value == '<CANCEL>':
                await msg.delete()
                raise PromptCanceled
            keep_changes =  view.value or view.value is None
            if not keep_changes:
                self.spell_check.ban_texts(prompt, prompt_new)
                prompt_new,was_sp_corrected = self.fix_prompt(prompt, check_spelling=False)
                
            await msg.delete()
        return prompt_new

    def status_list(self, keep_advanced:bool=True, ctx=None):
        model_alias = imgman.AVAILABLE_MODELS[self.igen.model_name]['desc']
        keymap=None if keep_advanced else {'guidance_scale':'guidance', 'negative_prompt':'negprompt'}
        status = [
            StatusItem('Image status', ('Up' if self.igen.is_ready else 'Down'), " ✔" if self.igen.is_ready else " ✖"),
            #StatusItem('Model', model_alias, f' (offload: {self.igen.offload})'),#(' ✅' if self.igen.is_ready else ' ❌')),
            StatusItem('Model', model_alias),#(' ✅' if self.igen.is_ready else ' ❌')),
            StatusItem('offload', self.igen.offload), # Note: this show the _unmerged_ length.
            # StatusItem('','---',''),
            #StatusItem('Default settings', value=None, advanced=True),
            StatusItem('Default settings', '', extra=self.igen.config.to_md(keymap)), # .strip()
            
            #StatusItem(, value=None),
        ]
        if self.igen.is_ready:
            status.append(StatusItem(None,None, extra='```json\n' + 'Scheduler ' + self.igen.base.scheduler.to_json_string() + '\n```', advanced=True))
        
        return [s for s in status if keep_advanced or not s.advanced]
    
    @commands.command(name='istatus', aliases=['istat'])
    async def istatus_report(self, ctx: commands.Context):
        '''Full image model status report.'''
        msg = '\n'.join(stat.md() for stat in self.status_list(keep_advanced=True))
        return await ctx.send(msg)
    
    @commands.command(name='imgup', aliases=['iup','imageup'])
    async def imgup(self, ctx: commands.Context):
        '''Loads in the default image generation model'''
        msg = None
        was_called = hasattr(ctx,'command') and ctx.command.name=='imgup'
        was_iset = hasattr(ctx,'command') and ctx.command.qualified_name == 'iset model'
        #was_called = hasattr(ctx, 'channel')
        model_alias = imgman.AVAILABLE_MODELS[self.igen.model_name]['desc']
        async with self._load_lock:
            if not self.igen.is_ready:
                pre_msg = 'Loadin\' in model' if was_iset else 'Warming up default model'
                msg = await ctx.send(f'{pre_msg}: {model_alias} ...', silent=True)
                await self.igen.load_pipeline()
                # Disable if redirecting
                if not settings.DISCORD_SESSION_INTERACTIVE:
                    self.igen.pbar_config(disable=True)
            
            await self.bot.report_state('draw', ready=True)

        #model_alias = imgman.AVAILABLE_MODELS[self.igen.model_name]['desc']
        complete_msg = f'{model_alias} (offload={self.igen.offload}) all fired up'
        complete_msg += self.igen.config.to_md(keymap={'guidance_scale':'guidance', 'negative_prompt':'negprompt'})
        if msg:
            return await msg.edit(content = complete_msg)
        elif was_called:
            return await ctx.send(complete_msg)  
        
            
        
    @commands.command(name='imgdown', aliases=['idown','imagedown'])
    async def imgdown(self, ctx: commands.Context):
        '''Unloads the current image generation model'''
        await self.igen.unload_pipeline()
        await self.bot.report_state('draw', ready=False)
        await ctx.send('Drawing disabled.')
                

    @commands.command(name='optimize', aliases=['compile'])
    #@check_up('igen', '❗ Drawing model not loaded. Call `!imgup`')
    async def optimize(self, ctx: commands.Context):
        '''Compiles the model to make it go brrrr (~20% faster)'''

        if self.igen.is_compiled:
            return await ctx.send('Already going brrrrr')
        if self.igen.offload:
            return await ctx.send("Can't go brrrrr when model is offloaded. Call `iset model` with `offload=False`")
        msg = await ctx.send("Sit tight. This'll take 2-4 minutes...")

        
        await self.view_check_defer(ctx)
        async with (ctx.channel.typing(), self.bot.busy_status('Compiling...')):
            await self.igen.compile_pipeline()
            await msg.edit(content='🏎 I am SPEED. 🏎 ')


    @commands.hybrid_command(name='caption')
    #@check_up('igen', '❗ Drawing model not loaded. Call `!imgup`')
    async def caption(self, ctx: commands.Context, imgurl:str, level:typing.Literal['brief', 'detailed', 'verbose']='verbose', text_only:bool=False):
        """Write a text description for an image.

        Args:
            imgurl: URL of the image to be described.
            level: level of detail in the description. Default=verbose.
            text_only: If True, only return text description, otherwise show the image. Default=False.
        """
        await self.view_check_defer(ctx)
        image_url = imgutil.clean_image_url(imgurl, check_tenor=False)
        # test for gif/mp4/animated file

        image, imgmeta = await imgutil.aload_image(image_url, result_type='np')
        
        image = imgutil.image_fix(image, animated=False, transparency=False)
        image = Image.fromarray(image)
        
        desc = self.igen.caption(image, level)
        file = None if text_only else imgutil.to_bfile(image, description=desc)

        return await ctx.send(desc, file=file)
    
    @commands.hybrid_command(name='ichat')
    #@check_up('igen', '❗ Drawing model not loaded. Call `!imgup`')
    async def imgchat(self, ctx: commands.Context, prompt:str=None, imgurls:str=None, ):
        """Chat with model about an image, gif (.mp4), or multiple images.
        
        Note: prompt defaults to the following if not specificed:
            If imgurls is 1 non-animated image: 
                'Describe this image.'
            If imgurls is 1 animated image
                'Describe this video.'
            If imgurls is >1 non-animated image
                'Identify the similarities between these images.'

        Args:
            prompt: What to ask about the image(s) or gif. Default = "Describe this {image|video}."
            imgurls: URL(s) of the image or gif/mp4. If multiple, separated them with a space.
        """

        if prompt is None and imgurls is None:
            return await ctx.send('You need to set at least 1 of `imgurls` and `prompt`, preferably both', ephemeral=True)
                
        if imgurls is None:
            await self.view_check_defer(ctx)
            desc = await self.igen.vqa_chat(prompt=prompt, images=None)
            return await ctx.send(desc)
                
        
        img_urls = imgurls.split()
        frames_as_video = len(img_urls)==1
        
        color_logger.debug(f'{img_urls=}, {frames_as_video=}, {prompt=}, {imgurls=}')

        image_urls = []
        for image_url in img_urls:
            try:
                image_url = imgutil.clean_image_url(image_url.strip(), check_tenor=True)
            except ValueError:
                return await ctx.send('Looks like your using a tenor link. You need to click the gif in Discord to open the pop-up view then "Copy Link" to get the `.mp4` link', ephemeral=True)
            image_urls.append(image_url)
        
        await self.view_check_defer(ctx)
        
        if prompt is not None:
            try:
                prompt = await self.validate_prompt(ctx, prompt)
            except PromptCanceled:
                return await ctx.send('Canceled', silent=True, delete_after=1)

        images,img_metas = [], []
        bfiles = [] 
        for url in image_urls:
            image, imgmeta = await imgutil.aload_image(url, result_type='np')
            image = imgutil.image_fix(image, animated=True, transparency=False)
            
            images.append(image.squeeze())
            img_metas.append(imgmeta)

            if image.ndim > 3:
                bfiles.append(imgutil.animation_to_bfile(image, ext='MP4', fps=imgmeta.get('fps', 10)))
            else:
                bfiles.append(imgutil.to_bfile(Image.fromarray(image), ext='JPEG'))


        desc = await self.igen.vqa_chat(prompt=prompt, images=images, frames_as_video=frames_as_video, img_metas=img_metas)

        #file = None if text_only else imgutil.to_bfile(image, description=desc)
        return await ctx.send(desc, files=bfiles)
    
    @commands.hybrid_command(name='morph')
    async def morph(self, ctx: commands.Context, imgurl1: str, imgurl2:str, *,
                    midframes:int=28, 
                    imgfile1: discord.Attachment|None=None,
                    imgfile2: discord.Attachment|None=None,
                    ):
        """Morph one image into another image.

        Args:
            imgurl1: Url of start image. Ignored if `imgfile1` is used. 
            imgurl2: Url of end image. Ignored if `imgfile2` is used. 
            midframes: The number of frames to add between the images. Default=28.
            imgfile1: start image attachment. To use, write anything in `imgurl1` and drag+drop image. 
            imgfile2: end image attachment. To use, write anything in `imgurl2` and drag+drop image. 
        """
        
        needs_view = await self.view_check_defer(ctx)
        
        # this may be passed a url string in drawUI config
        image_url1 = imgutil.clean_image_url(imgfile1.url if isinstance(imgfile1, discord.Attachment) else imgurl1, check_tenor=False)
        image_url2 = imgutil.clean_image_url(imgfile2.url if isinstance(imgfile2, discord.Attachment) else imgurl2, check_tenor=False)
        
        images:list[Image.Image] = []
        for image_url in [image_url1,image_url2]:
            image, imgmeta = await imgutil.aload_image(image_url, result_type='np')
            image = imgutil.image_fix(image, animated=False, transparency=False)

            images.append(Image.fromarray(image))
        
        print(images[0].size, images[1].size)

        async with self.bot.busy_status(activity='draw'):
            w,h = images[0].size
            fake_prompt = f'interpolation_n{midframes+2}_{w}x{h}'
            image_frames = self.igen.interpolate(images=images, inter_frames = midframes, batch_size=1, allow_resize=True)
            out_imgpath = imgutil.save_animation_prompt(image_frames, fake_prompt, ext='MP4', optimize=False)
            if needs_view:
                msg = await ctx.send('Morphed')
                view = imageui.GifUIView(image_frames, timeout=GUI_TIMEOUT)
                msg = await view.send(ctx, msg, out_imgpath, fake_prompt)
                self.msg_views[msg.id] = view
            

    @commands.hybrid_command(name='draw')
    #@check_up('igen', '❗ Drawing model not loaded. Call `!imgup`')
    async def _draw(self, ctx: commands.Context, prompt:commands.Range[str,1,1000], *, flags: cmd_flags.DrawFlags):
        """Generate an image from a text prompt description.

        Args:
            prompt: A description of the image to be generated.
            n: The number of images to draw from the prompt.
            steps: Num iters to run. Increase = ⬆Quality, ⬆Run Time. Default varies.

            negprompt: Negative prompt. What to exclude from image. Usually comma sep list of words. Default=None.
            guidance: Guidance scale. Increase = ⬆Prompt Adherence, ⬇Quality, ⬇Creativity. Default varies.
            detail: Detail weight. Value -3.0 to 3.0, >0 = add detail, <0 = remove detail. Default=0.
            aspect: Image aspect ratio (shape). square w=h = 1:1. portrait w<h = 13:19. Default='square'. 
            
            hdstrength: HD steps strength. 0=Alter Nothing. 100=Alter Everything. Default=0.
            fast: Trades image quality for speed - about 2-3x faster. Default=False.
            seed: Random Seed. An arbitrary number to make results reproducable. Default=None.
        """
        # dblend: Percent of `steps` for Base before Refine stage. ⬇Quality, ⬇Run Time. Default=None (SDXL Only).
        # flags can't be created by drawUI, so need to separate out draw/redraw functionality
        #view = imageui.DrawUIView(timeout=GUI_TIMEOUT)
        
        return await self.draw(ctx, prompt, 
                               n_images = flags.n,
                               steps = flags.steps, 
                               negative_prompt = flags.negprompt, 
                               guidance_scale = flags.guidance, 
                               detail_weight= flags.detail,
                               aspect = flags.aspect, 
                               
                               refine_strength = flags.hdstrength, 
                               fast = flags.fast,
                               seed = flags.seed,
                               _view = None
                               )
        
    async def draw(self, ctx: commands.Context, prompt:str, *,
                   n_images:int = 1,
                   steps: int = None, 
                   negative_prompt: str = None, 
                   guidance_scale: float = None, 
                   detail_weight: float = 0,
                   aspect: typing.Literal['square', 'portrait', 'landscape'] = None,
                  
                   refine_strength: float = None, 
                   #denoise_blend: float|None = None, 
                   fast: bool = False,
                   seed: int = None,
                   _view: imageui.DrawUIView = None,
                   ):
        refine_strength = cmd_tfms.percent_transform(refine_strength)
        
        view = imageui.DrawUIView(timeout=GUI_TIMEOUT) if _view is None else _view
        needs_view = await self.view_check_defer(ctx, view)
        try:
            prompt = await self.validate_prompt(ctx, prompt)
        except PromptCanceled:
            return await ctx.send('Canceled', silent=True, delete_after=1)

        async with self.bot.busy_status(activity='draw'):
            self.igen.dc_fastmode(enable=fast)
            output = await self.igen.generate_image(prompt, n_images=n_images, steps=steps, negative_prompt=negative_prompt, guidance_scale=guidance_scale, 
                                                    detail_weight=detail_weight, aspect=aspect, refine_strength=refine_strength, seed=seed)
            
            if needs_view:
                msg = await view.send(ctx, n_init_images = n_images, )
                self.msg_views[msg.id] = view
            
            async for imbatch,kwbatch in async_gen(output):
                await view.add_images(imbatch, call_kwargs=kwbatch)
                


    @commands.hybrid_command(name='redraw')
    #@check_up('igen', '❗ Drawing model not loaded. Call `!imgup`')
    async def _redraw(self, ctx: commands.Context, prompt: commands.Range[str,1,1000], imgurl:str, *, flags: cmd_flags.RedrawFlags):
        """Remix an image from a text prompt and image.

        Args:
            prompt: A description of what you want to infuse the image with.
            imgurl: Url of image. Ignored if `imgfile` is used. 
            n: The number of redrawn images to produce from the image and prompt.
            imgfile: image attachment. To use, write anything in `imgurl` and drag+drop image. 
            steps: Num of iters to run. Increase = ⬆Quality, ⬆Run Time. Default varies.
            strength: How much to change input image. 0 = Change Nothing. 100=Change Completely. Default varies.
            
            negprompt: What to exclude from image. Usually comma sep list of words. Default=None.
            guidance: Guidance scale. Increase = ⬆Prompt Adherence, ⬇Quality, ⬇Creativity. Default varies.
            detail: Detail weight. Value -3.0 to 3.0, >0 = add detail, <0 = remove detail. Default=0.
            aspect: Image aspect ratio (shape). If None, will pick nearest to imgfile.

            hdstrength: HD steps strength. 0=Alter Nothing. 100=Alter Everything. Ignored unless > 0.
            fast: Trades image quality for speed - about 2-3x faster. Default=False.
            seed: Random Seed. An arbitrary number to make results reproducable. Default=None.
        """
        # hdsteps: High Definition steps. If > 0, image is upscaled 1.5x and refined. Default=0. Usually < 3.
        
        return await self.redraw(ctx, prompt, imgurl, 
                                 n_images=flags.n,
                                 imgfile=flags.imgfile, 
                                 steps = flags.steps, 
                                 strength = flags.strength, 
                                 negative_prompt = flags.negprompt, 
                                 guidance_scale = flags.guidance, 
                                 detail_weight= flags.detail,
                                 aspect = flags.aspect, 
                                 
                                 refine_strength = flags.hdstrength, 
                                 fast = flags.fast,
                                 seed = flags.seed,
                                 _view = None
                                 )

    async def redraw(self, ctx: commands.Context, prompt: str, imgurl:str, *, 
                     imgfile: discord.Attachment|None=None,
                     n_images:int = 1,
                     steps: int = None, 
                     strength: float = None, 
                     negative_prompt: str = None, 
                     guidance_scale: float = None, 
                     detail_weight: float = 0,
                     aspect: typing.Literal['square', 'portrait', 'landscape'] = None,
                
                     refine_strength: float = None,
                     fast:bool = False,
                     seed:int = None,
                     _view: imageui.DrawUIView = None,
                     ):
        
        # manual conversion may be needed because discord transformer doesn't proc when function called internally (e.g. by GUI redo button)
        strength = cmd_tfms.percent_transform(strength)
        refine_strength = cmd_tfms.percent_transform(refine_strength)
        
        view = imageui.DrawUIView(timeout=GUI_TIMEOUT) if _view is None else _view
        needs_view = await self.view_check_defer(ctx, view)
        
        # this may be passed a url string in drawUI config
        image_url = imgfile.url if isinstance(imgfile, discord.Attachment) else imgurl
        image_url = imgutil.clean_image_url(image_url, check_tenor=False)
        image, imgmeta = await imgutil.aload_image(image_url, result_type='np')
        image = imgutil.image_fix(image, animated=False, transparency=False)
        
        image = Image.fromarray(image)
        

        try:
            prompt = await self.validate_prompt(ctx, prompt)
        except PromptCanceled:
            return await ctx.send('Canceled', silent=True, delete_after=1)
        

        async with self.bot.busy_status(activity='draw'):
            self.igen.dc_fastmode(enable=fast)
            output = await self.igen.regenerate_image(prompt=prompt, image=image, n_images=n_images, steps=steps, strength=strength, 
                                                                  negative_prompt=negative_prompt, guidance_scale=guidance_scale, detail_weight=detail_weight, 
                                                                  aspect=aspect, refine_strength=refine_strength, seed=seed)
            if needs_view:
                msg = await view.send(ctx, n_init_images = n_images)
                self.msg_views[msg.id] = view
            
            async for imbatch,kwbatch in async_gen(output):
                for kws in kwbatch:
                    kws.pop('image')
                    kws.update(imgurl=image_url)
                await view.add_images(imbatch, call_kwargs=kwbatch)

    
    @commands.hybrid_command(name='hd')
    #@check_up('igen', '❗ Drawing model not loaded. Call `!imgup`')
    async def _hd_upsample(self, ctx: commands.Context, imgurl:str, prompt: commands.Range[str,None,1000] = None, imgfile: discord.Attachment=None, *, flags: cmd_flags.UpsampleFlags, ):
        """Make an image HD (big n' smooooth).
        
        Args:
            imgurl: Url of image. Will be ignored if imgfile is used. 
            prompt: A description of the image. Not required, but if high hdstrength helps A LOT.
            imgfile: image attachment. To use, write anything in `imgurl` and drag+drop image. 
            hdstrength: HD steps strength. 0=Alter Nothing. 100=Alter Everything. Default=30
            steps: Num of iters to run. Increase = ⬆Quality, ⬆Run Time. Default varies.
            
            negprompt: What to exclude from image. Usually comma sep list of words. Default=None.
            guidance: Guidance scale. Increase = ⬆Prompt Adherence, ⬇Quality, ⬇Creativity. Default varies.
            detail: Detail weight. Value -3.0 to 3.0, >0 = add detail, <0 = remove detail. Default=0.
            seed: Random Seed. An arbitrary number to make results reproducible. Default=None.
        """
        return await self.hd_upsample(ctx, imgurl, prompt, imgfile=imgfile, 
                                 steps = flags.steps, 
                                 refine_strength = flags.hdstrength,
                                 
                                 negative_prompt = flags.negprompt, 
                                 guidance_scale = flags.guidance, 
                                 detail_weight = flags.detail,
                                 seed = flags.seed,
                                 _needs_view=True, # if calling as a command, need "view" (aka send img), if calling from UI, no view
                                 )

    async def hd_upsample(self, ctx: commands.Context, imgurl:str, prompt: str = None, imgfile: discord.Attachment|None=None, *,
                          refine_strength:float = 0.3, 
                          steps:int = None, 
                          negative_prompt: str = None, 
                          guidance_scale: float = None,
                          detail_weight: float = 0,
                          seed: int = None,
                          _needs_view=False,
                          ):
        refine_strength = cmd_tfms.percent_transform(refine_strength)
        
        _ = await self.view_check_defer(ctx)
        needs_view=_needs_view
        
        if isinstance(imgfile, Image.Image):
            image = imgfile # get a little cheeky here to bypass the download
        else:
            image_url = imgfile.url if isinstance(imgfile,discord.Attachment) else imgurl
            image_url = imgutil.clean_image_url(image_url, check_tenor=False)
            image, imgmeta = await imgutil.aload_image(image_url)
        
        try:
            prompt = await self.validate_prompt(ctx, prompt)
        except PromptCanceled:
            return await ctx.send('Canceled', silent=True, delete_after=1)


        async with self.bot.busy_status(activity='draw'):
            image, call_kwargs = await self.igen.refine_image(image=image, prompt=prompt, 
                                                           refine_strength=refine_strength, steps=steps,
                                                           negative_prompt=negative_prompt, guidance_scale=guidance_scale,
                                                           detail_weight=detail_weight, seed=seed, )
            if needs_view:
                msg = await ctx.send(file=imgutil.to_bytes_file(image, prompt=prompt, ext='WebP', lossless=True))
                out_imgpath = imgutil.save_image_prompt(image, prompt, ext='WebP', lossless=True)

        return image

    @commands.hybrid_command(name='animate')
    #@check_up('igen', '❗ Drawing model not loaded. Call `!imgup`')
    #@commands.max_concurrency(1, wait=True)
    async def _animate(self, ctx: commands.Context, prompt: str, imgurl:str=None, *, flags: cmd_flags.AnimateFlags):
        """Create a gif from a text prompt and (optionally) a starting image

        Args:
            prompt: A description of the gif to be animated.
            imgurl: Url of image. If set, strengths are linearly scaled from min to max.

            nframes: Number of frames to generate. Default=11.
            
            steps: Num of iters to run per frame. Increase = ⬆Quality, ⬆Run Time. Default varies.
            strength_end: Strength range end. 0 = No Change. 100=Change All. Default=80.
            strength_start: Strength range start. ignored if `imgurl` is None. Default=30.
            
            negprompt: What to exclude from image. Usually comma sep list of words. Default=None.
            guidance: Guidance scale. Increase = ⬆Prompt Adherence, ⬇Quality, ⬇Creativity. Default varies.
            detail: Detail weight. Value -3.0 to 3.0, >0 = add detail, <0 = remove detail. Default=0.
            midframes: The number of frames to add between each pair of images. Default=4.
            aspect: Image aspect ratio (shape). If None, will pick nearest to img if provided.

            fast: Trades image quality for speed - about 2-3x faster. Default=False (Turbo ignores).
            seed: Random Seed. An arbitrary number to make results reproducable. Default=None.
        """
        return await self.animate(ctx, prompt, imgurl=imgurl,
                                  nframes=flags.nframes,
                                  steps = flags.steps, 
                                  strength_end = flags.strength_end, 
                                  strength_start = flags.strength_start, 
                                  negative_prompt = flags.negprompt, 
                                  guidance_scale = flags.guidance, 
                                  detail_weight = flags.detail,
                                  mid_frames = flags.midframes,
                                  aspect = flags.aspect, 
                                  fast = flags.fast,
                                  seed = flags.seed)
    
        
    async def animate(self, ctx: commands.Context, prompt: str, imgurl:str=None, *, 
                      nframes: int = 11,
                      steps: int = None, 
                      strength_end: float = 0.80, 
                      strength_start: float = 0.30, 
                      negative_prompt: str = None, 
                      guidance_scale: float = None, 
                      detail_weight: float = 0.,
                      mid_frames: int = 4,
                      aspect: typing.Literal['square', 'portrait', 'landscape'] = None,
                      fast: bool = False,
                      seed: int = None
                     ):
        strength_end = cmd_tfms.percent_transform(strength_end)
        strength_start = cmd_tfms.percent_transform(strength_start)
        
        if self.igen.config.strength == 0:
            return await ctx.send(f'{self.igen.model_name} does not support `/animate`', ephemeral=True)
        
        needs_view = await self.view_check_defer(ctx)

        image = None
        if imgurl is not None:
            image_url = imgutil.clean_image_url(imgurl, check_tenor=False)
            image, imgmeta = await imgutil.aload_image(image_url, result_type=None)
        
        try:
            prompt = await self.validate_prompt(ctx, prompt)
        except PromptCanceled:
            return await ctx.send('Canceled', silent=True, delete_after=1)
        
        
        async with self.bot.busy_status(activity='draw'):
            self.igen.dc_fastmode(enable=fast, img2img=True)

            #image_frames, fwkg
            frame_gen = await self.igen.generate_frames(prompt=prompt, image=image, nframes=nframes, steps=steps, 
                                                    strength_end=strength_end,strength_start=strength_start, negative_prompt=negative_prompt, 
                                                    guidance_scale=guidance_scale, detail_weight=detail_weight, aspect=aspect, mid_frames=mid_frames,
                                                    seed=seed)
            
            cf = 0
            msg  = await ctx.send(f'Cooking... {cf}/{nframes}', silent=True)

            async for image in async_gen(frame_gen):
                if isinstance(image, list):
                    image_frames = image
                else:
                    cf += 1
                    msg  = await msg.edit(content=f'Cooking... {cf}/{nframes}')
                    
            msg  = await msg.edit(content=f"Seasoning...")
            
            out_imgpath = imgutil.save_animation_prompt(image_frames, prompt, ext='MP4', optimize=False)
           
            if needs_view:
                view = imageui.GifUIView(image_frames, timeout=GUI_TIMEOUT)
                msg = await view.send(ctx, msg, out_imgpath, prompt)
                self.msg_views[msg.id] = view
            
        #out_imgpath = save_image_prompt(image, prompt)
        return image_frames, out_imgpath

    @commands.hybrid_command(name='reanimate')
    #@check_up('igen', '❗ Drawing model not loaded. Call `!imgup`')
    async def _reanimate(self, ctx: commands.Context, prompt: commands.Range[str,1,1000], imgurl:str, *, flags: cmd_flags.ReanimateFlags):
        """Redraw the frames of an animated image.

        Args:
            prompt: A description of what you want to infuse the animation with.
            imgurl: Url of animated image. Can be .gif, .mp4, .webm, and some others.
            
            steps: Num of iters to run. Increase = ⬆Quality, ⬆Run Time. Default varies.
            astrength: Animation Strength. Frame alteration intensity. 0 = No change. 100=Complete Change. Default=50.
            imsize: How big gif should be. smaller = fast, full = slow, higher quality. Default=small. 
            
            negprompt: What to exclude from image. Usually comma sep list of words. Default=None.
            guidance: Guidance scale. Increase = ⬆Prompt Adherence, ⬇Quality, ⬇Creativity. Default varies.
            detail: Detail weight. Value -3.0 to 3.0, >0 = add detail, <0 = remove detail. Default=0.
            
            stage2: If True, run it twice to refine the output result. Default=True. 
            fast: Trades image quality for speed - about 2-3x faster. Default=False (Turbo ignores).
            
            aseed: Animation Seed. Improves frame cohesion. Unlike `seed`, always auto-set. Set -1 to disable. 
        """
        return await self.reanimate(ctx, prompt, imgurl, 
                                    steps = flags.steps, 
                                    astrength = flags.astrength, 
                                    imsize = flags.imsize, 
                                    negative_prompt = flags.negprompt, 
                                    guidance_scale = flags.guidance, 
                                    detail_weight= flags.detail,
                                    #aspect = flags.aspect, 
                                    two_stage = flags.stage2,
                                    fast = flags.fast,
                                    aseed = flags.aseed
                                 )
    
    async def reanimate(self, ctx: commands.Context, prompt: str, imgurl:str, *, 
                        steps: int = None, 
                        astrength: float = 0.50, 
                        imsize: typing.Literal['small','med','full']='small',
                        negative_prompt: str = None, 
                        guidance_scale: float = None, 
                        detail_weight: float = 0.,
                        two_stage: bool = False,
                        fast: bool = False,
                        aseed: int = None
                        ):
        astrength = cmd_tfms.percent_transform(astrength)
        if self.igen.config.strength == 0:
            return await ctx.send(f'{self.igen.model_name} does not support `/reanimate`', ephemeral=True)
        # this may be passed a url string in drawUI config
        try:
            image_url = imgutil.clean_image_url(imgurl, check_tenor=True)
        except ValueError:
            return await ctx.send('Looks like your using a tenor link. You need to click the gif in Discord to open the pop-up view then "Copy Link" to get the `.mp4` link', ephemeral=True)
        
        
        gif_array, imgmeta = await imgutil.aload_image(image_url, result_type='np') #iio.imread(image_url)
        
        if gif_array.ndim < 4:
            return await ctx.send('You passed a non-animated image url. Did you mean to call `/animate`?', ephemeral=True)
        
        needs_view = await self.view_check_defer(ctx) # This needs to be AFTER the checks or message will not be ephemeral because of ctx.defer()
        
        try:
            prompt = await self.validate_prompt(ctx, prompt)
        except PromptCanceled:
            return await ctx.send('Canceled', silent=True, delete_after=1)
                
        image_frames = []
        nf = None
        cf = 0
        async with self.bot.busy_status(activity='draw'):
            self.igen.dc_fastmode(enable=fast, img2img=True)
            #image_frames, fwkg
            frame_gen = await self.igen.regenerate_frames(frame_array=gif_array, prompt=prompt, imsize=imsize, steps=steps, 
                                                           astrength=astrength, negative_prompt=negative_prompt, 
                                                           guidance_scale=guidance_scale, detail_weight=detail_weight, two_stage=two_stage, aseed=aseed)
            # first item yielded is total number of frames
            nf = next(frame_gen)
            
            msg = None
            msg_template = 'Cooking... {cf}/{nf}' if not two_stage else 'Cooking...\nStage 1: {cf}/{nf}'
            
            async for frames in async_gen(frame_gen):                
                if msg is None:
                    msg = await ctx.send(msg_template.format(cf=cf, nf=nf), silent=True)
                
                if isinstance(frames, list):
                    image_frames = frames
                else:
                    cf += frames
                    stage,nth = divmod(cf, nf)
                    if two_stage and stage == 1 and nth == 0: # only want to update once
                        msg_template = ('Cooking...\n' + 'Stage 1: Done.\n' + 'Stage 2: {nth}/{nf}')
                    
                    if stage < 2: # prevent last update being 0/nf
                        msg  = await msg.edit(content=msg_template.format(cf=cf, nf=nf, nth=nth))
                    
            
            msg  = await msg.edit(content=f'Seasoning...')
            out_imgpath = imgutil.save_animation_prompt(image_frames, prompt, ext='MP4', optimize=False, imgmeta=imgmeta)


            if needs_view:
                view = imageui.GifUIView(image_frames, timeout=GUI_TIMEOUT)
                msg = await view.send(ctx, msg, out_imgpath, prompt)
                self.msg_views[msg.id] = view
            
        #out_imgpath = save_image_prompt(image, prompt)
        return image_frames, out_imgpath
    
    
    

async def setup(bot: BotUs):
    igen = ImageGen(bot)

    await bot.add_cog(igen)


