import re
import math
import json
import datetime
import more_itertools

import discord
from discord.ext import commands

from cloneus.data import roles


class BaseButtonPageView(discord.ui.View):
    def __init__(self, data: list, *, timeout: float | None = 180):
        super().__init__(timeout=timeout)
        self.data = data
        self.num_pages = len(data)
        self.current_page: int = 1
        

    @property
    def current_page_data(self):
        return self.data[self.current_page-1]
    

    async def send(self, ctx):
        self.message = await ctx.send(view=self)
        await self.update_message(self.data[0])

    def create_embed(self, data):
        raise NotImplementedError

    async def update_message(self, data):
        # only enable buttons if more than one page
        if self.num_pages > 1:
            self.update_buttons()
        await self.message.edit(embed=self.create_embed(data), view=self)

    def disable_button(self, button_method):
        button_method.disabled = True
        button_method.style = discord.ButtonStyle.gray

    def enable_button(self, button_method, style=discord.ButtonStyle.primary):
        button_method.disabled = False
        button_method.style = style

    def update_buttons(self):
        if self.current_page == 1:
            self.disable_button(self.first_page_button)
            self.disable_button(self.prev_button)

            self.enable_button(self.next_button)
            self.enable_button(self.last_page_button, discord.ButtonStyle.green)


        elif self.current_page == self.num_pages:
            self.disable_button(self.next_button)
            self.disable_button(self.last_page_button)

            self.enable_button(self.prev_button)
            self.enable_button(self.first_page_button, discord.ButtonStyle.green)

        else:
            if any([self.first_page_button.disabled, self.next_button.disabled, self.prev_button.disabled, self.last_page_button.disabled]):
                self.enable_button(self.first_page_button, discord.ButtonStyle.green)
                self.enable_button(self.prev_button)
                self.enable_button(self.next_button)
                self.enable_button(self.last_page_button, discord.ButtonStyle.green)


    @discord.ui.button(label="|<", style=discord.ButtonStyle.green,  disabled=True)
    async def first_page_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        self.current_page = 1

        await self.update_message(self.current_page_data)

    @discord.ui.button(label="<", style=discord.ButtonStyle.primary, disabled=True)
    async def prev_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        self.current_page -= 1
        
        await self.update_message(self.current_page_data)

    @discord.ui.button(label=">", style=discord.ButtonStyle.primary,  disabled=True)
    async def next_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        self.current_page += 1
        
        await self.update_message(self.current_page_data)

    @discord.ui.button(label=">|", style=discord.ButtonStyle.green,  disabled=True)
    async def last_page_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        self.current_page = self.num_pages
        
        await self.update_message(self.current_page_data)



class PagedChangelogView(BaseButtonPageView): #discord.ui.View):
    #current_page: int = 1
    def __init__(self, changelogs: list, *, timeout: float | None = 180):
        #self.changelogs = changelogs
        self.num_pages = len(changelogs)
        data = self.parse_logs(changelogs)
        super().__init__(data, timeout=timeout)

    @property
    def current_page_data(self):
        return self.data[self.current_page-1]
    
    def parse_logs(self, changelogs):
        data = []
        for i,clog in enumerate(changelogs):
            ver = self.num_pages-i
            #clog = clogs[0]
            cloglines= clog.splitlines()
            clog = '\n'.join(cloglines[:-1])
            release_date = datetime.datetime.strptime(re.findall(r'\(([0-9-]+)\)',clog)[0],'%Y-%m-%d')
            
            metadata = json.loads(cloglines[-1].split('METADATA: ',1)[-1])

            metadata['changelog'] = clog
            metadata['release_date'] = release_date
            metadata.setdefault('version', f'v0.0.{ver}')
            
            data.append(metadata)

        return data

    async def send(self, ctx):
        self.message = await ctx.send(view=self)
        await self.update_message(self.data[0])

    def create_embed(self, data):
        botname = roles.USER_DATA['BOT']['displayName']
        embed = discord.Embed(
            color=discord.Color.dark_green(),
            title=f"{botname} ({data['version']})",
            url=data['url'],
            description=data['changelog'],
            timestamp=data['release_date'],
        )

        return embed


class PaginatorView(discord.ui.View):
    current_page: int = 1
    def __init__(self, paginator: commands.Paginator, *, timeout: float | None = 180):
        self.paginator = paginator
        self.total_pages = len(self.paginator.pages)
        super().__init__(timeout=timeout)

    @property
    def current_page_data(self):
        return self.paginator.pages[self.current_page]
    
    async def send(self, ctx):
        self.message = await ctx.send(view=self)
        await self.update_message(self.paginator.pages[0])

    def create_embed(self, data):
        embed = discord.Embed(title=f"Command Context {self.current_page} / {self.total_pages}")
        for item in data:
            embed.add_field(name=item['label'], value=item["item"], inline=False)
        return embed

    async def update_message(self, data):
        if self.total_pages > 1:
            self.update_buttons()
        await self.message.edit(embed=self.create_embed(data), view=self)

    def update_buttons(self):
        if self.current_page == 1:
            self.first_page_button.disabled = True
            self.prev_button.disabled = True
            self.first_page_button.style = discord.ButtonStyle.gray
            self.prev_button.style = discord.ButtonStyle.gray


        elif self.current_page == self.total_pages:
            self.next_button.disabled = True
            self.last_page_button.disabled = True
            self.last_page_button.style = discord.ButtonStyle.gray
            self.next_button.style = discord.ButtonStyle.gray
        else:
            #self.first_page_button.
            if any([self.first_page_button.disabled, self.next_button.disabled, self.prev_button.disabled, self.last_page_button.disabled]):
                self.first_page_button.disabled = False
                self.next_button.disabled = False
                self.prev_button.disabled = False
                self.last_page_button.disabled = False

                self.first_page_button.style = discord.ButtonStyle.green
                self.prev_button.style = discord.ButtonStyle.primary

                self.last_page_button.style = discord.ButtonStyle.green
                self.next_button.style = discord.ButtonStyle.primary




    @discord.ui.button(label="|<", style=discord.ButtonStyle.green, )
    async def first_page_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        self.current_page = 1

        await self.update_message(self.current_page_data)

    @discord.ui.button(label="<", style=discord.ButtonStyle.primary, )
    async def prev_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        self.current_page -= 1
        
        await self.update_message(self.current_page_data)

    @discord.ui.button(label=">", style=discord.ButtonStyle.primary, )
    async def next_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        self.current_page += 1
        
        await self.update_message(self.current_page_data)

    @discord.ui.button(label=">|", style=discord.ButtonStyle.green,)
    async def last_page_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        self.current_page = self.total_pages#math.ceil(len(self.data) / self.sep)#int(len(self.data) / self.sep) + 1
        
        await self.update_message(self.current_page_data)

class MessageContextView(BaseButtonPageView):
    def __init__(self, data, items_per_page=10, raw=False, *, timeout: float | None = 180):
        super().__init__(data, timeout=timeout)
        self.data = data
        self.num_items = len(data)
        self.items_per_page = items_per_page
        self.raw = raw
        self.batched_data = list(more_itertools.batched(data, items_per_page))
        self.num_pages = len(self.batched_data)
        self.current_page = self.num_pages
        

    async def send(self, ctx: commands.Context):
        if self.num_pages < 2:
            bdata = self.batched_data[-1] if self.batched_data else []
            self.message = await ctx.send(embed=self.create_embed(bdata), view=self, ephemeral=True)
            return 
        self.message = await ctx.send(view=self, ephemeral=True)
        await self.update_message(self.batched_data[-1])

    def create_embed(self, data):
        batch_start = max(self.current_page-1, 0)*self.items_per_page
        batch_end = batch_start+len(data)
        embed = discord.Embed(title=f"Message Context", color=discord.Color.blurple(),) # f"Message Context ({batch_start}:{batch_end} / {self.num_items})",
        for author, message in data: 
            # wmessage = '\n'.join([f'`{m}`' for m in message.splitlines()])
            if len(message) > 1000:
                msgparts = [' '.join(msgpart) for msgpart in list(more_itertools.constrained_batches(message.split(' '), max_size=1000, get_len=lambda g: len(' '.join(g))))]
                n_parts = len(msgparts)
                for i,msg in enumerate(msgparts,1):
                    wmsg = f"```{msg!r}```" if self.raw else msg
                    if i<n_parts:
                        wmsg += '...'
                    embed=embed.add_field(name=f'{author} ({i}/{n_parts})', value=wmsg, inline=False)
            else:
                wmessage = f"```{message!r}```" if self.raw else message
                embed.add_field(name=author, value=wmessage, inline=False)
        embed.set_footer(text='📄{:\t>32}'.format(f"({batch_start} : {batch_end} / {self.num_items})"))
        #embed.set_footer(text='📄{:>256}'.format(f"({batch_start} : {batch_end} / {self.num_items})"))
        return embed


    @property
    def current_page_data(self):
        return self.batched_data[self.current_page-1]


    @discord.ui.button(label="X", style=discord.ButtonStyle.danger, )
    async def delete_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        #self.current_page = self.num_pages#math.ceil(len(self.data) / self.sep)#int(len(self.data) / self.sep) + 1
        
        await self.message.delete()#(self.get_current_page_data())
        self.clear_items()
        

class PrePagedView(BaseButtonPageView):
    def __init__(self, data_pages: list[str,list[tuple[str,str]]], *, timeout: float | None = 180):
        super().__init__(data=data_pages, timeout=timeout)
        self.data = data_pages
        self.num_pages = len(self.data)
        self.current_page: int = 1
    
    @property
    def current_page_data(self):
        return self.data[self.current_page-1]
    

    async def send(self, ctx, content=None):
        self.message = await ctx.send(content=content, view=self)
        await self.update_message(self.data[0])


    def create_embed(self, data):
        title,items = data
        rawdesc = '\n'.join(['**{}**\n{}'.format(*item) for item in items])
        embed = discord.Embed(title=title, description=rawdesc)
        #for item in data:
        #    embed.add_field(name=item['label'], value=item["item"], inline=False)
        return embed

    @discord.ui.button(label="X", style=discord.ButtonStyle.danger, )
    async def delete_button(self, interaction:discord.Interaction, button: discord.ui.Button):
        await interaction.response.defer()
        #self.current_page = self.num_pages#math.ceil(len(self.data) / self.sep)#int(len(self.data) / self.sep) + 1
        
        await self.message.delete()
        self.clear_items()


#async def setup(bot: commands.Bot):
#    bot.add_view(MessageContextView())