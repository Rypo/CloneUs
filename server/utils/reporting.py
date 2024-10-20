import typing

import torch


class StatusItem(typing.NamedTuple):
    label: str
    value: str
    extra: str = ''
    advanced:bool = False

    def md(self, fmt:str=None):
        # '{label}: **{value}**{extra}'
        if fmt is None:
            labelfmt,valuefmt,extrafmt = '','',''
            if self.label not in ['',None]:
                labelfmt = '{label}: '
            if self.value not in ['',None]:
                valuefmt = '**{value}**'
            if self.extra is not None:
                extrafmt='{extra}'

            fmt = labelfmt+valuefmt+extrafmt
        # if self.value == '' or self.value is None: # want to keep some falsey values
        #     fmt='{label}:{extra}' if self.extra else '{label}'
            
        # elif self.label == '' or self.label is None:
        #     fmt = '**{value}**{extra}'

        return fmt.format_map(self._asdict())
    

def vram_usage(device:torch.device = None):
    try:
        memfree,memtotal = torch.cuda.mem_get_info(device)
        vram_total = round(memtotal / (1024**2))
        vram_used = round((memtotal-memfree) / (1024**2))
        
        return vram_used, vram_total
    except (IndexError, ValueError) as e:
        print(e)
        return 0,0