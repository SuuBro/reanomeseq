import asyncio
import monome

class VaribrightGrid(monome.Grid):
    def __init__(self):
        super().__init__()

    def _set_varibright(self):
        self.varibright = True

class Grid(monome.GridApp):
    def __init__(self):
        super().__init__(VaribrightGrid())

    def on_grid_ready(self):
        print('Grid ready.')
        for x in range(16):
            for y in range(8):
                self.grid.led_level_set(x, y, x)

    def on_grid_disconnect(self):
        print('Grid disconnected.')

    def on_grid_key(self, x, y, s):
        print(f'on_grid_key {x} {y} {s}')
