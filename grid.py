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
        self.ready = False

    def on_grid_ready(self):
        print('Grid ready.')
        self.grid.led_all(0)
        self.ready = True

    def on_grid_disconnect(self):
        print('Grid disconnected.')

    def clear(self):
        self.grid.led_all(0)

    def draw_note(self, start: int, end: int, row: int):
        if not self.ready:
            return

        for x in range(start, end):
            self.grid.led_set(x, 7-row, 15)


    def on_grid_key(self, x: int, y: int, s: int):
        print(f'on_grid_key {x} {y} {s}')


