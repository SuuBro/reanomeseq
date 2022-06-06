import asyncio
import monome
import random


class Grid(monome.GridApp):
    def __init__(self):
        super().__init__()

    def on_grid_ready(self):
        print('Grid ready.')
        self.grid.led_all(0)

        self.row_values = []
        row_value = 0
        for i in range(self.grid.height):
            self.row_values.append(int(round(row_value)))
            row_value += 100 / (self.grid.height - 1)

        self.values = [random.randint(0, 100) for f in range(self.grid.width)]
        self.faders = [asyncio.ensure_future(self.fade_to(f, 0)) for f in range(self.grid.width)]

    def on_grid_disconnect(self):
        print('Grid disconnected.')

    def on_grid_key(self, x, y, s):
        if s == 1:
            self.faders[x].cancel()
            self.faders[x] = asyncio.ensure_future(self.fade_to(x, self.row_to_value(y)))

    def value_to_row(self, value):
        return sorted([i for i in range(self.grid.height)], key=lambda i: abs(self.row_values[i] - value))[0]

    def row_to_value(self, row):
        return self.row_values[self.grid.height - 1 - row]

    async def fade_to(self, x, new_value):
        while self.values[x] != new_value:
            if self.values[x] < new_value:
                self.values[x] += 1
            else:
                self.values[x] -= 1
            col = [0 if c > self.value_to_row(self.values[x]) else 1 for c in range(self.grid.height)]
            col.reverse()
            self.grid.led_col(x, 0, col)
            await asyncio.sleep(1/100)