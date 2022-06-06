#! /usr/bin/env python3
#
# Test for a push-button monome arc

import asyncio
import monome
import mido
import random
import reapy

from collections import deque
from mido import Message


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


class Arc(monome.ArcApp):

    def __init__(self):
        super().__init__()
        self.value = [0, 64, 64, 64]
        self.maxVal = 127
        self.offset = 0

    def on_arc_ready(self):
        for n in range(0, 4):
            self.arc.ring_all(n, 0)
            self.on_arc_delta(n, 0)

    def on_arc_disconnect(self):
        print('Arc disconnected.')

    def set_value(self, ring, value):
        start = 5
        end = 59
        span = end-start
        val = (value / self.maxVal * span) + start

        print(f'Ring: {ring} Val: {val}')

        values = [0] * 64
        for i in range(start, int(val)+1):
            values[i] = 15

        if(val - int(val) >= 0.4):
            values[int(val)+1] = 7

        offset = deque(values)
        offset.rotate(32)
        self.arc.ring_map(ring, offset)

    def on_arc_delta(self, ring, delta):
        print(f'Ring: {ring} Delta: {delta}')

        change = delta if (delta > 2 or delta < -2) else delta/2

        self.value[ring] = min(max(self.value[ring] + change, 0), self.maxVal)
        val = self.value[ring]
        print(f'Ring: {ring} Value: {val}')

        self.set_value(ring, val)

    def on_arc_key(self, ring, s):
        print(f'Ring: {ring} Pressed: {s > 0}')


async def main():
    loop = asyncio.get_running_loop()
    arc = Arc()
    grid = Grid()

    def serialosc_device_added(id, type, port):
        print(f'connecting to {id} ({type}) on {port}')
        if 'arc' in type:
            asyncio.ensure_future(arc.arc.connect('127.0.0.1', port))
        else:
            asyncio.ensure_future(grid.grid.connect('127.0.0.1', port))

    serialosc = monome.SerialOsc()
    serialosc.device_added_event.add_handler(serialosc_device_added)

    await serialosc.connect()
    await loop.create_future()


if __name__ == '__main__':
    asyncio.run(main())
