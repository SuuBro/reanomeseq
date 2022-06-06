from collections import deque

import monome

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