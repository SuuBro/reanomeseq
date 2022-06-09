#! /usr/bin/env python3
#
# Test for a push-button monome arc

import asyncio
import monome
import reapy
import numpy as np

from collections import deque
from reapy import reascript_api as RPR
from typing import List, Tuple

class Note():
    def __init__(self, index: int, start: int, end: int, pitch: int, velocity: int, channel: int):
        self.index = index
        self.start = start
        self.end = end
        self.pitch = pitch
        self.velocity = velocity
        self.channel = channel

    def to_string(self):
        return f'{self.start:10.4f} -> {self.end:10.4f}: {self.pitch}'


class VaribrightGrid(monome.Grid):
    def __init__(self):
        super().__init__()

    def _set_varibright(self):
        self.varibright = True


class GridApp(monome.GridApp):
    def __init__(self):
        super().__init__(VaribrightGrid())
        self.zoom = 2
        self.earliest_displayed_time = 0
        self.lowest_displayed_note = 48
        self.note_scale = None
        self.view = [[0]*8 for _ in range(16)]
        self.note_lookup = np.full((16,8), -1, dtype=int)
        self.last_downpress_by_row = np.full((8), -1, dtype=int)

    def on_grid_ready(self):
        print('Grid ready')
        self.grid.led_all(0)
        asyncio.create_task(self.reaper_loop())


    def on_grid_disconnect(self):
        print('Grid disconnected')


    def horizontal_scroll(self, delta: int):
        self.earliest_displayed_time = max(self.earliest_displayed_time+delta, 0)

    def horizontal_offset(self):
        return int(self.earliest_displayed_time / 4)

    def on_grid_key(self, x: int, y: int, s: int):
        last_downpress = self.last_downpress_by_row[y].item()

        if s == 1 and last_downpress >= 0:
            start = min(x, last_downpress)
            end = max(x, last_downpress)+1
            for i in range(start,end):
                existing_note_idx = self.note_lookup[i,7-y].item()
                if existing_note_idx >= 0: # clash with existing note
                    return
            self.create_note(start+self.horizontal_offset(), end+self.horizontal_offset(), y)
            self.last_downpress_by_row[y] = -1
        elif s == 0:
            if last_downpress == x: # same key released
                existing_note_idx = self.note_lookup[x,7-y].item()
                if existing_note_idx >= 0:
                    self.delete_note(existing_note_idx)
                else:
                    self.create_note(x+self.horizontal_offset(), x+self.horizontal_offset()+1, y)
            self.last_downpress_by_row[y] = -1
        elif s == 1:
            self.last_downpress_by_row[y] = x


    def draw_note(self, start: int, end: int, row: int, starts_before_view: bool):
        self.view[start][7-row] = 5 if starts_before_view else 15
        for x in range(start+1, end):
            self.view[x][7-row] = 5

    def render_notes(self, bpm: float, notes: List[Note]):
        offset = self.horizontal_offset()
        self.note_lookup = np.full((16,8), -1, dtype=int)
        self.view = [[1 if (i+offset)%4 == 0 else 0 for i in range(16)]] * 8
        self.view = [[row[i] for row in self.view] for i in range(len(self.view[0]))]

        for note in notes:
            note_start = note.start / (bpm * self.zoom) - offset
            note_start_col = int(max(note_start, 0))
            note_end_col = int(min(note.end / (bpm * self.zoom) - offset, 16))
            note_row = note.pitch - self.lowest_displayed_note

            if note_end_col < 0 or note_start_col > 15 or note_row < 0 or note_row > 7:
                continue # before display

            for x in range(note_start_col,note_end_col):
                self.note_lookup[x,note_row] = note.index

            self.draw_note(note_start_col, note_end_col, note_row, note_start < 0)

        self.grid.led_level_map(0, 0, [[row[i] for row in self.view[:8]] for i in range(len(self.view[:8][0]))])
        self.grid.led_level_map(8, 0, [[row[i] for row in self.view[8:]] for i in range(len(self.view[8:][0]))])


    async def reaper_loop(self):
        previous_hash = ""
        previous_earliest_displayed_time = -1

        while True:
            track = RPR.GetSelectedTrack(0, 0)
            _, _, _, hash, _ = RPR.MIDI_GetTrackHash(track, True, "", 100)

            if hash == previous_hash and self.earliest_displayed_time == previous_earliest_displayed_time:
                await asyncio.sleep(0.1)
                continue

            bpm, notes = self.get_notes(track)
            self.render_notes(bpm, notes)

            previous_hash = hash
            previous_earliest_displayed_time = self.earliest_displayed_time
            await asyncio.sleep(0.1)

    @reapy.inside_reaper()
    def create_note(self, start: float, end: float, row: int):
        track = RPR.GetSelectedTrack(0, 0)
        media_item = RPR.GetTrackMediaItem(track, 0)
        project = RPR.GetItemProjectContext(media_item)
        _, bpm, _ = RPR.GetProjectTimeSignature2(project, 0, 0)
        take = RPR.GetTake(media_item, 0)

        startppqpos = self.earliest_displayed_time + (bpm * self.zoom * start)
        endppqpos = self.earliest_displayed_time + (bpm * self.zoom * end)
        pitch = (7-row) + self.lowest_displayed_note

        RPR.MIDI_InsertNote(take, False, False, startppqpos, endppqpos, 0, pitch, 96, False)

    @reapy.inside_reaper()
    def delete_note(self, note_index: int):
        track = RPR.GetSelectedTrack(0, 0)
        media_item = RPR.GetTrackMediaItem(track, 0)
        take = RPR.GetTake(media_item, 0)
        RPR.MIDI_DeleteNote(take, note_index)

    @reapy.inside_reaper()
    def get_notes(self, track: any) -> Tuple[int, List[Note]]:
        media_item = RPR.GetTrackMediaItem(track, 0)
        project = RPR.GetItemProjectContext(media_item)
        _, bpm, _ = RPR.GetProjectTimeSignature2(project, 0, 0)
        take = RPR.GetTake(media_item, 0)
        _, _, n, _, _  = RPR.MIDI_CountEvts(take, 0, 0, 0)

        notes = []
        for i in range(0, n):
            _, take, i, _, _, start, end, chan, pitch, vel = RPR.MIDI_GetNote(take, i, False, False, 0, 1, 0, 0, 0)
            notes.append(Note(i, start, end, pitch, vel, chan))

        return bpm, notes


class Arc(monome.ArcApp):

    def __init__(self, grid: GridApp):
        super().__init__()
        self.grid = grid
        self.value = [0, 64, 64, 64]
        self.maxVal = 127
        self.offset = 0


    def on_arc_ready(self):
        print('Arc ready')
        for n in range(0, 4):
            self.arc.ring_all(n, 0)
            self.on_arc_delta(n, 0)


    def on_arc_disconnect(self):
        print('Arc disconnected')


    def set_value(self, ring, value):
        start = 5
        end = 59
        span = end-start
        val = (value / self.maxVal * span) + start

        values = [0] * 64
        for i in range(start, int(val)+1):
            values[i] = 15

        if(val - int(val) >= 0.4):
            values[int(val)+1] = 7

        offset = deque(values)
        offset.rotate(32)
        self.arc.ring_map(ring, offset)


    def on_arc_delta(self, ring, delta):
        change = delta # if (delta > 2 or delta < -2) else delta/2

        self.value[ring] = min(max(self.value[ring] + change, 0), self.maxVal)
        val = self.value[ring]

        self.set_value(ring, val)

        if(ring == 3):
            self.grid.horizontal_scroll(change)


    def on_arc_key(self, ring, s):
        print(f'Ring: {ring} Pressed: {s > 0}')


async def main():
    loop = asyncio.get_running_loop()
    grid = GridApp()
    arc = Arc(grid)

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
