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

GRID_HEIGHT = 8
GRID_WIDTH = 16

NOTE_START_BRIGHTNESS = 12
NOTE_BRIGHTNESS = 7
DIVIDER_BRIGHTNESS = 3
PLAY_POS_BRIGHTNESS_LOW =  10
PLAY_POS_BRIGHTNESS_HIGH =  15

ZOOM_LEVELS = [60, 60, 60, 120, 120, 120, 240, 240, 240, 480, 480, 480, 960, 960, 960]

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
        self.zoom_index = 10
        self.zoom = ZOOM_LEVELS[self.zoom_index]
        self.earliest_displayed_time = 0
        self.lowest_displayed_note = 48
        self.note_scale = None
        self.view = [[0]*GRID_HEIGHT for _ in range(GRID_WIDTH)]
        self.note_lookup = np.full((GRID_WIDTH,GRID_HEIGHT), -1, dtype=int)
        self.last_downpress_by_row = np.full((GRID_HEIGHT), -1, dtype=int)

    def on_grid_ready(self):
        print('Grid ready')
        self.grid.led_all(0)
        asyncio.create_task(self.reaper_loop())


    def on_grid_disconnect(self):
        print('Grid disconnected')


    def apply_vertical_scroll(self, delta: int):
        self.lowest_displayed_note = max(self.lowest_displayed_note+delta, 0)

    def apply_horizontal_scroll(self, delta: int):
        self.earliest_displayed_time = max(self.earliest_displayed_time+delta, 0)

    def apply_horizontal_zoom(self, delta: int):
        if delta == 0:
            return
        self.zoom_index = min(max(self.zoom_index - delta, 0), len(ZOOM_LEVELS)-1)
        self.zoom = ZOOM_LEVELS[self.zoom_index]


    def on_grid_key(self, x: int, y: int, s: int):
        last_downpress = self.last_downpress_by_row[y].item()

        if s == 1 and last_downpress >= 0:
            start = min(x, last_downpress)
            end = max(x, last_downpress)+1
            for i in range(start,end):
                existing_note_idx = self.note_lookup[i,7-y].item()
                if existing_note_idx >= 0: # clash with existing note
                    return
            self.create_note(start, end, y)
            self.last_downpress_by_row[y] = -1
        elif s == 0:
            if last_downpress == x: # same key released
                existing_note_idx = self.note_lookup[x,7-y].item()
                if existing_note_idx >= 0:
                    self.delete_note(existing_note_idx)
                else:
                    self.create_note(x, x+1, y)
            self.last_downpress_by_row[y] = -1
        elif s == 1:
            self.last_downpress_by_row[y] = x


    def draw_note(self, start: int, end: int, row: int, starts_before_view: bool):
        self.view[start][GRID_HEIGHT-1-row] = NOTE_BRIGHTNESS if starts_before_view else NOTE_START_BRIGHTNESS
        for x in range(start+1, end):
            self.view[x][GRID_HEIGHT-1-row] = NOTE_BRIGHTNESS

    def divider_brightness(self, position):
        if (position * self.zoom) % (240*8*8) == 0: return DIVIDER_BRIGHTNESS+2
        if (position * self.zoom) % (240*8*2) == 0: return DIVIDER_BRIGHTNESS
        return 0

    def update_view_with_notes(self, notes: List[Note]):
        self.view = [[self.divider_brightness(i+self.earliest_displayed_time) for i in range(GRID_WIDTH)]] * GRID_HEIGHT
        self.view = [[row[i] for row in self.view] for i in range(len(self.view[0]))]

        self.note_lookup = np.full((GRID_WIDTH,GRID_HEIGHT), -1, dtype=int)

        for note in notes:
            note_start = (note.start / self.zoom) - self.earliest_displayed_time
            note_start_col = int(max(note_start, 0))
            note_end = ((note.end + 1) / self.zoom) - self.earliest_displayed_time
            note_end_col = int(min(note_end, GRID_WIDTH))
            note_row = note.pitch - self.lowest_displayed_note

            #print(f'note.start: {note.start:.2f}  note.end: {note.end:.2f}  self.zoom: {self.zoom}  note_start: {note_start:.2f}   note_end: {note_end:.2f}')

            if note_end <= 0 or note_start > GRID_WIDTH-1 or note_row < 0 or note_row > GRID_HEIGHT-1:
                continue # outside display

            for x in range(note_start_col,note_end_col):
                self.note_lookup[x,note_row] = note.index

            self.draw_note(note_start_col, note_end_col, note_row, note_start < 0)

    def update_view_with_play_position(self):
        pos = RPR.GetPlayPosition() * 1920
        col = int(round((pos // self.zoom) - self.earliest_displayed_time))
        if col < 0 or col > GRID_WIDTH-1:
            return # outside display

        for y in range(0, GRID_HEIGHT):
            self.view[col][y] = PLAY_POS_BRIGHTNESS_HIGH if self.view[col][y] >= NOTE_BRIGHTNESS else PLAY_POS_BRIGHTNESS_LOW


    def render(self):
        self.grid.led_level_map(0, 0, [[row[i] for row in self.view[:8]] for i in range(len(self.view[:8][0]))])
        self.grid.led_level_map(8, 0, [[row[i] for row in self.view[8:]] for i in range(len(self.view[8:][0]))])


    async def reaper_loop(self):
        previous_hash = ""
        notes = []

        while True:
            track = RPR.GetSelectedTrack(0, 0)

            _, _, _, hash, _ = RPR.MIDI_GetTrackHash(track, True, "", 100)
            if hash != previous_hash:
                notes = self.get_notes(track)

            self.update_view_with_notes(notes)
            self.update_view_with_play_position()
            self.render()

            previous_hash = hash
            await asyncio.sleep(0.05)

    @reapy.inside_reaper()
    def create_note(self, start: float, end: float, row: int):
        track = RPR.GetSelectedTrack(0, 0)
        media_item = RPR.GetTrackMediaItem(track, 0)
        take = RPR.GetTake(media_item, 0)

        startppqpos = (start+self.earliest_displayed_time) * self.zoom
        endppqpos = (end+self.earliest_displayed_time) * self.zoom
        pitch = (GRID_HEIGHT-1-row) + self.lowest_displayed_note

        RPR.MIDI_InsertNote(take, False, False, startppqpos, endppqpos-1, 0, pitch, 96, False)

    @reapy.inside_reaper()
    def delete_note(self, note_index: int):
        track = RPR.GetSelectedTrack(0, 0)
        media_item = RPR.GetTrackMediaItem(track, 0)
        take = RPR.GetTake(media_item, 0)
        RPR.MIDI_DeleteNote(take, note_index)

    @reapy.inside_reaper()
    def get_notes(self, track: any) -> Tuple[int, List[Note]]:
        media_item = RPR.GetTrackMediaItem(track, 0)
        take = RPR.GetTake(media_item, 0)
        _, _, n, _, _  = RPR.MIDI_CountEvts(take, 0, 0, 0)

        notes = []
        for i in range(0, n):
            _, take, i, _, _, start, end, chan, pitch, vel = RPR.MIDI_GetNote(take, i, False, False, 0, 1, 0, 0, 0)
            notes.append(Note(i, start, end, pitch, vel, chan))

        return notes


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

        if(ring == 1):
            self.grid.apply_vertical_scroll(change)

        if(ring == 2):
            self.grid.apply_horizontal_scroll(change)

        if(ring == 3):
            self.grid.apply_horizontal_zoom(change)


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
