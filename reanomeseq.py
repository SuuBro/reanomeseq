#! /usr/bin/env python3
#
# Test for a push-button monome arc

import asyncio
import monome
import reapy
import numpy as np

from reapy import reascript_api as RPR
from typing import List, Tuple

from arc import Arc

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
        self.note_lookup = np.full((16,8), -1, dtype=int)
        self.last_downpress_by_row = np.full((8), -1, dtype=int)

    def on_grid_ready(self):
        print('Grid ready.')
        self.grid.led_all(0)
        asyncio.create_task(self.reaper_loop())


    def on_grid_disconnect(self):
        print('Grid disconnected.')


    def draw_note(self, start: int, end: int, row: int):
        self.grid.led_level_set(start, 7-row, 15)
        for x in range(start+1, end):
            self.grid.led_level_set(x, 7-row, 4)


    def on_grid_key(self, x: int, y: int, s: int):
        last_downpress = self.last_downpress_by_row[y].item()

        if s == 1 and last_downpress >= 0:
            start = min(x, last_downpress)
            end = max(x, last_downpress)+1
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


    def render_notes(self, bpm: float, notes: List[Note]):
        self.grid.led_all(0)
        self.note_lookup = np.full((16,8), -1, dtype=int)

        for note in notes:
            note_start_col = max(int(note.start / (bpm * self.zoom)), 0)
            note_end_col = min(int(note.end / (bpm * self.zoom)), 16)
            note_row = note.pitch - self.lowest_displayed_note

            if note_end_col < 0 or note_start_col > 15 or note_row < 0 or note_row > 7:
                continue # before display

            for x in range(note_start_col,note_end_col):
                self.note_lookup[x,note_row] = note.index

            self.draw_note(note_start_col, note_end_col, note_row)


    async def reaper_loop(self):
        previous_hash = ""

        while True:
            track = RPR.GetSelectedTrack(0, 0)
            _, _, _, hash, _ = RPR.MIDI_GetTrackHash(track, True, "", 100)

            if hash == previous_hash:
                await asyncio.sleep(0.1)
                continue

            bpm, notes = self.get_notes(track)
            self.render_notes(bpm, notes)

            previous_hash = hash
            await asyncio.sleep(0.05)

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

async def main():
    loop = asyncio.get_running_loop()
    arc = Arc()
    grid = GridApp()

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
