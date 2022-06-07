#! /usr/bin/env python3
#
# Test for a push-button monome arc

import asyncio
import monome
import reapy

from reapy import reascript_api as RPR
from typing import List, Tuple

from arc import Arc
from grid import Grid

class Note():
    def __init__(self, start: int, end: int, pitch: int, velocity: int, channel: int):
        self.start = start
        self.end = end
        self.pitch = pitch
        self.velocity = velocity
        self.channel = channel

    def to_string(self):
        return f'{self.start:10.4f} -> {self.end:10.4f}: {self.pitch}'

class GridView:
    def __init__(self, grid: Grid):
        self.grid = grid
        self.zoom = 2
        self.earliest_displayed_time = 0
        self.lowest_displayed_note = 48
        self.note_scale = None

    def render_notes(self, bpm: float, notes: List[Note]):
        self.grid.clear()
        for note in notes:
            note_start_col = int(note.start / (bpm * self.zoom))
            note_end_col = int(note.end / (bpm * self.zoom))

            if(note_end_col < 0):
                continue # before display
            if(note_start_col > 15):
                continue # after display

            note_row = note.pitch - self.lowest_displayed_note
            self.grid.draw_note(note_start_col, note_end_col, note_row)


@reapy.inside_reaper()
def get_notes(track: any) -> Tuple[int, List[Note]]:
    media_item = RPR.GetTrackMediaItem(track, 0)
    project = RPR.GetItemProjectContext(media_item)
    _, bpm, _ = RPR.GetProjectTimeSignature2(project, 0, 0)
    take = RPR.GetTake(media_item, 0)
    _, _, n, _, _  = RPR.MIDI_CountEvts(take, 0, 0, 0)

    notes = []
    for i in range(0, n):
        _, take, i, _, _, start, end, chan, pitch, vel = RPR.MIDI_GetNote(take, i, False, False, 0, 1, 0, 0, 0)
        notes.append(Note(start, end, pitch, vel, chan))

    return bpm, notes


async def reaper_loop(gridView: GridView):
    previous_hash = ""

    while True:
        track = RPR.GetSelectedTrack(0, 0)
        _, _, _, hash, hs = RPR.MIDI_GetTrackHash(track, True, "", 100)

        if hash == previous_hash:
            await asyncio.sleep(0.1)
            continue

        bpm, notes = get_notes(track)
        gridView.render_notes(bpm, notes)

        previous_hash = hash
        await asyncio.sleep(0.05)


async def main():
    loop = asyncio.get_running_loop()
    arc = Arc()
    grid = Grid()
    gridView = GridView(grid)

    def serialosc_device_added(id, type, port):
        print(f'connecting to {id} ({type}) on {port}')
        if 'arc' in type:
            asyncio.ensure_future(arc.arc.connect('127.0.0.1', port))
        else:
            asyncio.ensure_future(grid.grid.connect('127.0.0.1', port))

    serialosc = monome.SerialOsc()
    serialosc.device_added_event.add_handler(serialosc_device_added)

    asyncio.create_task(reaper_loop(gridView))

    await serialosc.connect()
    await loop.create_future()


if __name__ == '__main__':
    asyncio.run(main())
