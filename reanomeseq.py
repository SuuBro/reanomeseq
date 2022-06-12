#! /usr/bin/env python3
#
# Test for a push-button monome arc

import asyncio
import datetime
import monome
import reapy
import numpy as np
import scales
import signal
import sys
import time

from collections import deque
from datetime import datetime, timedelta
from display import *
from reapy import reascript_api as RPR
from scales import names_from_interval
from typing import List, Tuple

REANOMESEQ = 'reanomeseq'

GRID_HEIGHT = 8
GRID_WIDTH = 16

NOTE_START_BRIGHTNESS = 12
NOTE_BRIGHTNESS = 7
DIVIDER_BRIGHTNESS = 3
PLAY_POS_BRIGHTNESS_LOW =  10
PLAY_POS_BRIGHTNESS_HIGH =  15

NUM_OCTAVES = 10
ZOOM_LEVELS = [60, 120, 240, 480, 960]
SCALES = ['chromatic', 'major', 'minor']

keep_going = True

def clamp(v, minv, maxv):
    return max(min(maxv, v), minv)

def closest_index(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


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
        signal.signal(signal.SIGINT, lambda _, __: self.shutdown())
        self.arc = None
        self.keep_going = True
        self.zoom_index = 2
        self.zoom = ZOOM_LEVELS[self.zoom_index]
        self.earliest_displayed_time = 0
        self.lowest_displayed_pitch_index = 45
        self.held_note = -1
        self.held_down_time = datetime.utcnow()
        self.selected_scale_note = 'C'
        self.scale_select = False
        self.selected_scale_index = 0
        self.available_pitches = range(128)
        self.view = [[0]*GRID_HEIGHT for _ in range(GRID_WIDTH)]
        self.note_lookup = np.full((GRID_WIDTH,GRID_HEIGHT), -1, dtype=int)
        self.last_downpress_by_row = np.full((GRID_HEIGHT), -1, dtype=int)
        self.set_scale(0)

    def on_grid_ready(self):
        print('Grid ready')
        self.grid.led_all(0)
        asyncio.create_task(self.reaper_loop())

    def on_grid_disconnect(self):
        print('Grid disconnected')

    def set_scale(self, scale_index: int):
        self.scale_select = True
        if self.held_note >= 0:
            self.selected_scale_note = names_from_interval[self.held_note % 12]

        self.selected_scale_index = scale_index
        if SCALES[self.selected_scale_index] == 'chromatic':
            pitches = range(128)
        else:
            semitones = scales.scale(self.selected_scale_note, SCALES[self.selected_scale_index])
            pitches = []
            for octave in range(NUM_OCTAVES):
                for semitone in semitones:
                    pitches.append((octave * 12) + semitone.semitones_above_middle_c)

        tracking_note = self.held_note if self.held_note >= 0 else self.y_to_pitch(4)
        orig_pos = closest_index(tracking_note, self.available_pitches) - self.lowest_displayed_pitch_index

        self.available_pitches = pitches
        self.last_downpress_by_row = np.full((GRID_HEIGHT), -1, dtype=int)

        # re-position the view so that changing scale is less disorientiing
        new_pos = closest_index(tracking_note, self.available_pitches) - self.lowest_displayed_pitch_index
        diff = orig_pos - new_pos
        self.lowest_displayed_pitch_index -= diff


    def apply_vertical_scroll(self, delta: int):
        before = self.lowest_displayed_pitch_index
        self.lowest_displayed_pitch_index = clamp(self.lowest_displayed_pitch_index+delta, 0, len(self.available_pitches)-GRID_HEIGHT)
        return self.lowest_displayed_pitch_index - before

    def apply_horizontal_scroll(self, delta: int):
        before = self.earliest_displayed_time
        self.earliest_displayed_time = max(self.earliest_displayed_time+delta, 0)
        return self.earliest_displayed_time - before

    def apply_horizontal_zoom(self, zoom_index: int):
        self.set_zoom_index(zoom_index)

    def y_to_pitch(self, y: int):
        return self.available_pitches[self.lowest_displayed_pitch_index + (GRID_HEIGHT-1-y)]

    def on_grid_key(self, x: int, y: int, s: int):
        if s == 1 and x == 0:
            self.held_note = self.y_to_pitch(y)
            print(f'{names_from_interval[self.held_note % 12]}{((self.held_note - self.held_note%12)//12)-1} ({self.held_note})')
        else:
            self.held_note = -1
            self.scale_select = False

        held_down_time = datetime.utcnow() - self.held_down_time
        last_downpress = self.last_downpress_by_row[y].item()

        if s == 1 and last_downpress >= 0 and last_downpress != x:
            start = min(x, last_downpress)
            end = max(x, last_downpress)+1
            for i in range(start,end):
                existing_note_idx = self.note_lookup[i,7-y].item()
                if existing_note_idx >= 0: # clash with existing note
                    return
            self.create_note(start, end, y)
            self.last_downpress_by_row[y] = -1
        elif s == 0 and last_downpress >= 0 and held_down_time < timedelta(seconds=1):
            if last_downpress == x: # same key released
                existing_note_idx = self.note_lookup[x,7-y].item()
                if existing_note_idx >= 0:
                    self.delete_note(existing_note_idx)
                else:
                    self.create_note(x, x+1, y)
            self.last_downpress_by_row[y] = -1
        elif s == 0 and last_downpress >= 0:
            self.last_downpress_by_row[y] = -1
        elif s == 1:
            self.last_downpress_by_row[y] = x
            self.held_down_time = datetime.utcnow()


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
        self.view = [[row[i] for row in self.view] for i in range(len(self.view[0]))] # Rotate

        semitones = scales.interval_from_names[self.selected_scale_note]
        for i in range(len(self.view)):
            for j in range(len(self.view[i])):
                if self.y_to_pitch(j) % 12 == semitones:
                     self.view[i][j] += 2

        self.note_lookup = np.full((GRID_WIDTH,GRID_HEIGHT), -1, dtype=int)

        for note in notes:
            note_start = (note.start / self.zoom) - self.earliest_displayed_time
            note_start_col = int(max(note_start, 0))
            note_end = ((note.end + 1) / self.zoom) - self.earliest_displayed_time
            note_end_col = int(min(note_end, GRID_WIDTH))
            try:
                note_index = self.available_pitches.index(note.pitch)
                note_row = note_index - self.lowest_displayed_pitch_index
            except ValueError:
                continue # not not displayed as not in scale

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
        map = np.array([[row[i] for row in self.view[:8]] for i in range(len(self.view[:8][0]))])

        if self.held_note >= 0:
            map[0:8, 4:8] = np.array(NOTE_DISPLAY[self.held_note % 12])

        self.grid.led_level_map(0, 0, map.tolist())

        if self.held_note >= 0:
            top_half = np.array(SHARP if IS_SHARP[self.held_note % 12] else EMPTY)
            bottom_half = np.array(SCALE_MAPS[SCALES[self.selected_scale_index]] if self.scale_select else EMPTY)
            self.grid.led_level_map(8, 0, np.append(top_half, bottom_half, axis=0).tolist())
        else:
            self.grid.led_level_map(8, 0, [[row[i] for row in self.view[8:]] for i in range(len(self.view[8:][0]))])


    async def reaper_loop(self):
        previous_hash = ""
        previous_track = None
        notes = []

        while self.keep_going:
            track, hash, play_state = self.reaper_state()

            if track != previous_track:
                self.load_track_state(track)

            if hash != previous_hash:
                notes = self.get_notes(track)

            self.update_view_with_notes(notes)
            if play_state == 1:
                self.update_view_with_play_position()
            self.render()

            self.store_track_state(track)
            previous_hash = hash
            previous_track = track

            await asyncio.sleep(0.05)

        print("reaper_loop complete.")

    def restore_int(self, key, set):
        value = RPR.GetExtState(REANOMESEQ, key)
        if(value):
            set(int(value))

    def set_lowest_displayed_pitch_index(self, val: int):
        self.lowest_displayed_pitch_index = val

    def set_zoom_index(self, val: int):
        self.zoom_index = val
        self.zoom = ZOOM_LEVELS[self.zoom_index]

    @reapy.inside_reaper()
    def load_track_state(self, track):
        self.restore_int("lowpitch:"+track, self.set_lowest_displayed_pitch_index)
        self.restore_int("zoom:"+track, self.set_zoom_index)

    @reapy.inside_reaper()
    def store_track_state(self, track):
        RPR.SetExtState(REANOMESEQ, "lowpitch:"+track, str(self.lowest_displayed_pitch_index), True)
        RPR.SetExtState(REANOMESEQ, "zoom:"+track, str(self.zoom_index), True)

    @reapy.inside_reaper()
    def reaper_state(self):
        track = RPR.GetSelectedTrack(0, 0)
        _, _, _, hash, _ = RPR.MIDI_GetTrackHash(track, True, "", 100)
        play_state = RPR.GetPlayState()
        return track, hash, play_state

    @reapy.inside_reaper()
    def create_note(self, start: float, end: float, row: int):
        track = RPR.GetSelectedTrack(0, 0)
        media_item = RPR.GetTrackMediaItem(track, 0)
        take = RPR.GetTake(media_item, 0)

        startppqpos = (start+self.earliest_displayed_time) * self.zoom
        endppqpos = (end+self.earliest_displayed_time) * self.zoom
        pitch = self.y_to_pitch(row)

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

    def shutdown(self):
        print("Shutdown...")
        self.keep_going = False
        time.sleep(1)
        sys.exit(0)


class Arc(monome.ArcApp):

    def __init__(self, grid: GridApp):
        super().__init__()
        self.grid = grid
        self.options = [3, 0, 0, 5]
        self.max = [59, 64, 64, 49]
        self.value = [0, 0, 0, 0]

    def on_arc_ready(self):
        print('Arc ready')
        self.value[3] = self.grid.zoom_index

        self.set_option(0, self.value[0])
        self.spinner(1, self.value[1])
        self.spinner(2, self.value[2])
        self.set_option(3, self.value[3])

    def on_arc_disconnect(self):
        print('Arc disconnected')

    def spinner(self, ring, value):
        values = [0] * 64
        for i in range(0, 64):
            if (i - value) % 8 == 0:
                values[i] = 15
            if (i + 1 - value) % 8 == 0:
                values[i] = 12
            if (i + 2 - value) % 8 == 0:
                values[i] = 9
            if (i + 3 - value) % 8 == 0:
                values[i] = 6
            if (i + 4 - value) % 8 == 0:
                values[i] = 3

        offset = deque(values)
        offset.rotate(32)
        self.arc.ring_map(ring, offset)

    def set_option(self, ring, value):
        start = 5 if self.max[ring] == 3 else 5
        end = 59 if self.max[ring] == 3 else 60
        span = end-start
        option = int((value / (self.max[ring]+1)) * self.options[ring])
        leds_per_option = span // self.options[ring]
        first = start + (leds_per_option * option)
        last = first + leds_per_option

        values = [0] * 64
        for i in range(start, end):
            if i >= first and i < last:
                values[i] = 15

        offset = deque(values)
        offset.rotate(32)
        self.arc.ring_map(ring, offset)
        return option


    def on_arc_delta(self, ring, delta):

        if(ring == 0):
            self.value[ring] = clamp(self.value[ring] + delta, 0, self.max[ring])
            self.grid.set_scale(self.set_option(ring, self.value[ring]))

        if(ring == 1):
            change = self.grid.apply_vertical_scroll(delta)
            self.value[ring] = ((self.value[ring] + change) % self.max[ring])
            self.spinner(ring, self.value[ring])

        if(ring == 2):
            change = self.grid.apply_horizontal_scroll(delta)
            self.value[ring] = ((self.value[ring] + change) % self.max[ring])
            self.spinner(ring, self.value[ring])

        if(ring == 3):
            self.value[ring] = clamp(self.value[ring] + delta, 0, self.max[ring])
            self.grid.apply_horizontal_zoom(self.set_option(ring, self.value[ring]))


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
