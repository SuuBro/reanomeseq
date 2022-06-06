#! /usr/bin/env python3
#
# Test for a push-button monome arc

import asyncio
import monome
import reapy

from reapy import reascript_api as RPR

from arc import Arc
from grid import Grid

@reapy.inside_reaper()
async def reaper_loop():
    while True:
        print("Syncing state from Reaper...")

        track = RPR.GetSelectedTrack(0, 0)
        num_media_items = RPR.GetTrackNumMediaItems(track)
        for media_item_idx in range(0, num_media_items):
            media_item = RPR.GetTrackMediaItem(track, media_item_idx)
            take = RPR.GetTake(media_item, 0)
            _, _, n, _, _  = RPR.MIDI_CountEvts(take, 0, 0, 0)
            for i in range(0, n):
                _, take, i, _, _, start, end, chan, pitch, vel = RPR.MIDI_GetNote(take, i, False, False, 0, 1, 0, 0, 0)
                print(f'{chan} {pitch} {vel} {start}->{end}')

        await asyncio.sleep(4)


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

    asyncio.create_task(reaper_loop())

    await serialosc.connect()
    await loop.create_future()


if __name__ == '__main__':
    asyncio.run(main())
