#!/usr/bin/env python3
"""Convert the pinned Rigid IPC Figure 2 fixture to the native frame-0 pose table.

The table keeps the authored model origin, SI scale, Euler angles in degrees,
and initial z angular velocity in degrees/second. It does not implement the
upstream motor or one-axis sprocket constraints. See source_manifest.json and
the three LICENSE files for source and asset licensing.
"""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def table():
    fixture = json.loads((ROOT / 'fixture.json').read_text())
    bodies = fixture['rigid_body_problem']['rigid_bodies']
    assert len(bodies) == 138
    lines = ['# Rigid IPC Figure 2 bike chain; 34 segments, 138 separate bodies.',
             '# mesh tx ty tz rx_deg ry_deg rz_deg uniform_scale omega_z_deg_s role']
    for body in bodies:
        name = Path(body['mesh']).name
        role = 'drive' if name == 'sprocket-20teeth.obj' else 'driven' if name == 'sprocket-8teeth.obj' else 'chain'
        omega = body.get('angular_velocity', [0, 0, 0])
        assert omega[:2] == [0, 0]
        values = body.get('position', [0, 0, 0]) + body.get('rotation', [0, 0, 0]) + [body['scale'], omega[2]]
        assert all(math.isfinite(v) for v in values) and values[-2] > 0
        lines.append(name + ' ' + ' '.join(format(v, '.17g') for v in values) + ' ' + role)
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--check', action='store_true')
    args = parser.parse_args()
    text = table()
    path = ROOT / 'initial_bodies.txt'
    if args.write:
        path.write_text(text)
    elif path.read_text() != text:
        raise SystemExit('initial_bodies.txt differs from the source fixture')
    print('initial_bodies.txt: 138 source body poses verified')


if __name__ == '__main__':
    main()
