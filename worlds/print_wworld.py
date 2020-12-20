#!/usr/bin/env python

import argparse
import json

from wumpus import WumpusWorld


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infiles', type=argparse.FileType('r'), nargs='+', help='Wumpus world JSON files')
    args = parser.parse_args()

    for fd in args.infiles:
        try:
            world = WumpusWorld.from_JSON(json.load(fd))
            print(world)
        except Exception:
            pass


if __name__ == "__main__":
    main()
