#!/usr/bin/env python

import argparse
from contextlib import contextmanager
import copy
import datetime
import importlib
import io
import json
import os
from os import name
from pathlib import Path
import random
import sys
from typing import ClassVar, Dict, Iterable, Iterator, NamedTuple, Sequence, TextIO

from wumpus import WumpusWorld
from wumpus.gridworld import Actions, Agent, GridWorld, GridWorldException, Player, InformedPlayer, UninformedPlayer
from wumpus.runner import get_player_class, check_entrypoint, get_world_class


@contextmanager
def capture_stdout(dest: io.TextIOBase = None):
    new_target = open(os.devnull, "w") if dest is None else dest
    old_target = sys.stdout
    sys.stdout = new_target
    try:
        yield new_target
    finally:
        sys.stdout = old_target


class EpisodeResult(NamedTuple):
    player: str
    world: Dict
    outcome: int
    success: bool
    alive: bool
    cutoff: bool
    moves: Sequence[Actions]
    exception: Exception

    @classmethod
    def JSONEncoder(cls) -> ClassVar[json.JSONEncoder]:
        return cls._JSONEncoder

    class _JSONEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, NamedTuple):
                return o._asdict()
            # if isinstance(o, EpisodeResult):
            #     return {
            #         'player': o.player,
            #         'world': o.world,
            #         'outcome': o.outcome,
            #         'moves': [a.name for a in o.moves],
            #         'alive': o.alive,
            #         'exception': str(o.exception) if o.exception is not None else None
            #     }
            elif isinstance(o, Actions):
                return o.name
            elif isinstance(o, Exception):
                return repr(o)
            # Let the base class default method raise the TypeError
            return json.JSONEncoder.default(self, o)


def run_episode(world: GridWorld, player: Player, horizon: int = 0, show=True, outf: io.TextIOWrapper = sys.stdout) -> EpisodeResult:
    """Run an episode on the world using the player to control the agent. The horizon specifies the maximum number of steps, 0 or None means no limit. If show is true then the world is printed ad each iteration before the player's turn.

        Raise the exception GridWorldException is the agent is not in the world.

    Args:
        world (GridWorld): the world in which the episode is run
        player (Player): the player
        horizon (int, optional): stop after this number of steps, 0 for no limit. Defaults to 0.
        show (bool, optional): whether to show the environment before a step and player's output. Defaults to True.
        outf (io.TextIOWrapper, optional): writes output to the given stream. Defaults to sys.stdout.

    Raises:
        GridWorldException: [description]
    """

    def _say(text: str):
        if outf is not None:
            print(text, file=outf)

    agent = next(iter(o for o in world.objects if isinstance(o, Agent)), None)
    if agent is None:
        raise GridWorldException('Missing agent, cannot run the episode')

    world_desc = world.to_dict()
    poutf = outf if show else None
    moves = []

    cutoff = False

    # inform the player of the start of the episode
    try:
        with capture_stdout(dest=poutf):
            if isinstance(player, InformedPlayer):
                player.start_episode(copy.deepcopy(world))
            else:
                player.start_episode()
    except Exception as e:
        return EpisodeResult(
            player=player.name, world=world_desc, outcome=-1000,
            moves=moves, alive=agent.isAlive, success=agent.success(),
            exception=e, cutoff=cutoff
        )


    step = 0
    while not horizon or step < horizon:
        if agent.success():
            _say('The agent {} succeeded!'.format(agent.name))
            break
        if not agent.isAlive:
            _say('The agent {} died!'.format(agent.name))
            break

        if show or step < 1:
            _say(world)

        try:
            with capture_stdout(dest=poutf):
                action = player.play(step, agent.percept(), agent.actions())
                moves.append(action)
        except Exception as e:
            return EpisodeResult(
                player=player.name, world=world_desc, outcome=agent.reward - 1000,
                moves=moves, alive=agent.isAlive, success=agent.success(),
                exception=e, cutoff=cutoff
            )
        if action is None:
            _say('Episode terminated by the player {}.'.format(player.name))
            break
        reward = agent.do(action)
        _say('Step {}: agent {} executing {} -> reward {}'.format(step, agent.name, action.name, reward))

        try:
            with capture_stdout(dest=poutf):
                player.feedback(action, reward, agent.percept())
        except Exception as e:
            return EpisodeResult(
                player=player.name, world=world_desc, outcome=agent.reward - 1000,
                moves=moves, alive=agent.isAlive, success=agent.success(),
                exception=e, cutoff=cutoff
            )

        step += 1
    else:
        cutoff = True
        _say('Episode terminated by maximum number of steps ({}).'.format(horizon))

    try:
        with capture_stdout(dest=poutf):
            player.end_episode(agent.reward, agent.isAlive, agent.success)
    except Exception as e:
        return EpisodeResult(
            player=player.name, world=world_desc, outcome=agent.reward - 1000,
            moves=moves, alive=agent.isAlive, success=agent.success(),
            exception=e, cutoff=cutoff
        )

    _say(world)
    _say('Episode terminated with a reward of {} for agent {}'.format(agent.reward, agent.name))

    return EpisodeResult(
        player=player.name, world=world_desc, outcome=agent.reward,
        moves=moves, alive=agent.isAlive, success=agent.success(),
        exception=None, cutoff=cutoff
    )


def exercise_player(player: Player, episodes: Iterable[GridWorld], horizon: int = 1000) -> Sequence[EpisodeResult]:
    """Run a sequence of episodes for the given player returning the log of all episodes.
    
    Arguments:
        player {Player} -- the player
        episodes {Iterable[GridWorld]} -- sequence of worlds to play over
    
    Keyword Arguments:
        horizon {int} -- each episode is limited to a maximun number of steps (default: {1000})
    
    Returns:
        Sequence[EpisodeResult] -- a log results for each episode
    """
    return [run_episode(world, player, horizon=horizon, show=False, outf=None) for world in episodes]


def random_worlds(worlds: int, min_size: int = 4, max_size: int = 4, gw_class: ClassVar[GridWorld] = WumpusWorld) -> Iterator[GridWorld]:
    """Generates a sequence of random worlds to be used as an iterator.
    
    Arguments:
        worlds {int} -- number of worlds to be generated
    
    Keyword Arguments:
        min_size {int} -- minimum size of each world (default: {4})
        max_size {int} -- maximun size of each world (default: {8})
        gw_class {ClassVar[GridWorld]} -- the GridWorld subclass (default: {wumpus.WumpusWorld})

    Returns:
        Iterator[GridWorld] -- iterator over a sequence of Grid Worlds
    """
    for i in range(worlds):
        yield gw_class.random(size=random.randint(min_size, max_size))


def worlds_fromJSON(fp: TextIO, gw_class: ClassVar[GridWorld] = WumpusWorld) -> Iterator[GridWorld]:
    """Read a sequence of world descriptions from a JSON file.
    
    Arguments:
        fp {TextIO} -- JSON file containing a sequence of GridWorld description, see GridWorld.from_JSON method for details
    
    Keyword Arguments:
        gw_class {ClassVar[GridWorld]} -- the GridWorld subclass (default: {wumpus.WumpusWorld})

    Returns:
        Iterator[GridWorld] -- iterator over a sequence of Wumpus Worlds
    """
    world_descs = json.load(fp)

    for wds in world_descs:
        yield gw_class.from_JSON(wds)


def main(*cmd_args: str):
    def dir_path(path):
        if len(path) == 0 or os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('player', type=check_entrypoint, help="object reference for a Player subclass in the form 'importable.module:object.attr'. See <https://packaging.python.org/specifications/entry-points/#data-model> for details.")
    parser.add_argument('--path', help='The path to the player library, if not given it loads the files from the current directory', type=dir_path)
    parser.add_argument('--name', help='Name of the player', type=str)
    parser.add_argument('--worlds', help='JSON file with the description of the Grid worlds to use', type=argparse.FileType('r'))
    parser.add_argument('--wclass', '-w', type=str, default='WumpusWorld', help='class name of the Grid world to use')
    parser.add_argument('--random', help='Generate a number of random worlds', type=int, default=100)
    parser.add_argument('--horizon', help='Maximun number of steps for the agent', type=int, default=1000)
    parser.add_argument('--log', help='Write the log in the given file', type=argparse.FileType('w'))
    parser.add_argument('--no-summary', help='Don\'t print the summary', action='store_false', dest='summary')
    parser.add_argument('--logdir', help='Root directory for logs', type=dir_path, default='')
    args = parser.parse_args(cmd_args)

    player = get_player_class(args.player, path=args.path)(name=args.player if args.name is None else args.name)

    world_class = get_world_class(args.wclass)

    wgen = worlds_fromJSON(args.worlds, gw_class=world_class) if args.worlds else random_worlds(args.random, gw_class=world_class)

    log = exercise_player(player, wgen, horizon=args.horizon)

    if args.summary:
        print('Player {}: run {} episodes with average outcome of {} dying {} times (raised {} exceptions).'.format(player.name, len(log), sum(l.outcome for l in log) / len(log), len([l for l in log if not l.alive]), len([l for l in log if l.exception])), file=sys.stdout if args.log != sys.stdout else sys.stderr)

    if args.log:
        json.dump([e._asdict() for e in log], args.log, cls=EpisodeResult.JSONEncoder(), indent=None)


if __name__ == "__main__":
    main(*sys.argv[1:])
