import signal
import warnings
import random
import numpy as np
from env.utils.utils import tile_images
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from queue import Queue
from threading import Thread
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)
try:
    # Use torch.multiprocessing if we can.
    # We have yet to find a reason to not use it and
    # you are required to use it when sending a torch.Tensor
    # between processes
    from torch import multiprocessing as mp  # type:ignore
    import torch
except ImportError:
    torch = None
    import multiprocessing as mp  # type:ignore
import gym
from gym import spaces
from env.parallel_utils import profiling_wrapper
from env.make_collavn_env import CollaVN
from gibson2.utils.utils import parse_config

STEP_COMMAND = "step"
RESET_COMMAND = "reset"
RENDER_COMMAND = "render"
CLOSE_COMMAND = "close"
OBSERVATION_SPACE_COMMAND = "observation_space"
ACTION_SPACE_COMMAND = "action_space"
NUMBER_OF_EPISODES_COMMAND = "number_of_episodes"
CALL_COMMAND = "call"
EPISODE_COMMAND = "current_episode"
COUNT_EPISODES_COMMAND = "count_episodes"
EPISODE_OVER = "episode_over"
GET_METRICS = "get_metrics"


def init_env(config_filename, args, model_id, mode='headless', render_to_tensor=True, devices=None, rank=0):
    seed_num = random.randint(0, 100000)
    # seed_num = 100
    return CollaVN(config_file=config_filename, mode=mode, model_id=model_id, args=args, action_timestep=1 / 1.0,
                                                 physics_timestep=1 / 40.0, render_to_tensor=render_to_tensor, device_idx=devices, rank=rank, seed=seed_num)


class ProcessPyEnvironment(object):
    """Step a single env in a separate process for lock free paralellism."""

    # Message types for communication via the pipe.
    observation_spaces: List[spaces.Dict]
    number_of_episodes: List[Optional[int]]
    action_spaces: List[spaces.Dict]
    _workers: List[Union[mp.Process, Thread]]
    _is_waiting: bool
    _num_envs: int
    _auto_reset_done: bool
    _mp_ctx: BaseContext
    _connection_read_fns: List[Callable[[], Any]]
    _connection_write_fns: List[Callable[[Any], None]]

    def __init__(self,
                 env_constructor,
                 env_fn_args = None,
                 auto_reset_done = False,
                 multiprocessing_start_method='forkserver',
                 workers_ignore_signals: bool = False,
                 flatten=False):
        """Step environment in a separate process for lock free paralellism.

    The environment is created in an external process by calling the provided
    callable. This can be an environment class, or a function creating the
    environment and potentially wrapping it. The returned environment should
    not access global variables.

    Args:
      env_constructor: Callable that creates and returns a Python environment.
      flatten: Boolean, whether to assume flattened actions and time_steps
        during communication to avoid overhead.

    Attributes:
      observation_spec: The cached observation spec of the environment.
      action_spec: The cached action spec of the environment.
      time_step_spec: The cached time step spec of the environment.
    """
        self._env_constructor = env_constructor
        self._flatten = flatten
        #self._observation_spec = None
        #self._action_spec = None
        #self._time_step_spec = None
        self._is_waiting = False
        self._is_closed = True

        assert (
                env_fn_args is not None and len(env_fn_args) > 0
        ), "number of environments to be created should be greater than 0"
        self._num_envs = len(env_fn_args)
        assert multiprocessing_start_method in self._valid_start_methods, (
            "multiprocessing_start_method must be one of {}. Got '{}'"
        ).format(self._valid_start_methods, multiprocessing_start_method)
        self._auto_reset_done = auto_reset_done
        self._mp_ctx = mp.get_context(multiprocessing_start_method)
        self._workers = []
        (
            self._connection_read_fns,
            self._connection_write_fns,
        ) = self._spawn_workers(  # noqa
            env_fn_args,
            env_constructor,
            workers_ignore_signals=workers_ignore_signals,
        )
        self._is_closed = False

        for write_fn in self._connection_write_fns:
            write_fn((OBSERVATION_SPACE_COMMAND, None))
        self.observation_spaces = [
            read_fn() for read_fn in self._connection_read_fns
        ]
        for write_fn in self._connection_write_fns:
            write_fn((ACTION_SPACE_COMMAND, None))
        self.action_spaces = [
            read_fn() for read_fn in self._connection_read_fns
        ]
        for write_fn in self._connection_write_fns:
            write_fn((NUMBER_OF_EPISODES_COMMAND, None))
        self.number_of_episodes = [
            read_fn() for read_fn in self._connection_read_fns
        ]
        self._paused: List[Tuple] = []

    @property
    def num_envs(self):
        r"""number of individual environments."""
        return self._num_envs - len(self._paused)

    @staticmethod
    @profiling_wrapper.RangeContext("_worker_env")
    def _worker_env(
            connection_read_fn: Callable,
            connection_write_fn: Callable,
            env_fn: Callable,
            env_fn_args: Tuple[Any],
            auto_reset_done: bool,
            mask_signals: bool = False,
            child_pipe: Optional[Connection] = None,
            parent_pipe: Optional[Connection] = None,
    ) -> None:
        r"""process worker for creating and interacting with the environment."""
        if mask_signals:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, signal.SIG_IGN)

            signal.signal(signal.SIGUSR1, signal.SIG_IGN)
            signal.signal(signal.SIGUSR2, signal.SIG_IGN)
        env = env_fn(*env_fn_args)  # single environment
        if parent_pipe is not None:
            parent_pipe.close()  # close the parent_pipe
        try:
            command, data = connection_read_fn()
            while command != CLOSE_COMMAND:
                if command == STEP_COMMAND:
                    observations = env.step(data)
                    if auto_reset_done and env.episode_over:
                        observations = env.reset()
                    connection_write_fn(observations)
                elif command == RESET_COMMAND:
                    observations = env.reset()
                    connection_write_fn(observations)
                #   elif command == RENDER_COMMAND:
                #       connection_write_fn(env.render(*data[0], **data[1]))
                elif command in {
                    OBSERVATION_SPACE_COMMAND,
                    ACTION_SPACE_COMMAND,
                    NUMBER_OF_EPISODES_COMMAND,
                }:
                    connection_write_fn(getattr(env, command))
                #  Call other custom function
                elif command == CALL_COMMAND:
                    function_name, function_args = data
                    if function_args is None or len(function_args) == 0:
                        result = getattr(env, function_name)()
                    else:
                        result = getattr(env, function_name)(**function_args)
                    connection_write_fn(result)
                # TODO: update CALL_COMMAND for getting attribute like this
                elif command == EPISODE_COMMAND:
                    connection_write_fn(env.current_episode)

                elif command == COUNT_EPISODES_COMMAND:
                    connection_write_fn(len(env.episodes))

                elif command == EPISODE_OVER:
                    connection_write_fn(env.episode_over)

                #  elif command == GET_METRICS:
                #      result = env.get_metrics()
                #      connection_write_fn(result)

                else:
                    raise NotImplementedError

                with profiling_wrapper.RangeContext("worker wait for command"):
                    command, data = connection_read_fn()

            if child_pipe is not None:
                child_pipe.close()
        except KeyboardInterrupt:
            print("Worker KeyboardInterrupt")
        finally:
            env.close()

    def _spawn_workers(
            self,
            env_fn_args: Sequence[Tuple],
            env_constructor: Callable[..., CollaVN] = init_env,
            workers_ignore_signals: bool = False,
    ) -> Tuple[List[Callable[[], Any]], List[Callable[[Any], None]]]:
        parent_connections, worker_connections = zip(
            *[self._mp_ctx.Pipe(duplex=True) for _ in range(self._num_envs)]
        )
        self._workers = []
        for worker_conn, parent_conn, env_args in zip(
                worker_connections, parent_connections, env_fn_args
        ):
            ps = self._mp_ctx.Process(
                target=self._worker_env,
                args=(
                    worker_conn.recv,
                    worker_conn.send,
                    env_constructor,
                    env_args,
                    self._auto_reset_done,
                    workers_ignore_signals,
                    worker_conn,
                    parent_conn,
                ),
            )
            self._workers.append(cast(mp.Process, ps))
            ps.daemon = True
            ps.start()
            worker_conn.close()
        return (
            [p.recv for p in parent_connections],
            [p.send for p in parent_connections],
        )

    def current_episodes(self):
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((EPISODE_COMMAND, None))  # call env.episode
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())  # return result
        self._is_waiting = False
        return results

    def count_episodes(self):
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((COUNT_EPISODES_COMMAND, None))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def episode_over(self):
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((EPISODE_OVER, None))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    def reset(self):
        r"""Reset all the vectorized environments
        :return: list of outputs from the reset method of envs.
        """
        self._is_waiting = True
        for write_fn in self._connection_write_fns:
            write_fn((RESET_COMMAND, None))
        results = []

        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        obs, rew, done, infos = zip(*results)
        self._is_waiting = False
        return obs, rew, done, infos

    def reset_at(self, index_env: int):
        r"""Reset in the index_env environment in the vector.
        :param index_env: index of the environment to be reset
        :return: list containing the output of reset method of indexed env.
        """
        self._is_waiting = True
        self._connection_write_fns[index_env]((RESET_COMMAND, None))
        results = [self._connection_read_fns[index_env]()]
        self._is_waiting = False
        return results

    def step_at(self, index_env: int, action: Dict[str, Any]):
        r"""Step in the index_env environment in the vector.
        :param index_env: index of the environment to be stepped into
        :param action: action to be taken
        :return: list containing the output of step method of indexed env.
        """
        self._is_waiting = True
        self._warn_cuda_tensors(action)
        self._connection_write_fns[index_env]((STEP_COMMAND, action))
        results = [self._connection_read_fns[index_env]()]
        self._is_waiting = False
        return results

    def async_step(self, actions: List[Union[int, str, Dict[str, Any]]]) -> None:
        r"""Asynchronously step in the environments.
        :param data: list of size _num_envs containing keyword arguments to
            pass to :ref:`step` method for each Environment. For example,
            :py:`[{"action": "TURN_LEFT", "action_args": {...}}, ...]`.
        """

        self._is_waiting = True
        for write_fn, action in zip(self._connection_write_fns, actions):
            write_fn((STEP_COMMAND, action))

    @profiling_wrapper.RangeContext("wait_step")
    def wait_step(self) -> List[Any]:
        r"""Wait until all the asynchronized environments have synchronized."""
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        obs, rews, dones, infos = zip(*results)
        self._is_waiting = False
        return obs, rews, dones, infos

    def step(self, data: List[Union[int, str, Dict[str, Any]]]) -> Tuple[List[Any], List[Any]]:
        r"""Perform actions in the vectorized environments.
        :param data: list of size _num_envs containing keyword arguments to
            pass to :ref:`step` method for each Environment. For example,
            :py:`[{"action": "TURN_LEFT", "action_args": {...}}, ...]`.
        :return: list of outputs from the step method of envs.
        """
        self.async_step(data)
        return self.wait_step()

    def close(self) -> None:
        if self._is_closed:
            return

        if self._is_waiting:
            for read_fn in self._connection_read_fns:
                read_fn()

        for write_fn in self._connection_write_fns:
            write_fn((CLOSE_COMMAND, None))

        for _, _, write_fn, _ in self._paused:
            write_fn((CLOSE_COMMAND, None))

        for process in self._workers:
            process.join()

        for _, _, _, process in self._paused:
            process.join()

        self._is_closed = True

    def pause_at(self, index: int) -> None:
        r"""Pauses computation on this env without destroying the env.
        :param index: which env to pause. All indexes after this one will be
            shifted down by one.
        This is useful for not needing to call steps on all environments when
        only some are active (for example during the last episodes of running
        eval episodes).
        """
        if self._is_waiting:
            for read_fn in self._connection_read_fns:
                read_fn()
        read_fn = self._connection_read_fns.pop(index)
        write_fn = self._connection_write_fns.pop(index)
        worker = self._workers.pop(index)
        self._paused.append((index, read_fn, write_fn, worker))

    def resume_all(self) -> None:
        r"""Resumes any paused envs."""
        for index, read_fn, write_fn, worker in reversed(self._paused):
            self._connection_read_fns.insert(index, read_fn)
            self._connection_write_fns.insert(index, write_fn)
            self._workers.insert(index, worker)
        self._paused = []

    def call_at(
        self,
        index: int,
        function_name: str,
        function_args: Optional[Dict[str, Any]] = None,
    ) -> Any:
        r"""Calls a function (which is passed by name) on the selected env and
        returns the result.
        :param index: which env to call the function on.
        :param function_name: the name of the function to call on the env.
        :param function_args: optional function args.
        :return: result of calling the function.
        """
        self._is_waiting = True
        self._connection_write_fns[index](
            (CALL_COMMAND, (function_name, function_args))
        )
        result = self._connection_read_fns[index]()
        self._is_waiting = False
        return result

    def call(
        self,
        function_names: List[str],
        function_args_list: Optional[List[Any]] = None,
    ) -> List[Any]:
        r"""Calls a list of functions (which are passed by name) on the
        corresponding env (by index).
        :param function_names: the name of the functions to call on the envs.
        :param function_args_list: list of function args for each function. If
            provided, :py:`len(function_args_list)` should be as long as
            :py:`len(function_names)`.
        :return: result of calling the function.
        """
        self._is_waiting = True
        if function_args_list is None:
            function_args_list = [None] * len(function_names)
        assert len(function_names) == len(function_args_list)
        func_args = zip(function_names, function_args_list)
        for write_fn, func_args_on in zip(
            self._connection_write_fns, func_args
        ):
            write_fn((CALL_COMMAND, func_args_on))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        self._is_waiting = False
        return results

    @property
    def _valid_start_methods(self) -> Set[str]:
        return {"forkserver", "spawn", "fork"}

    def _warn_cuda_tensors(
        self, action: Dict[str, Any], prefix: Optional[str] = None
    ):
        # action should be a cpu tensor
        if torch is None:
            return

        for k, v in action.items():
            if isinstance(v, dict):
                subk = f"{prefix}.{k}" if prefix is not None else k
                self._warn_cuda_tensors(v, prefix=subk)
            elif torch.is_tensor(v) and v.device.type == "cuda":
                subk = f"{prefix}.{k}" if prefix is not None else k
                warnings.warn(
                    "Action with key {} is a CUDA tensor."
                    "  This will result in a CUDA context in the subproccess worker."
                    "  Using CPU tensors instead is recommended.".format(subk)
                )

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class ThreadedVectorEnv(ProcessPyEnvironment):
    r"""Provides same functionality as :ref:`VectorEnv`, the only difference
    is it runs in a multi-thread setup inside a single process.
    The :ref:`VectorEnv` runs in a multi-proc setup. This makes it much easier
    to debug when using :ref:`VectorEnv` because you can actually put break
    points in the environment methods. It should not be used for best
    performance.
    """

    def _spawn_workers(
        self,
        env_fn_args: Sequence[Tuple],
        env_constructor: Callable[..., CollaVN] = init_env,
        workers_ignore_signals: bool = False,
    ) -> Tuple[List[Callable[[], Any]], List[Callable[[Any], None]]]:
        queues: Iterator[Tuple[Any, ...]] = zip(
            *[(Queue(), Queue()) for _ in range(self._num_envs)]
        )
        parent_read_queues, parent_write_queues = queues
        self._workers = []
        for parent_read_queue, parent_write_queue, env_args in zip(
            parent_read_queues, parent_write_queues, env_fn_args
        ):
            thread = Thread(
                target=self._worker_env,
                args=(
                    parent_write_queue.get,
                    parent_read_queue.put,
                    env_constructor,
                    env_args,
                    self._auto_reset_done,
                ),
            )
            self._workers.append(thread)
            thread.daemon = True
            thread.start()
        return (
            [q.get for q in parent_read_queues],
            [q.put for q in parent_write_queues],
        )

class VecPyTorch():

    def __init__(self, venv, device):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_spaces
        self.action_space = venv.action_spaces
        self.device = device

    def reset(self):
        obs,rew, done, info = self.venv.reset()
        # obs = torch.from_numpy(obs).float().to(self.device)
        obs = [obs_i.to(self.device) for obs_i in obs] # cuda:0
        obs = torch.stack(obs, dim=0)  # s, a, h, w, 3
        return obs, rew, done, info

    def step_async(self, actions):
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        # obs = torch.from_numpy(obs).float().to(self.device)
        obs = [obs_i.to(self.device) for obs_i in obs]
        obs = torch.stack(obs, dim=0)
        reward = torch.stack(reward, dim=0).to(self.device)
        # reward = torch.from_numpy(reward).float()
        return obs, reward, done, info

    def step(self, actions):
        # actions = actions.cpu().numpy()
        obs, reward, done, info = self.venv.step(actions)
        obs = [obs_i.to(self.device) for obs_i in obs]
        reward = [rew_i.to(self.device) for rew_i in reward]
        obs = torch.stack(obs, dim=0)
        reward = torch.stack(reward, dim=0)

        return obs, reward, done, info


def construct_envs(config_file_name, args, scenes_split=None, mode='headless', render_to_tensor=True, gpu_id_list=None):

    args_list = []

    # temp TODO:
    # args.num_processes == len(scenes_split)
    if len(scenes_split) > 0:
        assert len(scenes_split) >= args.num_processes, (
            "reduce the number of processes as there "
            "aren't enough number of scenes"
        )
        scenes_split_size = int(np.floor(len(scenes_split) / args.num_processes))

    # current available gpu id
    for i in range(args.num_processes):
        if i < args.num_processes_on_first_gpu:
            gpu_id = 0
        else:
            gpu_id = int((i - args.num_processes_on_first_gpu)
                         // args.num_processes_per_gpu) + args.sim_gpu_id
        gpu_id = min(torch.cuda.device_count() - 1, gpu_id)
        # render_device_id = gpu_id_list[gpu_id]
        render_device_id = gpu_id
        model_id = scenes_split[i]
        args_list.append([config_file_name, args, model_id, mode, render_to_tensor, render_device_id, i])
    envs = ProcessPyEnvironment(
        env_constructor=init_env,
        env_fn_args=args_list,
    )

    return envs

def make_parallel_env(config_filename, args, scenes_split=None, mode='headless', render_to_tensor=True, gpu_id_list=None):
    # TODO: temp simple version
    # config = parse_config(config_filename)
    # args.num_process_per_gpu = args.num_prog
    envs = construct_envs(config_filename, args, scenes_split, mode, render_to_tensor, gpu_id_list)
    envs = VecPyTorch(envs, args.device)
    return envs
