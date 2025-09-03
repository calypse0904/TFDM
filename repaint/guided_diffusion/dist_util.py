
import io  # 导入 io 库，用于处理二进制数据流，比如读取和写入二进制文件。
import blobfile as bf  # 导入 blobfile 库（简称 bf），用于处理分布式或远程文件存储，如 Google Cloud Storage 等云存储。
import torch as th  # 导入 PyTorch 库并将其重命名为 th，便于使用 PyTorch 的张量计算功能。
from mpi4py import MPI  # 导入 mpi4py 库，这是 Python 的 MPI（Message Passing Interface）接口，用于进行分布式计算中的进程间通信。
import os  # 导入 os 库，提供与操作系统交互的功能，如文件操作、环境变量设置等。
import socket  # 导入 socket 库，用于处理与网络相关的操作，如获取主机名、分配端口等。
import torch.distributed as dist  # 导入 PyTorch 的分布式模块，并将其重命名为 dist，用于在多个进程之间进行分布式训练。

GPUS_PER_NODE = 2  # 定义每个计算节点上使用的 GPU 数量，这里设置为 2 个 GPU。如果你有更多 GPU，可以调整这个值。
SETUP_RETRY_COUNT = 3  # 设置在初始化分布式进程组时的最大重试次数。如果初始化失败，它会重试 3 次。


def setup_dist(dist_num='1'):
    """
    Setup a distributed process group.
    初始化分布式进程组，用于分布式训练。
    """
    if dist.is_initialized():  # 检查分布式进程组是否已经初始化。
        return  # 如果已初始化，则直接返回，不做重复初始化。

    # os.environ["CUDA_VISIBLE_DEVICES"] = f"{MPI.COMM_WORLD.Get_rank() % GPUS_PER_NODE}"  # 通过 MPI 获取进程的 rank 来选择可用的

    # GPU（此行被注释掉）。
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 设置当前使用的 GPU 为设备 0。

    comm = MPI.COMM_WORLD  # 获取 MPI 的通信对象，用于进程间的通信。
    backend = "gloo" if not th.cuda.is_available() else "nccl"  # 根据是否有 CUDA 支持，选择使用 "gloo"（CPU 端）还是 "nccl"（GPU
    # 端）作为通信后端。

    if backend == "gloo":  # 如果使用 "gloo" 后端，通常是没有 CUDA 的情况（CPU 模式）。
        hostname = "localhost"  # 设置主机名为本地计算机。
    else:  # 如果使用 "nccl" 后端，表示使用 GPU。
        # hostname = socket.gethostbyname(socket.getfqdn())  # 通过获取主机的完整域名来获得主机 IP（此行被注释掉）。
        hostname = socket.gethostbyname("localhost")  # 通过获取本地机器的 IP 地址作为主机地址。

    # 设置环境变量 MASTER_ADDR 为当前主机地址，主节点进程通过广播主机地址给其他进程。
    os.environ["MASTER_ADDR"] = comm.bcast(hostname, root=0)
    os.environ["RANK"] = str(comm.rank)  # 设置当前进程的 rank（即进程 ID），每个进程都有一个唯一的 rank。
    os.environ["WORLD_SIZE"] = str(comm.size)  # 设置全局进程数（世界大小），即所有参与训练的进程数量。

    # 通过广播寻找一个可用的端口号，并将该端口号作为主端口设置。
    port = comm.bcast(_find_free_port(), root=0)
    os.environ["MASTER_PORT"] = str(port)  # 设置环境变量 MASTER_PORT 为广播的端口号，主节点将此端口用于监听其他进程。

    dist.init_process_group(backend=backend, init_method="env://")  # 初始化分布式进程组，指定后端和初始化方法。


def dev():
    """
    Get the device to use for torch.distributed.
    根据系统的硬件配置，选择适合的设备（GPU 或 CPU）来进行计算，并返回该设备，这样可以确保代码在不同环境下运行时，自动选择合适的硬件来执行计算
    """

    if th.cuda.is_available():
        return th.device(f"cuda")
    return th.device("cpu")


def load_state_dict(path, backend=None, **kwargs):
    """
    Load a PyTorch model's state_dict from a file path and return it.
    从文件路径加载 PyTorch 模型的 state_dict 并返回。
    """
    # 使用 BlobFile 打开指定路径的文件，并以二进制模式读取文件内容。
    # `bf.BlobFile` 是一个自定义的文件操作类，它支持从云存储或本地存储加载文件。
    with bf.BlobFile(path, "rb") as f:
        data = f.read()  # 读取文件中的二进制数据

    # 使用 io.BytesIO 将二进制数据包装为一个内存中的字节流。
    # 然后通过 PyTorch 的 `th.load` 加载这个字节流，并将模型的权重数据加载到内存中。
    # `**kwargs` 可以传递给 `th.load` 函数，以支持其他加载时的配置。
    return th.load(io.BytesIO(data), **kwargs)


def sync_params(params):
    """
    Synchronize a sequence of Tensors across ranks from rank 0.
    """
    for p in params:
        with th.no_grad():
            dist.broadcast(p, 0)


def _find_free_port():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]
    finally:
        s.close()
