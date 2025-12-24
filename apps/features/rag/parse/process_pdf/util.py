import os
import logging
import torch
import onnxruntime as ort
from . import operators

logger = logging.getLogger("book.util")

loaded_models = {}


def load_model(model_dir, nm, device_id: int | None = None):
    model_file_path = os.path.join(model_dir, nm + ".onnx")
    model_cached_tag = model_file_path + str(device_id) if device_id is not None else model_file_path

    global loaded_models
    loaded_model = loaded_models.get(model_cached_tag)
    if loaded_model:
        logger.info(f"load_model {model_file_path} reuses cached model")
        return loaded_model

    if not os.path.exists(model_file_path):
        raise ValueError("not find model file path {}".format(
            model_file_path))

    def cuda_is_available():
        try:
            target_id = 0 if device_id is None else device_id
            if torch.cuda.is_available() and torch.cuda.device_count() > target_id:
                return True
        except Exception as e:
            logger.info(f"Cannot load the model using GPU: {e}")
            return False
        return False

    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 2

    # https://github.com/microsoft/onnxruntime/issues/9509#issuecomment-951546580
    # Shrink GPU memory after execution
    run_options = ort.RunOptions()
    if cuda_is_available():
        gpu_mem_limit_mb = int(os.environ.get("OCR_GPU_MEM_LIMIT_MB", f"{2048*8}"))
        arena_strategy = os.environ.get("OCR_ARENA_EXTEND_STRATEGY", "kNextPowerOfTwo")
        provider_device_id = 0 if device_id is None else device_id
        cuda_provider_options = {
            "device_id": provider_device_id,  # Use specific GPU
            "gpu_mem_limit": max(gpu_mem_limit_mb, 0) * 1024 * 1024,
            "arena_extend_strategy": arena_strategy,  # gpu memory allocation strategy
        }
        sess = ort.InferenceSession(
            model_file_path,
            options=options,
            providers=['CUDAExecutionProvider'],
            provider_options=[cuda_provider_options]
        )
        run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "gpu:" + str(provider_device_id))
        logger.info(
            f"load_model {model_file_path} uses GPU (device {provider_device_id}, gpu_mem_limit={cuda_provider_options['gpu_mem_limit']}, arena_strategy={arena_strategy})")
    else:
        sess = ort.InferenceSession(
            model_file_path,
            options=options,
            providers=['CPUExecutionProvider'])
        run_options.add_run_config_entry("memory.enable_memory_arena_shrinkage", "cpu")
        logger.info(f"load_model {model_file_path} uses CPU")
    loaded_model = (sess, run_options)
    loaded_models[model_cached_tag] = loaded_model
    return loaded_model


def create_operators(op_param_list, global_config=None):
    """
    create operators based on the config

    Args:
        params(list): a dict list, used to create some operators
    """
    assert isinstance(
        op_param_list, list), ('operator config should be a list')
    ops = []
    for operator in op_param_list:
        assert isinstance(operator,
                          dict) and len(operator) == 1, "yaml format error"
        op_name = list(operator)[0]
        param = {} if operator[op_name] is None else operator[op_name]
        if global_config is not None:
            param.update(global_config)
        op = getattr(operators, op_name)(**param)
        ops.append(op)
    return ops


def transform(data, ops=None):
    """ transform """
    if ops is None:
        ops = []
    for op in ops:
        data = op(data)
        if data is None:
            return None
    return data
