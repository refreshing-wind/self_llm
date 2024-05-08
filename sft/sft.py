# -*- coding:utf-8 -*-

'''
 Author       : Xuexin
 Date         : 2024-05-07 10:32:43
 LastEditTime : 2024-05-08 18:18:38
 FilePath     : \\self_llm\\sft\\sft.py
 Description  : 
'''

import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

"""
Optional
Optional，可选类型， Optional[X] 等价于 X | None （或 Union[X, None] ）。 意思是说这个参数可以为空或已经声明的类型。

但值得注意的是，这个并不等价于可选参数，当它作为参数类型注解的时候，不代表这个参数可以不传递了，而是说这个参数可以传为 None。
"""

import datasets
import evaluate
import torch
import transformers
from datasets import load_dataset
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES) 
# 模型元组，transformers支持的全部模型的名称，包括 llama，gpt

@dataclass
class DataTrainingArguments:
    """
    定义用于训练的和评估的数据参数
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "使用数据集的名称"}
        )
    # field用来数据描述
    dataset_config_name: Optional[str] = field(
        default="None", metadata={"help": "使用数据集的配置名称"}
    )
    train_file: Optional[str] = field(default=None, metadata = {"help": "训练数据集的路径(文本文件)"})

    validation_file: Optional[str] = field(
        default=None,
        metadata = {"help": "用于评估的数据集，评估困惑度。（文本文件）"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "调试使用的最大训练数据量。会在训练数据中取出指定数量的数据",
                "例如：训练集有1000条数据，max_train_samples=100，那么只取100条数据进行训练"
            )
        }
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": ("调试使用的最大评估数据量。会在评估数据中取出指定数量的数据",
                "例如：评估集有1000条数据，max_eval_samples=100，那么只取100条数据进行评估")
        }
    )
    streaming: bool = field(default=False, metadata={"help": "是否使用流式输出"})
    block_size: Optional[int] = field(
        default=None,
        metadata={help: (
            "标记化后的可选输入序列长度。",
            "训练数据集将被截断为这个大小的块以进行训练。",
            "默认为单句输入的模型最大输入长度（考虑特殊标记）。"
        )}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "覆盖缓存的训练集和评估集，一个命令行参数，指示在缓存存在的情况下覆盖缓存。"}
    )
    validation_split_percentage:Optional[int] = field(
        default=5,
        metadata={"help": "如果不存在评估使用的验证集，从数据集中划分的比例"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "用于预处理的进程数。"},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "使用TXT文件时是否保留换行符."}
    )

    def __post_init__(self):
        """
        类实例化后再执行的操作，此处用于版本、数据的校验
        """
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")
        
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1] # assert 条件, assert报错信息
                assert extension in ["csv","json","txt"], "“train_file” should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(sep=".")[-1]
                assert extension in ["csv","json","txt"], "“validation_file” should be a csv, a json or a txt file."


@dataclass
class ModelArguments:
    """
    模型、tokenizer的微调或从头开始训练的配置
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help":(
                "用于初始化权重的检查点，如果想要从头开始训练则不设置该项"
            )
        },
    )

    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "如果需要从头开始训练，请从下列列表中选择模型类型： " +"，".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help":(
                "从头开始训练时，覆盖掉模型的默认配置。",
                "例如：n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        }
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help":"预训练使用的配置文件名称或路径(如果和模型名称不同)"
        }
    )
    tokenizer_name:Optional[str]=field(
        default=None,
        help="预训练使用的分词器名称或路径(如果和模型名称不同)"
    )
    cache_dir:Optional[str] = field(
        default=None,
        metadata={
            "help":"缓存目录，用于存储从huggingface.co下载的预训练权重文件"
        }
    )
    use_fast_tokenizer:bool = field(
        default=True,
        metadata={"help": "是否使用快速分词器(由分词器库支持)"}
    )
    model_revision: str = field(
        default="main",
        metadata="要使用的特定模型版本（可以是分支名称、标记名称或提交id）"
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "认证使用的token",
                "用作远程文件的 HTTP 承载授权的令牌。 如果未指定，将使用令牌 ",
                "运行“huggingface-cli login”时生成（存储在“~/.huggingface”中）。"
            )
        },
    )
    use_auth_token: bool =field(
        default=None,
        metadata={
            "help": "`use_auth_token` 参数已弃用，并将在 v4.34 中删除。 请改用“token”。."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "它说明了设置此选项为True时，允许在 Hub 上定义的自定义模型文件中包含代码",
                "但同时也提醒了用户要谨慎使用，因为这将在本地机器上执行 Hub 上的代码。"
            )
        },
    )
    token_dtype: Optional[str] = field(
        default=None,
        metadata={"help": (
            "覆盖默认的‘torch.dtype’,并加载此类型的模型，如果设置'auto'",
            "根据加载的模型权重自动设置数据类型"
        ),
        "choices": ["auto", "bfloat16", "float16", "float32"],
    }
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help":(
                "可以选择将模型创建为空壳，然后仅在加载预训练权重时才具体化其参数。"
                "设置 True 将有利于 LLM 加载时间和 RAM 消耗。"
                "“可以选择将模型创建为空壳，然后仅在加载预训练权重时才具体化其参数。设置 True 将有利于 LLM 加载时间和 RAM 消耗。”"
            )
        }
    )
    def __post_init__(self):
        """校验使用覆盖配置为True时需要的前提，前提1.没有指定配置文件，前提2.没有指定模型路径模型为从头训练"""
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


def main():
    pass