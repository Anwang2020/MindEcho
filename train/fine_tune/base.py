import os
import datetime

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForSeq2Seq
from peft import PeftModel
from transformers import TextStreamer
import torch
import warnings

from train_mode import bit_fit, prompt_tuning, p_tuning, prefix_tuning, lora, IA3

warnings.filterwarnings('ignore')


class ModelTrainer:
    def __init__(self, model_path: str, data_path, max_len: int, train_type='lora',
                 use_gpu=True, output_dir=None, out_path=None,
                 logging_step=10, batch_size=1, epoch=1, gradient_accumulation_steps=8,
                 lr=5e-5, warmup_steps=5, lr_scheduler_type='linear',
                 prompt_tuning_init_text=None, num_virtual_tokens=10,
                 encoder_hidden_size=1024,
                 lora_r=8, lora_alpha=16, lora_dropout=0.05, lora_target_modules=[], modules_to_save=[],
                 model_params={}, train_params={}
                 ):
        """
        初始化模型训练器
        :param model_path: 预训练模型路径
        :param data_path: 训练数据路径
        :param max_len: 最大序列长度
        :param train_type: 微调类型，支持'lora'或'fitbit'
        :param use_gpu: 是否使用GPU加速训练
        :param output_dir: 训练输出目录
        :param out_path: 模型保存路径
        :param logging_step: 日志记录步长
        :param batch_size: 批处理大小
        :param epoch: 训练轮数
        :param gradient_accumulation_steps: 梯度累积步数
        """
        # base
        self.data_path = data_path
        self.model_name = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
                                                          dtype=compute_dtype,
                                                          low_cpu_mem_usage=True)
        self.cuda = use_gpu and torch.cuda.is_available()
        if self.cuda and self.model.device.type != 'cuda':
            self.model = self.model.cuda()
        self.MAX_LENGTH = max_len
        self.batch_size = batch_size
        self.epoch = epoch
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.lr_scheduler_type = lr_scheduler_type
        self.logging_step = logging_step
        self.out_path = out_path
        self.out_dir = output_dir
        self.train_type = train_type

        # prompt_tuning_config
        self.prompt_tuning_init_text = prompt_tuning_init_text
        self.num_virtual_tokens = num_virtual_tokens
        # prefix_tuning_config
        self.encoder_hidden_size = encoder_hidden_size
        # lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = lora_target_modules
        self.modules_to_save = modules_to_save

        self.model_params = model_params
        self.train_params = train_params

    def data_processor(self, example):
        instruction = self.tokenizer(
            "\n".join(["Human: " + example["instruction"], example["input"]]).strip() + "\n\nAssistant: ")
        response = self.tokenizer(example["output"] + self.tokenizer.eos_token)
        input_ids = instruction["input_ids"] + response["input_ids"]
        attention_mask = instruction["attention_mask"] + response["attention_mask"]
        labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]
        if len(input_ids) > self.MAX_LENGTH:
            input_ids = input_ids[:self.MAX_LENGTH]
            attention_mask = attention_mask[:self.MAX_LENGTH]
            labels = labels[:self.MAX_LENGTH]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

    def train(self):

        if '.json' in self.data_path:
            ds = load_dataset("json", data_files=self.data_path)
        else:
            ds = load_dataset("json", data_dir=self.data_path)
        ds = ds['train']
        tokenized_ds = ds.map(self.data_processor, remove_columns=ds.column_names)
        params_num = sum(param.numel() for param in self.model.parameters())
        print(f"模型总参数数量：{params_num}")

        match self.train_type:
            case 'lora':
                # model, lora_r, lora_alpha, lora_dropout,target_modules, modules_to_save
                self.model = lora(self.model, self.lora_r, self.lora_alpha, self.lora_dropout, self.target_modules,
                                  self.modules_to_save, self.model_params)
            case 'prompt_tuning':
                # prompt_tuning_init_text: str, model_path: str, tokenizer, model, num_virtual_tokens
                self.model = prompt_tuning(self.prompt_tuning_init_text, self.model_name, self.tokenizer, self.model,
                                           self.num_virtual_tokens, self.model_params)
            case "p_tuning":
                self.model = p_tuning(self.model, self.num_virtual_tokens, self.model_params)
            case 'prefix_tuning':
                self.model = prefix_tuning(self.model, self.num_virtual_tokens, self.encoder_hidden_size,
                                           self.model_params)
            case 'fitbit':
                self.model = bit_fit(self.model)
            case "IA3":
                self.model = IA3(self.model, self.model_params)
            case _:
                raise ValueError("Invalid train_type")

        self.model.print_trainable_parameters()
        # 模型接收梯度
        self.model.enable_input_require_grads()
        # 在使用梯度检查点时禁用缓存
        self.model.config.use_cache = False

        args = TrainingArguments(
            learning_rate=self.lr,
            warmup_steps=self.warmup_steps,
            lr_scheduler_type=self.lr_scheduler_type,
            output_dir=self.out_dir,  # 输出文件夹存储模型的预测结果和模型文件checkpoints
            per_device_train_batch_size=self.batch_size,  # 默认8, 对于训练的时候每个 GPU核或者CPU 上面对应的一个批次的样本数
            gradient_accumulation_steps=self.gradient_accumulation_steps,  # 默认1, 在执行反向传播/更新参数之前, 对应梯度计算累积了多少次
            logging_steps=self.logging_step,  # 每隔10迭代落地一次日志
            num_train_epochs=self.epoch,  # 整体上数据集让模型学习多少遍
            gradient_checkpointing=True,
            **self.train_params
        )

        tokenized_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=tokenized_ds,
            # 构建一个个批次数据所需要的
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer, padding=True)
        )
        trainer.train()

    @staticmethod
    def get_model_info(model_path):
        model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True)
        params_num = sum(param.numel() for param in model.parameters())
        print(f"模型总参数数量：{params_num}")
        params_name = '\n'.join(name for name, _ in model.named_parameters())
        return f"模型总参数量{params_num}", params_name

    @staticmethod
    def inference(model, model_id, input_text, max_length=2000, stream=False):
        peft_model = PeftModel.from_pretrained(model=model, model_id=model_id)
        peft_model = peft_model.cuda()
        tokenizer = AutoTokenizer.from_pretrained(model)
        assert input_text, "请输入问题"
        ipt = tokenizer("Human: {}\n{}".format(input_text, "").strip() + "\n\nAssistant: ",
                        return_tensors="pt").to(model.device)
        # 把model输出的response结果再次转为文本
        resp = tokenizer.decode(peft_model.generate(**ipt, max_length=max_length, do_sample=True)[0],
                                skip_special_tokens=True)
        return resp


if __name__ == '__main__':
    output_base_dir = r'E:\file\custom_model\lora_models'
    version = datetime.datetime.now().strftime("%Y%m%d%H%M%S")[2:-2]
    output_dir = os.path.join(output_base_dir, f"base_{version}")
    os.makedirs(output_dir, exist_ok=True)
    model_trainer = ModelTrainer(
        model_path=r'E:\llm_model\Qwen2.5-7B-Instruct',
        data_path=r'E:\file\custom_model\data\datasets_2026_01_12_19_03_27.json',
        max_len=2048,
        train_type='lora',
        use_gpu=True,
        output_dir=output_dir,

        logging_step=20,

        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        lora_target_modules=['q_proj', 'v_proj'],
    )
    model_trainer.train()
