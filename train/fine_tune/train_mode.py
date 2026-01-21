def bit_fit(model):
    num_param = 0
    for name, param in model.named_parameters():
        if "bias" not in name:
            # 参数进行冻结 frozen
            param.requires_grad = False
        else:
            num_param += param.numel()
    return model


def prompt_tuning(prompt_tuning_init_text: str, model_path: str, tokenizer, model, num_virtual_tokens=10, params={}):
    from peft import PromptTuningConfig, TaskType, get_peft_model, PromptTuningInit
    if prompt_tuning_init_text is None:
        config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=num_virtual_tokens,
            tokenizer_name_or_path=model_path,
            **params
        )
    else:
        config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            prompt_tuning_init_text=prompt_tuning_init_text,
            num_virtual_tokens=len(tokenizer(prompt_tuning_init_text)["input_ids"]),
            tokenizer_name_or_path=model_path,
            **params
        )

    model = get_peft_model(model, config)
    return model


def p_tuning(model, num_virtual_tokens, params):
    from peft import PromptEncoderConfig, get_peft_model, TaskType, PromptEncoderReparameterizationType
    config = PromptEncoderConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=num_virtual_tokens, **params)

    model = get_peft_model(model, config)
    return model


def prefix_tuning(model, num_virtual_tokens, encoder_hidden_size=1024, params={}):
    from peft import PrefixTuningConfig, get_peft_model, TaskType
    config = PrefixTuningConfig(task_type=TaskType.CAUSAL_LM, num_virtual_tokens=num_virtual_tokens,
                                prefix_projection=True, encoder_hidden_size=encoder_hidden_size,
                                **params
                                )
    model = get_peft_model(model, config)
    return model


def lora(model, lora_r, lora_alpha, lora_dropout, target_modules, modules_to_save, params):
    from peft import LoraConfig, get_peft_model, TaskType
    print(f"LoRA r: {lora_r}, alpha: {lora_alpha}, dropout: {lora_dropout}")
    if not lora_alpha:
        lora_alpha = 2 * lora_r
    if not target_modules:
        config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                            r=lora_r,
                            lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout,
                            inference_mode=False,
                            **params
                            )
    else:
        config = LoraConfig(task_type=TaskType.CAUSAL_LM,
                            target_modules=target_modules,
                            modules_to_save=modules_to_save,
                            r=lora_r,
                            lora_alpha=lora_alpha,
                            lora_dropout=lora_dropout,
                            inference_mode=False,
                            **params
                            )
    model = get_peft_model(model, config)
    return model


def IA3(model, params):
    from peft import IA3Config, get_peft_model, TaskType
    config = IA3Config(task_type=TaskType.CAUSAL_LM, **params)
    model = get_peft_model(model, config)
    return model
