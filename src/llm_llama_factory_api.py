import os
import yaml
import subprocess
import sys
from pathlib import Path

import os
import logging
import sys
from logger import init_logger

base_dir = os.path.split(os.path.realpath(__file__))[0]
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
if not logger.handlers:
    logger = init_logger('../data/log')


def create_inference_yaml_config(dataset_dir='./data/dataset',
                                 eval_dataset='alpaca_zh_demo',
                                 output_dir='./data/saves/Qwen3-8B/lora/predict',
                                 model_name_or_path='./data/model/Qwen3-8B',
                                 adapter_name_or_path=None,
                                 yaml_file_path='./data/train_lora/predict.yaml',
                                 test_max_samples=100000,
                                 per_device_eval_batch_size=16,
                                 temperature=0.2):
    pass
    # Define configuration data.
    config = {
        'model_name_or_path': model_name_or_path,
        'trust_remote_code': True,
        'adapter_name_or_path': adapter_name_or_path,
        # method
        'stage': 'sft',
        'do_predict': True,
        'finetuning_type': 'lora',

        'dataset_dir': dataset_dir,
        'eval_dataset': eval_dataset,
        'template': 'qwen',
        'cutoff_len': 1024,
        'max_samples': test_max_samples,
        'overwrite_cache': True,
        'preprocessing_num_workers': 16,
        'dataloader_num_workers': 4,
        'per_device_eval_batch_size': per_device_eval_batch_size,
        'temperature': temperature,

        'output_dir': output_dir,
        'overwrite_output_dir': True,
        'predict_with_generate': True

    }

    # Ensure the data directory exists.
    os.makedirs('./data', exist_ok=True)

    try:
        with open(yaml_file_path, 'w', encoding='utf-8') as f:
            # Add section comments.
            f.write("### model\n")
            if config['adapter_name_or_path'] is not None:
                yaml.dump({
                    'model_name_or_path': config['model_name_or_path'],
                    'trust_remote_code': config['trust_remote_code'],
                    'adapter_name_or_path': config['adapter_name_or_path']
                }, f, default_flow_style=False, allow_unicode=True)

                f.write("\n### method\n")
                yaml.dump({
                    'stage': config['stage'],
                    'do_predict': config['do_predict'],
                    'finetuning_type': config['finetuning_type']
                }, f, default_flow_style=False, allow_unicode=True)
            else:
                yaml.dump({
                    'model_name_or_path': config['model_name_or_path'],
                    'trust_remote_code': config['trust_remote_code'],
                }, f, default_flow_style=False, allow_unicode=True)

                f.write("\n### method\n")
                yaml.dump({
                    'stage': config['stage'],
                    'do_predict': config['do_predict'],
                }, f, default_flow_style=False, allow_unicode=True)


            f.write("\n### dataset\n")
            yaml.dump({
                'dataset_dir': config['dataset_dir'],
                'eval_dataset': config['eval_dataset'],
                'template': config['template'],
                'cutoff_len': config['cutoff_len'],
                'max_samples': config['max_samples'],
                'overwrite_cache': config['overwrite_cache'],
                'preprocessing_num_workers': config['preprocessing_num_workers'],
                'dataloader_num_workers': config['dataloader_num_workers'],
                'per_device_eval_batch_size': config['per_device_eval_batch_size']
            }, f, default_flow_style=False, allow_unicode=True)

            f.write("\n### output\n")
            yaml.dump({
                'output_dir': config['output_dir'],
                'overwrite_output_dir': config['overwrite_output_dir'],
                'predict_with_generate': config['predict_with_generate']
            }, f, default_flow_style=False, allow_unicode=True)

        logger.info(f"""Configuration file created: {yaml_file_path}""")
        return yaml_file_path

    except Exception as e:
        logger.info(f"""Error creating configuration file: {e}""")
        return None


def create_train_yaml_config(dataset_dir='../data/dataset',  # '/mnt/d/project-hxd/pprs/data',
                             dataset='alpaca_zh_demo',
                             output_dir='./data/saves/Qwen3-8B/lora/sft',
                             # '/home/dell/LLaMA-Factory/saves/Qwen3-4B/lora/sft',
                             model_name_or_path='../data/llm/model/Qwen3-8B',  # '/home/dell/LLaMA-Factory/models/Qwen3-4B',
                             yaml_file_path='./data/train_lora/train.yaml',
                             train_max_samples=500000,
                             num_train_epochs=3):
    "Create and return a YAML configuration. dataset_dir contains dataset_info.json and datasets; dataset is a key in that file; output_dir stores fine-tuned parameters; model_name_or_path locates the base model; yaml_file_path is the destination configuration path."
    # Define configuration data.
    config = {
        # model
        'model_name_or_path': model_name_or_path,
        'trust_remote_code': True,

        # method
        'stage': 'sft',
        'do_train': True,
        'finetuning_type': 'lora',
        'lora_rank': 8,
        'lora_target': 'all',

        # Dataset settings (adjust as needed)
        'dataset_dir': dataset_dir,
        'dataset': dataset,
        'template': 'qwen',
        'cutoff_len': 2048,
        'max_samples': train_max_samples,
        'overwrite_cache': True,
        'preprocessing_num_workers': 16,
        'dataloader_num_workers': 4,

        # Output settings (adjust as needed)
        'output_dir': output_dir,
        'logging_steps': 10,
        'save_steps': 500,
        'plot_loss': True,
        'overwrite_output_dir': True,
        'save_only_model': False,
        'report_to': 'none',

        # train
        'per_device_train_batch_size': 1,
        'gradient_accumulation_steps': 8,
        'learning_rate': 1.0e-4,
        'num_train_epochs': num_train_epochs,
        'lr_scheduler_type': 'cosine',
        'warmup_ratio': 0.1,
        'bf16': True,
        'ddp_timeout': 180000000,
        'resume_from_checkpoint': None,

        # Evaluation settings (optional; omitted or set to None)
        # 'eval_dataset': dataset,
        # 'val_size': 0.1,
        # 'per_device_eval_batch_size': 1,
        # 'eval_strategy': 'steps',
        # 'eval_steps': 500
    }

    # Ensure the data directory exists.
    os.makedirs('./data', exist_ok=True)

    try:
        with open(yaml_file_path, 'w', encoding='utf-8') as f:
            # Add section comments.
            f.write("### model\n")
            yaml.dump({
                'model_name_or_path': config['model_name_or_path'],
                'trust_remote_code': config['trust_remote_code']
            }, f, default_flow_style=False, allow_unicode=True)

            f.write("\n### method\n")
            yaml.dump({
                'stage': config['stage'],
                'do_train': config['do_train'],
                'finetuning_type': config['finetuning_type'],
                'lora_rank': config['lora_rank'],
                'lora_target': config['lora_target']
            }, f, default_flow_style=False, allow_unicode=True)

            f.write("\n### dataset\n")
            yaml.dump({
                'dataset_dir': config['dataset_dir'],
                'dataset': config['dataset'],
                'template': config['template'],
                'cutoff_len': config['cutoff_len'],
                'max_samples': config['max_samples'],
                'overwrite_cache': config['overwrite_cache'],
                'preprocessing_num_workers': config['preprocessing_num_workers'],
                'dataloader_num_workers': config['dataloader_num_workers']
            }, f, default_flow_style=False, allow_unicode=True)

            f.write("\n### output\n")
            yaml.dump({
                'output_dir': config['output_dir'],
                'logging_steps': config['logging_steps'],
                'save_steps': config['save_steps'],
                'plot_loss': config['plot_loss'],
                'overwrite_output_dir': config['overwrite_output_dir'],
                'save_only_model': config['save_only_model'],
                'report_to': config['report_to']
            }, f, default_flow_style=False, allow_unicode=True)

            f.write("\n# choices: [none, wandb, tensorboard, swanlab, mlflow]\n")

            f.write("\n### train\n")
            yaml.dump({
                'per_device_train_batch_size': config['per_device_train_batch_size'],
                'gradient_accumulation_steps': config['gradient_accumulation_steps'],
                'learning_rate': config['learning_rate'],
                'num_train_epochs': config['num_train_epochs'],
                'lr_scheduler_type': config['lr_scheduler_type'],
                'warmup_ratio': config['warmup_ratio'],
                'bf16': config['bf16'],
                'ddp_timeout': config['ddp_timeout'],
                'resume_from_checkpoint': config['resume_from_checkpoint']
            }, f, default_flow_style=False, allow_unicode=True)

            f.write("\n### eval\n")
            f.write("# eval_dataset: alpaca_en_demo\n")
            f.write("# val_size: 0.1\n")
            f.write("# per_device_eval_batch_size: 1\n")
            f.write("# eval_strategy: steps\n")
            f.write("# eval_steps: 500\n")

        logger.info(f"""Configuration file created: {yaml_file_path}""")
        return yaml_file_path

    except Exception as e:
        logger.info(f"""Error creating configuration file: {e}""")
        return None


def run_llamafactory_train(config_path):
    "Run training with llamafactory-cli train."
    try:
        # Check that the configuration file exists.
        if not os.path.exists(config_path):
            logger.info(f"""Configuration file does not exist: {config_path}""")
            return False

        # Build the command.
        cmd = ['llamafactory-cli', 'train', config_path]

        logger.info(f"""Running training command: {' '.join(cmd)}""")
        logger.info("=" * 50)

        # Execute the command.
        result = subprocess.run(
            cmd,
            capture_output=False,  # Stream output directly to the terminal.
            text=True,
            check=False  # Inspect the return code without raising automatically.
        )

        if result.returncode == 0:
            logger.info("=" * 50)
            logger.info("Training command completed successfully")
            return True
        else:
            logger.info("=" * 50)
            logger.info(f"""Training command failed with return code: {result.returncode}""")
            return False

    except FileNotFoundError:
        logger.info("llamafactory-cli was not found. Check that LLaMA-Factory is installed.")
        logger.info("   Try: pip install llamafactory")
        return False
    except Exception as e:
        logger.info(f"""Training error: {e}""")
        return False


def llm_train(dataset, output_dir, yaml_file_path, dataset_dir, model_name_or_path, train_max_samples=500000, num_train_epochs=3):
    "Run the configuration and execution workflow."
    logger.info("LLaMA-Factory configuration and training script")
    logger.info("=" * 50)

    # 1. Create the YAML configuration.
    logger.info("Step 1: Create the training YAML configuration")
    config_path = create_train_yaml_config(dataset_dir=dataset_dir,
                                           dataset=dataset,
                                           output_dir=output_dir,
                                           model_name_or_path=model_name_or_path,
                                           yaml_file_path=yaml_file_path,
                                           train_max_samples=train_max_samples,
                                           num_train_epochs=num_train_epochs)  # create_train_yaml_config()

    if config_path is None:
        logger.info("Configuration creation failed; exiting")
        sys.exit(1)

    # Display the generated configuration.
    logger.info("\nGenerated configuration:")
    logger.info("-" * 30)
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            logger.info(f.read())
    except Exception as e:
        logger.info(f"""Unable to read configuration file: {e}""")

    logger.info("-" * 30)

    # 2. Run training.
    logger.info("\nStep 2: Run training")
    success = run_llamafactory_train(config_path)

    if success:
        logger.info("\nAll steps completed")
    else:
        logger.info("\nConfiguration created, but training failed")
        logger.info(f"""   Run manually: llamafactory-cli train {config_path}""")


def run_llamafactory_predict(config_path):
    "Run training with llamafactory-cli train."
    try:
        # Check that the configuration file exists.
        if not os.path.exists(config_path):
            logger.info(f"""Configuration file does not exist: {config_path}""")
            return False

        # Build the command.
        cmd = ['llamafactory-cli', 'train', config_path]

        logger.info(f"""Running inference command: {' '.join(cmd)}""")
        logger.info("=" * 50)

        # Execute the command.
        result = subprocess.run(
            cmd,
            capture_output=False,  # Stream output directly to the terminal.
            text=True,
            check=False  # Inspect the return code without raising automatically.
        )

        if result.returncode == 0:
            logger.info("=" * 50)
            logger.info("Inference command completed successfully")
            return True
        else:
            logger.info("=" * 50)
            logger.info(f"""Inference command failed with return code: {result.returncode}""")
            return False

    except FileNotFoundError:
        logger.info("llamafactory-cli was not found. Check that LLaMA-Factory is installed.")
        logger.info("   Try: pip install llamafactory")
        return False
    except Exception as e:
        logger.info(f"""Inference error: {e}""")
        return False


def llm_predict(dataset_dir, eval_dataset, output_dir, yaml_file_path, model_name_or_path,
                adapter_name_or_path=None, test_max_samples=100000, per_device_eval_batch_size=16, temperature=0.6):
    "Run the configuration and execution workflow."
    logger.info("LLaMA-Factory configuration and inference script")
    logger.info("=" * 50)

    # 1. Create the YAML configuration.
    logger.info("Step 1: Create the inference YAML configuration")
    config_path = create_inference_yaml_config(dataset_dir=dataset_dir,
                                               eval_dataset=eval_dataset,
                                               output_dir=output_dir,
                                               model_name_or_path=model_name_or_path,
                                               adapter_name_or_path=adapter_name_or_path,
                                               yaml_file_path=yaml_file_path,
                                               test_max_samples=test_max_samples,
                                               per_device_eval_batch_size=per_device_eval_batch_size,
                                               temperature=temperature)

    if config_path is None:
        logger.info("Configuration creation failed; exiting")
        sys.exit(1)

    # Display the generated configuration.
    logger.info("\nGenerated configuration:")
    logger.info("-" * 30)
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            logger.info(f.read())
    except Exception as e:
        logger.info(f"""Unable to read configuration file: {e}""")

    logger.info("-" * 30)

    # 2. Run inference.
    logger.info("\nStep 2: Run inference")
    success = run_llamafactory_predict(config_path)

    if success:
        logger.info("\nAll steps completed")
    else:
        logger.info("\nConfiguration created, but inference failed")
        logger.info(f"""   Run manually: llamafactory-cli train {config_path}""")


if __name__ == "__main__":
    pass