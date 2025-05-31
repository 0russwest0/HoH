from wiki_processor import WikiDiffProcessor
import yaml
from pathlib import Path
import argparse

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def parse_args():
    parser = argparse.ArgumentParser(description='Wiki diff extraction and processing')
    
    # 配置文件路径
    parser.add_argument('--config', type=str, default='config.yaml',
                      help='Path to config file')
    
    # 常用参数，可以通过命令行覆盖配置文件中的值
    parser.add_argument('--model-path', type=str,
                      help='Path to the model')
    parser.add_argument('--cuda-devices', type=str,
                      help='CUDA devices to use (e.g. "0,1")')
    parser.add_argument('--batch-size', type=int,
                      help='Batch size for model inference')
    parser.add_argument('--min-text-length', type=int,
                      help='Minimum text length for filtering')
    
    # 添加跳过处理步骤的参数
    parser.add_argument('--skip-process-old', action='store_true',
                      help='Skip old wiki processing')
    
    return parser.parse_args()

def update_config_with_args(config, args):
    """使用命令行参数更新配置"""
    
    # 更新模型配置
    if args.model_path:
        config['model']['path'] = args.model_path
    if args.cuda_devices:
        config['model']['cuda_devices'] = args.cuda_devices
    if args.batch_size:
        config['model']['batch_size'] = args.batch_size
        
    # 更新处理配置
    if args.min_text_length:
        config['processing']['min_text_length'] = args.min_text_length
    
    return config

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 加载配置文件
    config_path = Path(__file__).parent / args.config
    config = load_config(config_path)
    
    # 使用命令行参数更新配置
    config = update_config_with_args(config, args)
    
    processor = WikiDiffProcessor(config)
    
    # Process raw data
    if not args.skip_process_old:
        processor.process_raw_data(
            config['data']['raw_old_data_path'], 
            config['data']['old_wiki_path']
        )
    processor.process_raw_data(
        config['data']['raw_new_data_path'], 
        config['data']['new_wiki_path']
    )
    
    # Extract diffs
    processor.extract_diffs(
        config['data']['old_wiki_path'],
        config['data']['new_wiki_path'],
        config['data']['diff_path']
    )
    
    # Filter diffs
    processor.filter_diffs(
        config['data']['diff_path'],
        config['data']['filtered_diff_path']
    )
    
    # Apply model filter
    processor.model_filter(
        config['data']['filtered_diff_path'],
        config['data']['final_diff_path'],
        config['model']['path']
    )
    
    # Add passages
    processor.add_passage(
        config['data']['final_diff_path'],
        config['data']['final_diff_path'].replace('.jsonl', '_text.jsonl'),
        config['data']['old_wiki_path'],
        config['data']['new_wiki_path']
    )

if __name__ == "__main__":
    main() 