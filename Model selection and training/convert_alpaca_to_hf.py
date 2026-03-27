import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

def main():
    # Define the paths to your JSONL files
    train_path = r"C:\Users\dell\Downloads\train.jsonl"
    eval_path = r"C:\Users\dell\Downloads\eval.jsonl"

    # Ensure the files exist
    if not os.path.exists(train_path):
        print(f"Error: Train file not found at {train_path}")
        return
    if not os.path.exists(eval_path):
        print(f"Error: Eval file not found at {eval_path}")
        return

    # Define the data files mapping
    data_files = {
        "train": train_path,
        "validation": eval_path
    }

    print("Loading dataset...")
    # Load dataset using Hugging Face datasets library
    dataset = load_dataset("json", data_files=data_files)

    # Print dataset structure to verify
    print("\nDataset successfully loaded!")
    print(dataset)
    
    # Example of accessing an item
    if len(dataset["train"]) > 0:
        print("\nExample item from train split:")
        print(dataset["train"][0])

    # Optional: Save it to disk using Hugging Face format
    save_path = r"C:\Users\dell\Downloads\hf_dataset"
    dataset.save_to_disk(save_path)
    print(f"\nSaved dataset to {save_path}")
    
    return dataset

def setup_model(model_id: str, quantization_enabled: bool = True, peft_enabled: bool = True):
    print("\n--- MODEL LOADING PIPELINE ---")
    print(f"Base Model: HuggingFace Hub ({model_id})")
    
    # 1. Quantization Check
    if quantization_enabled:
        print("-> Quantization Enabled? Yes -> QLoRA")
        # BitsAndBytesConfig 4-bit NF4
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        print("-> AutoModelForCausalLM (Quantized)")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
    else:
        print("-> Quantization Enabled? No -> LoRA / Full Precision Model")
        print("-> AutoModelForCausalLM (Full Precision)")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto"
        )

    # 2. PEFT Check
    if peft_enabled:
        print("-> PEFT Enabled? Yes")
        print("-> LoraConfig + get_peft_model")
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    else:
        print("-> PEFT Enabled? No -> Full Model Training")
    
    return model

def setup_training(model, tokenizer, dataset, training_method="SFT", output_dir="./results"):
    print("\n--- TRAINING PIPELINE ---")
    print(f"Training Method selected: {training_method}")
    
    if training_method == "SFT":
        from trl import SFTTrainer, SFTConfig
        print("-> Initializing SFTTrainer...")
        
        training_args = SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            logging_steps=10,
            max_length=512,
            dataset_text_field="text", # Adjust based on your dataset format
            #use_cpu=True
        )
        
        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation", None),
            args=training_args,
            processing_class=tokenizer,
        )
        
    elif training_method == "DPO":
        from trl import DPOTrainer, DPOConfig
        print("-> Initializing DPOTrainer...")
        
        training_args = DPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=5e-5,
            logging_steps=10,
            beta=0.1,
        )
        
        # Note: DPO requires 'prompt', 'chosen', and 'rejected' columns in the dataset!
        trainer = DPOTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation", None),
            args=training_args,
            tokenizer=tokenizer,
        )
        
    elif training_method == "ORPO":
        from trl import ORPOTrainer, ORPOConfig
        print("-> Initializing ORPOTrainer...")
        
        training_args = ORPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=8,
            learning_rate=5e-5,
            logging_steps=10,
            beta=0.1,
        )
        
        # Note: ORPO requires 'prompt', 'chosen', and 'rejected' columns
        trainer = ORPOTrainer(
            model=model,
            train_dataset=dataset["train"],
            eval_dataset=dataset.get("validation", None),
            tokenizer=tokenizer,
            args=training_args,
        )
        
    else:
        raise ValueError("Invalid training method. Choose SFT, DPO, or ORPO.")
        
    return trainer


def run_test():
    dataset = main()
    if not dataset:
        print("Dataset failed to load. Halting test.")
        return

    print("\n--- PREPARING TEST ENV ---")
    # Tiny model for fast sanity testing
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Map Alpaca formatting into a single 'text' column for SFTTrainer
    print("Formatting dataset to SFT expectations...")
    def map_to_text(example):
        if example.get('input'):
            text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        else:
            text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        return {"text": text}
        
    dataset = dataset.map(map_to_text)
    
    # 1. Setup Model
    model = setup_model(model_id, quantization_enabled=True, peft_enabled=True)
    
    # 2. Setup Training
    trainer = setup_training(model, tokenizer, dataset, training_method="SFT", output_dir="./test_results")
    
    # Override max_steps for a quick test
    trainer.args.max_steps = 1
    
    print("\nStarting a test training run (1 step only)...")
    trainer.train()
    
    print("\n--- TEST COMPLETED SUCCESSFULLY ---")

if __name__ == "__main__":
    run_test()
