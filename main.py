import os
import pickle
import torch
import psutil
import gc
import pandas as pd
import re
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv
from cw import common_words
import time
from transformers import AdamW, get_linear_schedule_with_warmup
class ModelManager:
    def __init__(self, model_name, device):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.bucket_name = "finetuned-model-bucket"
        self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        try:
            if os.path.exists(self.bucket_name):
                print(f"Model already exists at {self.bucket_name}. Loading from there.")
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
                # model = T5ForConditionalGeneration.from_pretrained("t5-small")  # Use the base model as a reference
                self.model.load_state_dict(torch.load("finetuned-model-bucket/pytorch_model.bin"))
                # self.tokenizer = T5Tokenizer.from_pretrained(self.bucket_name)
                tokenizer_path = f"{self.bucket_name}/tokenizer.pkl"
                with open(tokenizer_path, "rb") as tokenizer_file:
                    self.tokenizer = pickle.load(tokenizer_file)
                print(f"Tokenizer loaded from {tokenizer_path}")
            else:
                print(f"Downloaidng {self.model_name} model...")
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
                print(f"Downloading {self.model_name} tokenizer...")
                self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
                print("Model and Tokenizer Downloaded successfully.")
        except Exception as e:
            print("Error loading model/tokenizer:", e)
            raise e


class FineTuner:
    def __init__(self, model_manager, max_token_size, batch_size, output_dir, device):
        self.model_manager = model_manager
        self.max_token_size = max_token_size
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.training_args = None
        self.trainer = None
        self.current_position = 0
        self.iteration = 0
        self.device = device
        self.load_iteration()
        
    def check_missing_file(self):
        CHECKPOINT_PATH = "checkpoint.pth"
        ITERATION_PATH = "iteration.pkl"
        checkpoint_exists = os.path.exists(CHECKPOINT_PATH)
        iteration_exists = os.path.exists(ITERATION_PATH)
        if checkpoint_exists and not iteration_exists:
            raise ValueError("Warning: iteration.pkl is missing. May make conflicts so  Either delete Both or store both,")
        elif iteration_exists and not checkpoint_exists:
            raise ValueError("Warning: checkpoint.pkl is missing. May make conflicts so  Either delete Both or store both.")

    def save_iteration(self):
        iteration = {"iteration": self.iteration}
        with open("iteration.pkl", "wb") as f:
            pickle.dump(iteration, f)       
        print(f"✅ Checkpoint saved at iteration {self.iteration}")


    def load_iteration(self):
        iteration_path = ('iteration.pkl')
        self.check_missing_file()
        
        if os.path.exists(iteration_path):
            with open(iteration_path, 'rb') as f:
                iteration = pickle.load(f)
                self.iteration = iteration["iteration"]

                print(f"🔄 Resumed training from iteration {self.iteration}")
        else:
            print("⚠️ No iteration found. Starting fresh.")      


    def save_checkpoint(self):
        """Saves the model state, optimizer, and scheduler into a single file."""
        checkpoint = {
            
            "model_state_dict": self.model_manager.model.state_dict(),
            "optimizer_state_dict": self.trainer.optimizer.state_dict(),
            "scheduler_state_dict": self.trainer.lr_scheduler.state_dict(),
            "tokenizer": self.model_manager.tokenizer,
            "training_params": {
                "batch_size": self.batch_size,
                "max_token": self.max_token_size
            }
        }
        # Save checkpoint to disk
        checkpoint_file = "checkpoint.pth"
        torch.save(checkpoint, checkpoint_file)
        print(f"Checkpoint saved to {checkpoint_file}")
    

    def load_checkpoint(self):
        """Loads the model state, optimizer, scheduler, and training parameters from the checkpoint file."""
        checkpoint_path = ('checkpoint.pth')
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load training parameters
            saved_params = checkpoint["training_params"]
            if saved_params != { "batch_size": self.batch_size, "max_token": self.max_token_size}:
                print("Warning: The training parameters have changed!")
                print(f"Saved parameters: {saved_params}")
                print(f"Current parameters: {self.batch_size}, {self.max_token_size}")
                # Ask the user whether they want to revert or start fresh
                choice = input("Press 'r' to revert to saved parameters or press Enter to start fresh: ").strip().lower()
                if choice == 'r':
                    self.batch_size = saved_params["batch_size"]
                    self.max_token_size = saved_params["max_token"]
                    print("Reverted to saved parameters.")
                else:
                    print("Starting fresh. Deleting saved state.")
                    os.remove(checkpoint_path)
                    iteration_path = "iteration.pkl"
                    os.remove(iteration_path)
                    self.iteration = 0
                    
                    print("State deleted. Proceeding with fresh start.")    

            print("Training iteration:", self.iteration)
            tokenized_dataset = self.load_next_chunk()
            if tokenized_dataset is None:
                print("No more data to process.")
                    
            self.model_manager.model.load_state_dict(checkpoint["model_state_dict"])


            
            # Initialize training arguments only once
            self.training_args = Seq2SeqTrainingArguments(
                output_dir=self.output_dir,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                predict_with_generate=True,
                logging_dir=f"{self.output_dir}/logs",
                logging_strategy="steps",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=2,
                load_best_model_at_end=True,
            )

            # Initialize trainer if not already initialized
            print("trainer")
            if self.trainer is None:
                self.trainer = Seq2SeqTrainer(
                    model=self.model_manager.model,
                    args=self.training_args,
                    train_dataset=tokenized_dataset["train"],
                    eval_dataset=tokenized_dataset["test"],
                    # optimizers=(optimizer, lr_scheduler),
                )
            # Load model
            # Now, self.trainer.optimizer and self.trainer.lr_scheduler should be initialized
            
            
            
           
            if "optimizer_state_dict" in checkpoint:
                self.trainer.create_optimizer_and_scheduler(num_training_steps=100)  # Create optimizer and scheduler
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("Optimizer state loaded successfully.")

            if "scheduler_state_dict" in checkpoint:
                self.trainer.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                print("LR Scheduler state loaded successfully.")
            # Load tokenizer
            self.model_manager.tokenizer = checkpoint["tokenizer"]

            

            
        
    def save_model_and_tokenizer(self, model, tokenizer, output_dir):
        # Save model's state_dict
        model_path = f"{output_dir}/pytorch_model.bin"
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        # Save tokenizer using pickle
        tokenizer_path = f"{output_dir}/tokenizer.pkl"
        with open(tokenizer_path, "wb") as tokenizer_file:
            pickle.dump(tokenizer, tokenizer_file)
            print(f"Tokenizer saved to {tokenizer_path}")

    def preprocess_function(self, sample, padding="max_length"):
        inputs = ["summarize: " + item for item in sample["question"]]  # Use the 'question' column
        model_inputs = self.model_manager.tokenizer(inputs, max_length=self.max_token_size, padding=padding, truncation=True)
        labels = self.model_manager.tokenizer(sample["answer"], max_length=self.max_token_size, padding=padding, truncation=True)
        if padding == "max_length":
            labels["input_ids"] = [[(l if l != self.model_manager.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    def apply_keyword_filter(self, input_string):
        try:
            cleaned_string = re.sub(r'[^a-zA-Z]', ' ', input_string)
            cleaned_string = re.sub(r'\s+', ' ', cleaned_string).strip()
            filtered_words = " ".join([word for word in cleaned_string.split() if word.lower() not in common_words])
            return self.optimize_token_length(filtered_words)
        except Exception as e:
            print(e, "Error applying keyword filter")

    def update_dataframe_column(self, df: pd.DataFrame, column_name: str, func) -> pd.DataFrame:
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the DataFrame.")
        df[column_name] = df[column_name].apply(func)
        return df
    def optimize_token_length(self, text):
        try:
            words = self.remove_duplicated_words(text).split()
            char_threshold = 2
            while len(" ".join(words)) > self.max_token_size:
                for i in range(len(words) - 1, -1, -1):
                    if len(words[i]) > char_threshold:
                        words.pop(i)
                        break
                else:
                    char_threshold += 1
            return " ".join(words)
        except Exception as e:
            print(e, "Error optimizing token length")
    def remove_duplicated_words(self, input_string):
        try:
            words = input_string.split()
            seen = set()
            return ' '.join([word for word in words if not (word in seen or seen.add(word))])
        except Exception as e:
            print(e, "Error removing duplicated words")

    def load_next_chunk(self):
        # Load next chunk of data, process, and return tokenized dataset
        print("Loading next chunk...")
        max_chunsize = int(os.getenv("DATASET_LOAD_CHUNKSIZE"))
        df = pd.read_csv("training_dataset.csv", skiprows=range(1, self.current_position + 1), nrows=max_chunsize)
        if df.empty:
            return None
        else:
            updated_df = self.update_dataframe_column(df, 'input', self.apply_keyword_filter)
            updated_df['instruction'] = updated_df['instruction'].str.replace(r"[']", '', regex=True) + "here is the content: "
            updated_df['input'] = updated_df['input'].str.replace(r"[']", '', regex=True)
            updated_df['question'] = updated_df['instruction'] + ' ' + updated_df['input']
            updated_df['answer'] = updated_df['output'].str.replace(r"[']", '', regex=True)
            train_data, test_data = train_test_split(updated_df, test_size=0.03, random_state=42)
            train_dataset = Dataset.from_pandas(train_data)
            test_dataset = Dataset.from_pandas(test_data)
            dataset = DatasetDict({
                "train": train_dataset,
                "test": test_dataset
            })
            tokenized_dataset = dataset.map(self.preprocess_function, batched=True, remove_columns=["instruction", "input", "output"])
            self.current_position += max_chunsize
            return tokenized_dataset

    def fine_tune(self):
        checkpoint_path = "checkpoint.pth"

        
        from accelerate import Accelerator
        self.accelerator = Accelerator()

        #Check if checkpoint exists and load it
        if os.path.exists(checkpoint_path):
            self.load_checkpoint()
        else:
            tokenized_dataset = self.load_next_chunk()
            if tokenized_dataset is None:
                print("No data available. Exiting.")
                
            # Initialize training arguments once
            self.training_args = Seq2SeqTrainingArguments(
                output_dir=self.output_dir,
                per_device_train_batch_size=self.batch_size,
                per_device_eval_batch_size=self.batch_size,
                predict_with_generate=True,
                logging_dir=f"{self.output_dir}/logs",
                logging_strategy="steps",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=2,
                load_best_model_at_end=True,
            )

            # ✅ Initialize Trainer once
            self.trainer = Seq2SeqTrainer(
                model=self.model_manager.model,
                args=self.training_args,
                train_dataset=tokenized_dataset["train"],  # Will update later
                eval_dataset=tokenized_dataset["test"],   # Will update later
            )


        while True:
            print("Training iteration:", self.iteration)
            tokenized_dataset = self.load_next_chunk()

            if tokenized_dataset is None:
                print("No more data to process.")
                break

            # ✅ Update datasets dynamically instead of reinitializing trainer
            self.trainer.train_dataset = tokenized_dataset["train"]
            self.trainer.eval_dataset = tokenized_dataset["test"]

            print("Starting training...")
            self.trainer.train()
            self.iteration += 1
            gc.collect()
            print("Training complete for this iteration.")
 
            # ✅ Save model and checkpoint periodically
            time.sleep(3)
            self.save_model_and_tokenizer(self.model_manager.model, self.model_manager.tokenizer, self.output_dir)
            

            self.save_checkpoint()
            self.save_iteration()
            break

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    attempts = 0
    MAX_RETRIES = 3
    while attempts < MAX_RETRIES:
        try:
            # Determine device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_name = os.getenv("model_name")

            model_manager = ModelManager(model_name=model_name, device=device)
            fine_tuner = FineTuner(
                model_manager=model_manager,
                max_token_size=int(os.getenv("MAX_TOKEN_SIZE_FOR_TRAINING")),
                batch_size=2,
                output_dir=os.getenv("model_bucket_name"),
                device = device
            )
            fine_tuner.fine_tune()
            print("Fine-tuning completed successfully.")
            break  # Exit the loop if successful
        except Exception as e:
            print(f"\n⚠️ Error: {e}")
            print(f"Attempt {attempts + 1} of {MAX_RETRIES}")
            print("🔧 Please resolve the issue and press Enter to retry...")

            input()  # Wait for user to fix the issue before retrying

            attempts += 1
            continue

    if attempts == MAX_RETRIES:
        print("❌ Maximum retry attempts reached. Please check the errors and restart the process manually.")