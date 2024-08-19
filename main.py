import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import requests
import json
from typing import List, Dict, Any
from transformers import GPT2Tokenizer
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Node:
    def __init__(self, model_name: str, name: str, max_tokens: int = 8192):
        self.model_name = model_name
        self.name = name
        self.definition = ""
        self.context = []
        self.max_tokens = max_tokens

    def __call__(self, input_text: str, additional_data: Dict[str, Any] = None) -> str:
        try:
            context_str = "\n".join([f"<|start_header_id|>{msg['role']}<|end_header_id|> {msg['content']}<|eot_id|>" for msg in self.context])
            
            prompt = f"""<|start_header_id|>system<|end_header_id|>{self.definition}<|eot_id|>
{context_str}
<|start_header_id|>user<|end_header_id|>{input_text}<|eot_id|>"""

            if additional_data:
                prompt += "\n<|start_header_id|>system<|end_header_id|>Additional data:\n"
                for key, value in additional_data.items():
                    prompt += f"{key}: {value}\n"
                prompt += "<|eot_id|>"

            prompt += "\n<|start_header_id|>assistant<|end_header_id|>"

            response = requests.post('http://localhost:11434/api/generate', 
                                     json={
                                         "model": self.model_name,
                                         "prompt": prompt,
                                         "stream": False,
                                         "options": {
                                             "stop": ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
                                             "num_predict": self.max_tokens
                                         }
                                     })
            
            if response.status_code == 200:
                output = response.json()['response'].strip()
                self.context.append({"role": "user", "content": input_text})
                self.context.append({"role": "assistant", "content": output})
                return output
            else:
                return f"Error in Ollama API call: {response.status_code} - {response.text}"
        except Exception as e:
            return f"Error in processing: {str(e)}"

class StudentModelNode(Node):
    def __init__(self, model_name: str, name: str, max_tokens: int = 8192):
        super().__init__(model_name, name, max_tokens)
        self.definition = "I am a student language model learning to generate high-quality responses."
        
        # Hyperparameters
        self.n_embd = 768
        self.n_head = 12
        self.n_layer = 6
        self.dropout = 0.1
        self.max_seq_len = 512
        self.vocab_size = 50257  # GPT-2 tokenizer vocabulary size

        # Initialize model
        self.model = self.TransformerModel(self.vocab_size, self.n_embd, self.n_head, self.n_layer, self.dropout)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Initialize tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

        # Initialize optimizer and learning rate
        self.learning_rate = 5e-5
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # Performance tracking
        self.performance_history = []

    class TransformerModel(nn.Module):
        def __init__(self, vocab_size, n_embd, n_head, n_layer, dropout):
            super().__init__()
            self.transformer = nn.Transformer(
                d_model=n_embd,
                nhead=n_head,
                num_encoder_layers=n_layer,
                num_decoder_layers=n_layer,
                dim_feedforward=4*n_embd,
                dropout=dropout,
                batch_first=True
            )
            self.embedding = nn.Embedding(vocab_size, n_embd)
            self.fc_out = nn.Linear(n_embd, vocab_size)

        def forward(self, src, tgt):
            src_embed = self.embedding(src)
            tgt_embed = self.embedding(tgt)
            output = self.transformer(src_embed, tgt_embed)
            return self.fc_out(output)

    def __call__(self, input_text: str) -> str:
        input_ids = self.tokenize(input_text)
        with torch.no_grad():
            output_ids = self.generate(input_ids)
        return self.detokenize(output_ids)

    def tokenize(self, text: str) -> torch.Tensor:
        return torch.tensor(self.tokenizer.encode(text, truncation=True, max_length=self.max_seq_len)).unsqueeze(0).to(self.device)

    def detokenize(self, ids: torch.Tensor) -> str:
        return self.tokenizer.decode(ids.squeeze().tolist(), skip_special_tokens=True)

    def generate(self, input_ids: torch.Tensor, max_length: int = 100) -> torch.Tensor:
        self.model.eval()
        current_ids = input_ids.to(self.device)
        
        for _ in range(max_length):
            with torch.no_grad():
                output = self.model(current_ids, current_ids)
                next_token = output[:, -1, :].argmax(dim=-1).unsqueeze(1)
                current_ids = torch.cat([current_ids, next_token], dim=1)
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
        
        return current_ids

    def update(self, prompt: str, ideal_response: str, judgment: str):
        score, feedback, lr_adjustment = self.parse_judgment(judgment)

        prompt_ids = self.tokenize(prompt)
        ideal_ids = self.tokenize(ideal_response)

        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(prompt_ids, ideal_ids[:, :-1])
        loss = F.cross_entropy(output.view(-1, self.vocab_size), ideal_ids[:, 1:].view(-1))
        loss.backward()
        self.optimizer.step()

        self.adjust_learning_rate(lr_adjustment)
        self.performance_history.append(score)

    def parse_judgment(self, judgment: str) -> tuple:
        score_match = re.search(r'Score:\s*([\d.]+)', judgment)
        feedback_match = re.search(r'Feedback:\s*(.+)', judgment)
        lr_match = re.search(r'Learning Rate Adjustment:\s*(\w+)', judgment)

        score = float(score_match.group(1)) if score_match else 0.5
        feedback = feedback_match.group(1) if feedback_match else "No feedback provided"
        lr_adjustment = lr_match.group(1).lower() if lr_match else "maintain"

        return score, feedback, lr_adjustment

    def adjust_learning_rate(self, adjustment: str):
        if adjustment == "increase":
            self.learning_rate *= 1.1
        elif adjustment == "decrease":
            self.learning_rate *= 0.9
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

    def get_performance_metric(self) -> float:
        if not self.performance_history:
            return 0.0
        return sum(self.performance_history[-10:]) / min(len(self.performance_history), 10)

class AdaptiveAITrainer:
    def __init__(self):
        self.curriculum_generator = Node("llama3.1:8b", "Curriculum Generator")
        self.ideal_responder = Node("llama3.1:8b", "Ideal Responder")
        self.response_judger = Node("llama3.1:8b", "Response Judger")
        self.student_model = StudentModelNode("student", "Student Model")
        
        self.initialize_node_definitions()

    def initialize_node_definitions(self):
        self.curriculum_generator.definition = """
        You are a curriculum generator for an AI language model. Your task is to generate appropriate training data
        based on the current performance of the student model. Adjust the complexity of the data as the model improves.
        Provide a diverse range of topics and writing styles. The complexity should be challenging but not overwhelming
        for the current skill level of the student model.
        """

        self.ideal_responder.definition = """
        You are an ideal responder for an AI language model training system. Your task is to generate high-quality,
        comprehensive responses to given prompts. These responses will serve as the gold standard for evaluating
        the student model's performance. Ensure your responses are accurate, well-structured, and appropriate for
        the complexity level of the prompt.
        """

        self.response_judger.definition = """
        You are a response judge for an AI language model training system. Your task is to evaluate the student model's
        response against the ideal response. Provide a score from 0 to 1, where 1 is a perfect match and 0 is completely
        incorrect or irrelevant. Also provide specific feedback for improvement, considering factors such as relevance,
        coherence, factual accuracy, and language quality. Suggest a learning rate adjustment based on the performance.

        Your response should be in the following format:
        Score: [score between 0 and 1]
        Feedback: [your feedback here]
        Learning Rate Adjustment: [increase/decrease/maintain]
        """

    def train(self, num_iterations: int):
        for i in range(num_iterations):
            try:
                performance_metric = self.student_model.get_performance_metric()
                curriculum_prompt = f"Generate a training prompt suitable for a model with current performance: {performance_metric:.2f}"
                training_prompt = self.curriculum_generator(curriculum_prompt)

                ideal_response = self.ideal_responder(training_prompt)
                student_response = self.student_model(training_prompt)

                judging_prompt = f"""
                Training Prompt: {training_prompt}
                Ideal Response: {ideal_response}
                Student Response: {student_response}
                
                Evaluate the student model's response. Provide a score from 0 to 1 and specific feedback.
                Also suggest a learning rate adjustment (increase/decrease/maintain).
                """
                judgment = self.response_judger(judging_prompt)

                self.student_model.update(training_prompt, ideal_response, judgment)

                logging.info(f"Iteration {i+1}/{num_iterations}")
                logging.info(f"Training Prompt: {training_prompt}")
                logging.info(f"Student Response: {student_response}")
                logging.info(f"Judgment: {judgment}")
                logging.info("---")
            except Exception as e:
                logging.error(f"Error in iteration {i+1}: {str(e)}")

def main():
    logging.info("Initializing Adaptive AI Trainer...")
    trainer = AdaptiveAITrainer()
    
    logging.info("Starting training process...")
    num_iterations = 100  # You can adjust this number
    trainer.train(num_iterations)
    
    logging.info("Training completed.")

if __name__ == "__main__":
    main()
