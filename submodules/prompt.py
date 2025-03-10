import torch
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer
import random
# random.seed(42)
#prompt generation

def generate_prompt_batch(text_prompts, tokenizer, text_encoder):
    # Process in batches of 32 to manage memory
    batch_size = 1000
    num_prompts = len(text_prompts)
    all_embeddings = []
    
    with torch.no_grad():
        for i in range(0, num_prompts, batch_size):
            batch_prompts = text_prompts[i:i + batch_size]
            
            # Tokenize the batch of text
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=tokenizer.model_max_length
            ).to('cuda')
            
            # Generate embeddings for batch
            text_embeddings = text_encoder(**inputs).last_hidden_state  # [batch_size, seq_len, hidden_dim]
            pooled_embeddings = text_embeddings.mean(dim=1)  # [batch_size, hidden_dim]
            pooled_embeddings = F.normalize(pooled_embeddings.unsqueeze(-1), dim=1, eps=1e-16)
            
            all_embeddings.append(pooled_embeddings)
            
            # Clean up batch tensors
            del inputs, text_embeddings
            torch.cuda.empty_cache()
    
    # Concatenate all batches
    final_embeddings = torch.cat(all_embeddings, dim=0)
    
    # Clean up
    del tokenizer, text_encoder, all_embeddings
    torch.cuda.empty_cache()
    
    return final_embeddings.clone().detach()

class prompt_generation:
    def __init__(self, key_prompt):
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        self.key_prompt = key_prompt
        self.key_prompt_embeddings = self.generate_prompt(self.key_prompt)

   
    def generate_prompt(self, text_prompt):

        # Tokenize the text
        inputs = self.tokenizer(text_prompt, return_tensors="pt", padding="max_length", truncation=True, max_length=self.tokenizer.model_max_length)

        # Generate the text embedding
        with torch.no_grad():
            text_embeddings = self.text_encoder(**inputs).last_hidden_state # Example: (1, 77, 768)
        
        pooled_embeddings = text_embeddings.mean(dim=1) #1,768
        pooled_embeddings = F.normalize(pooled_embeddings.unsqueeze(-1), dim=1, eps=1e-16)
        
        return pooled_embeddings.clone().detach()

    def generate_randomtext(self):

        # Generate a random sequence of token IDs
        # CLIP's vocabulary size is 49408
        num_tokens = random.randint(5, 10) #tocken 15-30
        random_token_ids = torch.randint(0, self.tokenizer.vocab_size, (num_tokens,)).tolist()
        random_text = self.tokenizer.decode(random_token_ids, skip_special_tokens=True)
        # Decode the random token IDs to text
        while F.cosine_similarity(self.generate_prompt(random_text), self.key_prompt_embeddings) > 0.9:
            random_token_ids = torch.randint(0, self.tokenizer.vocab_size, (num_tokens,)).tolist()
            random_text = self.tokenizer.decode(random_token_ids, skip_special_tokens=True)

        # print("Random Text:", random_text)
        return random_text
    
    def generate_randomprompt(self):

        # Generate a random sequence of token IDs
        # CLIP's vocabulary size is 49408
        num_tokens = random.randint(15, 30) #tocken 15-30
        random_token_ids = torch.randint(0, self.tokenizer.vocab_size, (num_tokens,)).tolist()
        eos_token_id = self.tokenizer.eos_token_id
        padded_token_ids = random_token_ids + [eos_token_id] * (77 - len(random_token_ids))
        token_ids = torch.tensor([padded_token_ids])
        inputs = {'input_ids': token_ids}
        #  Get the text embeddings using the text encoder
        with torch.no_grad():
            text_embeddings =self.text_encoder(**inputs).last_hidden_state
        pooled_embeddings = text_embeddings.mean(dim=1) #1,768
        pooled_embeddings = F.normalize(pooled_embeddings.unsqueeze(-1), dim=1, eps=1e-16)
        
        return pooled_embeddings.clone().detach()
