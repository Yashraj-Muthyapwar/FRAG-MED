from transformers import AutoTokenizer, AutoModel

# 1. Load it first 
model = AutoModel.from_pretrained("neuml/pubmedbert-base-embeddings") 
tokenizer = AutoTokenizer.from_pretrained("neuml/pubmedbert-base-embeddings") 

# 2. Save it to your specific directory 
output_dir = "./models" 
model.save_pretrained(output_dir) 
tokenizer.save_pretrained(output_dir)
