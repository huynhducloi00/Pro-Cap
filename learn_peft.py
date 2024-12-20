from utils import set_env
set_env()
from peft import PeftConfig,PeftModel
from transformers import AutoModelForSequenceClassification,AutoTokenizer

##loading the adapter1_config and adapter2_config
saved_dire='saved_weight/12_config1_lora/checkpoint-750'
adapter1_config=PeftConfig.from_pretrained(saved_dire+'/adapter1')
adapter2_config=PeftConfig.from_pretrained(saved_dire+'/adapter2')

## Path of the save_model dire


## loading the "Pretrained" base model and "Pretrained" tokenizer
id2label={0: 'World', 1: 'Sports', 2: 'Business', 3: 'Sci/Tech'}
base_model=AutoModelForSequenceClassification.from_pretrained(saved_dire,id2label= id2label)
tokenizer=AutoTokenizer.from_pretrained(saved_dire)




print(f'adapter1_config: {adapter1_config}')
print(f'adapter2_config: {adapter2_config}')



# Load the entire model with adapters
peft_model = PeftModel.from_pretrained(base_model, saved_dire)

# Load adapter1 and adapter2
peft_model.load_adapter(saved_dire + '/adapter1', adapter_name='adapter1')
peft_model.load_adapter(saved_dire + '/adapter2', adapter_name='adapter2')


import torch.nn.functional as F

def classify(peft_model,text, adapter_name: str):
    # Set the adapter
    peft_model.set_adapter(adapter_name)
    # Tokenize the input text
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    # Get the model's output
    output = peft_model(**inputs)
    # Get the predicted class and confidence
    probabilities = F.softmax(output.logits, dim=-1)
    prediction = probabilities.argmax(dim=-1).item()
    confidence = probabilities[0, prediction].item()
    print(f'Adapter: {adapter_name} | Text: {text} | Class: {prediction} | Label: {id2label[prediction]} | Confidence: {confidence:.2%}')




text1="Kederis proclaims innocence Olympic champion Kostas Kederis today left hospital ahead of his date with IOC inquisitors claiming his ..."
text2="Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall Street's dwindling\band of ultra-cynics, are seeing green again."



classify(peft_model,text1,'adapter1')
classify(peft_model,text1,'adapter2')


classify(text2,'adapter1') ## both correction are wrong 'trained on small dataset so
classify(text2,'adapter2')
