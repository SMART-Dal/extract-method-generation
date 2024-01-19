from transformers import T5ForConditionalGeneration, AutoTokenizer

# Load the model
model = T5ForConditionalGeneration.from_pretrained("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/tmp_trainer")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("/home/ip1102/projects/def-tusharma/ip1102/Ref_RL/POC/extract-method-generation/tmp_trainer")

# Encode the input text
# input_text = "    public static void showProperty(String key) {\n        System.out.println(key + \": \" + System.getProperty(key));\n    }\n"
input_text = "        public String getResourceFileName(final int i) {\n            return String.format(\"META-INF/crest/%s/%s.%s.properties\", clazzName, commandName, i);\n        }\n"
encoded_input = tokenizer.encode(input_text, return_tensors='pt')

# Generate prediction
output = model.generate(encoded_input)
print(output)
# Decode the output
# decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)

# print(decoded_output)