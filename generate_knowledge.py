from src.data_utils import load_json, format_gpt2_data
import tqdm
from openai import OpenAI
from tqdm import tqdm
import pickle

client = OpenAI(
    # defaults to os.environ.get("OPENAI_API_KEY")
    api_key="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",
)

input_fname = "data/ecbd/all_ent_2020_2021_np_500samples.json"
ecbd_data = load_json(input_fname)
ecbd_defs = [format_gpt2_data(ex, pad_token='</s>')['definition'] for ex in ecbd_data]

def format_prompt(definition):
    return f"Carefully read the provided sentence; this is a sentence from Wikipedia.\n\n{definition}\n\n\
Now, using the provided sentence as a template, replace the name of the entities(person, country, act, etc.)with other entities \
to create a paraphrased, and extended description on a completely different event, that have never happend in history in a paragraph. \
However, your sentence should sound enough to make someone believe that it could have actually happened.\n\n\
After generating the description, please create a probe sentence in a predict-the-next-word format that asks for the last 1 to 3 words \
denoted as a single underscore which is always placed at the end of the sentence. \
To fill in the blank in the provided probe sentence, one should have a knowledge directly related to the presented description. \
The answer in the blank should consist of 1 to 3 words, either a noun or a number. Also, ensure that the probe task is properly paraphrased to avoid \
significant lexical overlap between the description and the probe. Answer in the following format:\n\n\
Description: [YOUR_DESCRIPTION]\n\
Probe: [YOUR_PROBE_ENDS_WITH_AN_UNDERSCORE]\n\
Answer: [YOUR_ANSWER_TO_THE_PROBE]\n\n"
           
           
def gpt4(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
           
 
responses = []
                   
for idx, definition in enumerate(tqdm(ecbd_defs)):
    prompt = format_prompt(definition)
    try:
        response = gpt4(prompt)
        responses.append(response)
    except:
        print('!!!!!!!!!!!!!!!!!!!!!!!!')
        continue
    
with open("generated_knowledge_500.pkl", "wb") as f:
    pickle.dump(responses, f)