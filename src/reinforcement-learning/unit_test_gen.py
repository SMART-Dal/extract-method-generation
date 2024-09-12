# p_b = r6uBvP00MmfTxUUzuveYRg%3D%3D
# p_lat = iE3X89MyAHUpbEcaaMSlla3G%2Bhes8SifT9OaxvT3RA%3D%3D

# form = c534aa3a16b1de06cf71db8c5ca2f1c4aa
import json
from tqdm import tqdm
from openai import OpenAI

open_ai_api_key = ""
openai_client = OpenAI(api_key=open_ai_api_key, organization="", project="")

def get_msgs_for_batch(smelly_method, smelly_class):

    messages=[
        {
            "role": "system",
            "content": "You are a helpful software engineer assistant. Your task is to create test cases for given methods. Don't forget to include edge cases. Don't output anything else than the test cases. "
        },
        {
            "role": "user",
            "content": f"Write Unit Test case for the following method:\n\n{smelly_method}\n\n. The method is in the following class:\n\n{smelly_class}"
        },
        {
            "role": "user",
            "content": "Make sure to output STRICTLY the test cases. Do not output anything else."
        }
    ]

    return messages

def test_openai():
    # response = openai.complete("What is the capital of the United States?", model="davinci")
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful software engineer assistant. Your task is to create test cases for given methods. Don't forget to include edge cases. Don't output anything else than the test cases. "
            },
            {
                "role": "user",
                "content": "Write Unit Test case for "
            }
        ]
    )
    print(response.choices[0].message.content)

def generate_test_cases():
    with open(r"C:\Users\indra\Documents\Playground\Misc\Mis cosas\Output\aws-dynamo-db.jsonl", "+r") as f:
        data = f.readlines()
        for idx, obj in tqdm(enumerate(data)):
            dt = json.loads(obj)
            msgs = get_msgs_for_batch(dt["Smelly Sample"], dt["Smelly Class"])
            batch_json_obj = {"custom_id": f"request-{idx}", "method": "POST", "url": "/v1/chat/completions", "body": {"model": "gpt-3.5-turbo-0125", "messages": msgs}}
            with open(f"open_ai_batch.jsonl", "+a") as fp:
                fp.write(json.dumps(batch_json_obj))
                fp.write("\n")

def open_ai_batch():

    batch_input_file = openai_client.files.create(
                        file=open("open_ai_batch.jsonl", "rb"),
                        purpose="batch"
                        )
    batch_input_file_id = batch_input_file.id

    batch_meta_data = openai_client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={
        "description": "test generation"
        }
    )

    print(batch_meta_data)
    print(batch_meta_data.id)

    return

def batch_status(batch_id):
    batch = openai_client.batches.retrieve(batch_id)
    print(batch)
    print(batch.status)

# def batch_results(batch_id):

def openai_batch_results(file_id):
    file_response = openai_client.files.content(file_id)
    # print(file_response.text)
    with open("open_ai_batch_results.jsonl", "+w") as f:
        f.write(file_response.text)


def get_open_ai_result():
    with open("open_ai_batch_results.jsonl", "+r") as f:
        data = f.readlines()
        for idx, obj in enumerate(data):
            dt = json.loads(obj)
            print(dt["custom_id"])
            print(dt["response"]["body"]["choices"][0]["message"]["content"])
            break

if __name__ == "__main__":
    # test_openai()
    # generate_test_cases()
    # open_ai_batch()
    # batch_MiOX63Gi1IzghevTt5s25Nng
    # batch_status("batch_MiOX63Gi1IzghevTt5s25Nng")
    # openai_batch_results('file-PTORIb9F2HZUobzhqh30GjBv')
    # get_open_ai_result()
    with open ("aws-dynamo-db.jsonl", "r") as f:
        data = f.readlines()
        for idx, obj in enumerate(data):
            dt = json.loads(obj)
            print(dt["Smelly Sample"])
            print(dt["Smelly Class"])
            break



