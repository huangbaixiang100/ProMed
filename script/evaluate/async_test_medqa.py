from call_llm import call_gpt
from openai import OpenAI
import os
import json
import random
from multiprocessing import Pool, Value, Lock, Manager
from tqdm import tqdm
import re
from collections import defaultdict
from prompts import *
import numpy as np
import time


patient_client = OpenAI(api_key="",base_url="")
patient_model='qwen2.5:72b'
doctor_client = OpenAI(api_key="EMPTY", base_url="")
doctor_model = 'llama3_1_8b'  # Use the locally deployed model name


# Define the function to process data
def process_data(data):
    max_retries = 3
    retry_delay = 2  
    
    if isinstance(data['context'], list) and len(data['context']) > 0:
        initial_info = data['context'][0]
    elif isinstance(data['context'], str):
        # Assuming sentences are separated by periods, taking the first sentence
        initial_info = data['context'].split(". ")[0]
    else:
        initial_info = ""  # Default fallback
    partial_question = initial_info +'\n' +data['question']
    option_str = "\n".join([f"{key}: {value}" for key, value in data['options'].items()])
    doctor_prompt = doctor_system_prompt_en.format(question_type='multiple choice question', question=partial_question,
                                                option_str=option_str)

    patient_prompt = patient_system_prompt_en.format(atomic_facts='\n'.join(data['atomic_facts']))

    doctor_messages = [{'role': 'user', 'content': doctor_prompt}]
    patient_messages = [{'role': 'system', 'content': patient_prompt}]
    flag = 0

    for i in range(10):
        retry_count = 0
        while retry_count < max_retries:
            try:
                if retry_count > 0:
                    time.sleep(retry_delay)
                
                doctor_question = call_gpt(doctor_client, doctor_model, doctor_messages)  # 同步调用
                
                if not doctor_question:
                    raise Exception("Empty response")
                
                if '!model error:' in doctor_question:
                    print(f"Doctor model attempt {retry_count + 1}/{max_retries}: {doctor_question}")
                    retry_count += 1
                    continue
                
                
                break
                
            except Exception as e:
                print(f"Doctor model attempt {retry_count + 1}/{max_retries}: Error - {str(e)}")
                retry_count += 1
        
      
        if retry_count >= max_retries:
            print("All doctor model retry attempts failed")
            flag = 1
            break

        doctor_messages.append({'role': 'assistant', 'content': doctor_question})
        if 'answer:' in doctor_question:
            data['final_answer'] = doctor_question
            break
        patient_messages.append({'role': 'user', 'content': doctor_question})
        retry_count = 0
        while retry_count < max_retries:
            try:
                if retry_count > 0:
                    time.sleep(retry_delay)
                
                patient_reply = call_gpt(patient_client, patient_model, patient_messages)  # 同步调用
                
                if not patient_reply:
                    raise Exception("Empty response")
                
                if '!model error' in patient_reply:
                    print(f"Patient model attempt {retry_count + 1}/{max_retries}: {patient_reply}")
                    retry_count += 1
                    continue
                
                break
                
            except Exception as e:
                print(f"Patient model attempt {retry_count + 1}/{max_retries}: Error - {str(e)}")
                retry_count += 1
        
        if retry_count >= max_retries:
            print("All patient model retry attempts failed")
            flag = 1
            break
        doctor_messages.append({'role': 'user', 'content': patient_reply})
        patient_messages.append({'role': 'assistant', 'content': patient_reply})

    if "final_answer" not in data.keys():
        data["final_answer"] = ""
        if flag:
            data["final_answer"] = "error"

    data["dialogue"] = doctor_messages

    return data


def worker(args):
    data, out_path, lock = args

    result = process_data(data)

    if result['final_answer']=='error':
        return False
    with lock:
        with open(out_path, 'a', encoding='utf-8') as outfile:
            outfile.write(json.dumps(result, ensure_ascii=False) + '\n')
    return True


if __name__ == "__main__":
    out_path = "medqa_result/llama3_1_8b_medqa.jsonl" 

    datas=[]
    with open("dataset/medqa_test_convo.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            datas.append(json.loads(line))

    test_datas = []
    continue_id = set()
    if os.path.exists(out_path):
        with open(out_path, "r+", encoding='utf-8') as f:
            out_data = f.readlines()
            for line in out_data:
                line = json.loads(line)
                continue_id.add(line['id'])

    for data in datas:
        if data['id'] in continue_id:
            continue
        else:
            test_datas.append(data)

    manager = Manager()
    lock = manager.Lock()

    
    with Pool(5) as p:
        for result in tqdm(p.imap(worker, [(data, out_path, lock) for data in test_datas], chunksize=1),
                          total=len(test_datas)):
            if not result:
                print("Error in processing data")
                with open('log.txt', 'a') as log_file:
                    log_file.write("Error during processing\n")