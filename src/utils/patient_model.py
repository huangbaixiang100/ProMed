import requests
import json
from openai import OpenAI
from src.data.doctor_patient_prompts import patient_system_prompt
from src.utils.utils import call_gpt


patient_client = OpenAI(api_key="", base_url="")


def patient_answer(doctor_question, atomic_facts):
    patient_prompt = patient_system_prompt.format(atomic_facts='\n'.join(atomic_facts))
    patient_messages = [{'role': 'system', 'content': patient_prompt}]
    patient_messages.append({'role': 'user', 'content': doctor_question})
    patient_reply = call_gpt(patient_client, 'qwen2.5:72b', patient_messages)
    return patient_reply


class PatientModel:
    def __init__(self, atomic_information: str):
        self.atomic_information = atomic_information

    def get_answer(self, doctor_question: str) -> str:
        """
        Call the patient_answer function to obtain the patient's response
        """
        try:
            return patient_answer(doctor_question, self.atomic_information)
        except Exception as exc:
            return f"Error fetching patient answer: {exc}"
